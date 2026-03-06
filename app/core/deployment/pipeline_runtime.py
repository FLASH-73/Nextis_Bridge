"""Multi-step pipeline runtime: sequences policies with transition conditions."""
import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .observation_builder import ObservationBuilder
from .pipeline_types import (
    AlignmentWarning,
    PipelineConfig,
    PipelineState,
    PipelineStatus,
    TransitionProgress,
    TransitionTrigger,
)
from .types import DeploymentConfig, DeploymentMode

logger = logging.getLogger(__name__)
_GRIPPER_KEYS = ("gripper.pos", "left_gripper.pos", "right_gripper.pos")


class PipelineRuntime:
    def __init__(self, deploy, training_service):
        self._deploy, self._training = deploy, training_service
        self._state, self._lock = PipelineState.IDLE, threading.Lock()
        self._stop_event, self._manual_trigger = threading.Event(), threading.Event()
        self._loaded: Dict[int, Tuple] = {}
        self._config: Optional[PipelineConfig] = None
        self._step, self._step_start_time = 0, 0.0
        self._total_frames, self._start_time = 0, 0.0
        self._debounce_count, self._error_message = 0, ""
        self._monitor_thread: Optional[threading.Thread] = None

    def load(self, config: PipelineConfig) -> List[AlignmentWarning]:
        if not config.steps:
            raise ValueError("Pipeline must have at least 1 step")
        for i, s in enumerate(config.steps):
            if not s.policy_id:
                raise ValueError(f"Step {i} ('{s.name}') has empty policy_id")
        with self._lock:
            self._state = PipelineState.LOADING
        self._loaded.clear()
        self._error_message = ""
        try:
            for i, s in enumerate(config.steps):
                self._loaded[i] = self._load_step_policy(s.policy_id)
                logger.info("Pipeline: loaded step %d '%s'", i, s.name)
        except Exception as e:
            with self._lock:
                self._state, self._error_message = PipelineState.ERROR, str(e)
            raise RuntimeError(f"Failed to load step '{s.name}': {e}") from e
        warnings = self._check_alignment(config)
        self._config = config
        with self._lock:
            self._state = PipelineState.READY
        return warnings

    def start(self) -> None:
        with self._lock:
            if self._state != PipelineState.READY:
                raise RuntimeError(f"Cannot start from state {self._state.value}")
            self._state = PipelineState.RUNNING
        self._stop_event.clear()
        self._step, self._total_frames = 0, 0
        self._start_time = time.monotonic()
        self._begin_step(0)
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="pipeline-monitor")
        self._monitor_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
        try:
            self._deploy.stop()
        except Exception:
            pass
        with self._lock:
            self._state = PipelineState.IDLE
        self._loaded.clear()

    def estop(self) -> None:
        self._stop_event.set()
        try:
            self._deploy.estop()
        except Exception:
            pass
        with self._lock:
            self._state = PipelineState.ESTOP

    def trigger_manual(self) -> bool:
        cfg = self._config
        if not cfg or self._step >= len(cfg.steps):
            return False
        t = cfg.steps[self._step].transition
        if t and t.trigger == TransitionTrigger.MANUAL:
            self._manual_trigger.set()
            return True
        return False

    def get_status(self) -> PipelineStatus:
        try:
            cfg, now = self._config, time.monotonic()
            n = len(cfg.steps) if cfg else 0
            sf = getattr(self._deploy, "_frame_count", 0)
            return PipelineStatus(
                state=self._state, current_step_index=self._step,
                current_step_name=cfg.steps[self._step].name if cfg and self._step < n else "",
                total_steps=n, step_frame_count=sf,
                total_frame_count=self._total_frames + sf,
                step_elapsed_seconds=now - self._step_start_time if self._step_start_time else 0,
                total_elapsed_seconds=now - self._start_time if self._start_time else 0,
                transition_progress=self._compute_progress(), error_message=self._error_message)
        except Exception as e:
            return PipelineStatus(state=PipelineState.ERROR, error_message=str(e))

    def _begin_step(self, index: int) -> None:
        policy, obs_builder, ckpt = self._loaded[index]
        step = self._config.steps[index]
        if index == 0:
            self._deploy.start(
                DeploymentConfig(mode=DeploymentMode.INFERENCE, policy_id=step.policy_id,
                                 warmup_frames=step.warmup_frames, loop_hz=self._config.loop_hz),
                self._config.active_arms)
        self._deploy.swap_policy(policy, obs_builder, ckpt, step.warmup_frames)
        self._step_start_time = time.monotonic()
        self._debounce_count = 0
        self._manual_trigger.clear()

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._evaluate_condition(getattr(self._deploy, "_frame_count", 0)):
                    self._transition()
            except Exception as e:
                logger.error("Pipeline monitor error — estop: %s", e)
                self.estop()
                return
            self._stop_event.wait(0.1)

    def _evaluate_condition(self, step_frames: int) -> bool:
        cfg = self._config
        if not cfg or self._step >= len(cfg.steps):
            return False
        trans = cfg.steps[self._step].transition
        if trans is None:
            return False
        trigger, raw = trans.trigger, False
        if trigger == TransitionTrigger.FRAME_COUNT:
            raw = step_frames >= trans.threshold_value
        elif trigger == TransitionTrigger.TIMEOUT:
            raw = (time.monotonic() - self._step_start_time) >= trans.timeout_seconds
        elif trigger in (TransitionTrigger.GRIPPER_CLOSED, TransitionTrigger.GRIPPER_OPENED):
            obs = self._deploy.get_latest_observation()
            if obs:
                th = trans.threshold_value or (0.1 if trigger == TransitionTrigger.GRIPPER_CLOSED else 0.9)
                for k in _GRIPPER_KEYS:
                    v = obs.get(k)
                    if v is not None:
                        raw = v < th if trigger == TransitionTrigger.GRIPPER_CLOSED else v > th
                        break
        elif trigger == TransitionTrigger.TORQUE_SPIKE:
            obs = self._deploy.get_latest_observation()
            if obs:
                th = trans.threshold_value or 5.0
                raw = any(v > th for k, v in obs.items()
                          if k.endswith(".tau") and isinstance(v, (int, float)))
        elif trigger == TransitionTrigger.POSITION_REACHED:
            obs = self._deploy.get_latest_observation()
            if obs and trans.threshold_position:
                tol = trans.threshold_value or 0.05
                raw = all(abs(obs.get(j, float("inf")) - t) < tol
                          for j, t in trans.threshold_position.items())
        elif trigger == TransitionTrigger.MANUAL:
            raw = self._manual_trigger.is_set()
        return self._debounce(raw)

    def _transition(self) -> None:
        self._total_frames += getattr(self._deploy, "_frame_count", 0)
        self._step += 1
        if self._step >= len(self._config.steps):
            with self._lock:
                self._state = PipelineState.COMPLETED
            self._stop_event.set()
            try:
                self._deploy.stop()
            except Exception:
                pass
            logger.info("Pipeline completed (%d steps)", len(self._config.steps))
            return
        logger.info("Pipeline → step %d '%s'", self._step, self._config.steps[self._step].name)
        self._begin_step(self._step)
        with self._lock:
            self._state = PipelineState.RUNNING

    def _debounce(self, value: bool) -> bool:
        if value:
            self._debounce_count += 1
            trans = self._config.steps[self._step].transition
            return self._debounce_count >= (trans.debounce_frames if trans else 8)
        self._debounce_count = 0
        return False

    def _compute_progress(self) -> Optional[TransitionProgress]:
        if self._state != PipelineState.RUNNING or not self._config:
            return None
        if self._step >= len(self._config.steps):
            return None
        trans = self._config.steps[self._step].transition
        if trans is None:
            return None
        current, threshold, label = 0.0, trans.threshold_value, trans.trigger.value
        if trans.trigger == TransitionTrigger.FRAME_COUNT:
            current = float(getattr(self._deploy, "_frame_count", 0))
        elif trans.trigger == TransitionTrigger.TIMEOUT:
            current, threshold = time.monotonic() - self._step_start_time, trans.timeout_seconds
        elif trans.trigger == TransitionTrigger.MANUAL:
            current, threshold = (1.0 if self._manual_trigger.is_set() else 0.0), 1.0
        return TransitionProgress(
            current_value=current, threshold_value=threshold, label=label,
            debounce_current=self._debounce_count, debounce_required=trans.debounce_frames)

    def _load_step_policy(self, policy_id: str) -> Tuple:
        info = self._training.get_policy(policy_id)
        if info is None:
            raise RuntimeError(f"Policy not found: {policy_id}")
        ckpt = getattr(info, "checkpoint_path", None)
        if not ckpt:
            raise RuntimeError(f"Policy {policy_id} has no checkpoint")
        cp = Path(ckpt)
        with open(cp / "config.json") as f:
            ptype = json.load(f).get("type", getattr(info, "policy_type", ""))
        from lerobot.policies.factory import get_policy_class
        policy = get_policy_class(ptype).from_pretrained(str(cp))
        return policy, ObservationBuilder(cp, policy, ptype), str(cp)

    def _check_alignment(self, config: PipelineConfig) -> List[AlignmentWarning]:
        warnings: List[AlignmentWarning] = []
        for i in range(len(config.steps) - 1):
            if i not in self._loaded or (i + 1) not in self._loaded:
                continue
            try:
                sa, sb = self._load_stats(self._loaded[i][2]), self._load_stats(self._loaded[i+1][2])
                if sa is None or sb is None:
                    continue
                am, om = sa.get("action", {}).get("mean", []), sb.get("observation.state", {}).get("mean", [])
                for j, (a, o) in enumerate(zip(am, om)):
                    d = abs(a - o)
                    if d > 0.3:
                        warnings.append(AlignmentWarning(
                            step_from=config.steps[i].name, step_to=config.steps[i+1].name,
                            joint_name=f"joint_{j}", delta_rad=d,
                            message=f"Joint {j} mean delta {d:.2f} rad"))
            except Exception:
                continue
        return warnings

    @staticmethod
    def _load_stats(checkpoint_path: str) -> Optional[dict]:
        cp = Path(checkpoint_path)
        for name in ("stats.json", "dataset_stats.json"):
            if (cp / name).exists():
                with open(cp / name) as f:
                    return json.load(f)
        if (cp / "train_config.json").exists():
            with open(cp / "train_config.json") as f:
                root = json.load(f).get("dataset", {}).get("root")
            if root and (Path(root) / "meta" / "stats.json").exists():
                with open(Path(root) / "meta" / "stats.json") as f:
                    return json.load(f)
        return None
