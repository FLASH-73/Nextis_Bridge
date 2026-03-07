import dataclasses, json, re
from pathlib import Path
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.core.deployment import (
    PipelineConfig, PipelineStep, TransitionCondition, TransitionTrigger)
from app.core.deployment.pipeline_types import BridgeConfig
from app.dependencies import get_state

CONFIGS_DIR = Path("pipelines")
router = APIRouter(prefix="/pipeline", tags=["pipeline"])

def _rt():
    return get_state().pipeline_runtime

def _no_rt():
    return JSONResponse(status_code=503, content={"error": "Pipeline runtime not initialized"})

def parse_pipeline_config(body: dict) -> PipelineConfig:
    name = body.get("name")
    if not name or not isinstance(name, str):
        raise ValueError("'name' is required (non-empty string)")
    raw_steps = body.get("steps")
    if not raw_steps or not isinstance(raw_steps, list):
        raise ValueError("'steps' is required (non-empty list)")
    steps = []
    for i, s in enumerate(raw_steps):
        if not isinstance(s, dict):
            raise ValueError(f"Step {i} must be a dict")
        if not s.get("policy_id"):
            raise ValueError(f"Step {i} missing 'policy_id'")
        transition = None
        t = s.get("transition")
        if t and isinstance(t, dict):
            try:
                trigger = TransitionTrigger(t.get("trigger", "manual"))
            except ValueError:
                raise ValueError(f"Step {i}: invalid trigger '{t.get('trigger')}'")
            transition = TransitionCondition(
                trigger=trigger, threshold_value=float(t.get("threshold_value", 0.0)),
                threshold_position=t.get("threshold_position", {}),
                timeout_seconds=float(t.get("timeout_seconds", 0.0)),
                debounce_frames=int(t.get("debounce_frames", 8)))
        bridge_raw = s.get("bridge")
        if bridge_raw and isinstance(bridge_raw, dict):
            bridge = BridgeConfig(
                enabled=bridge_raw.get("enabled", True),
                speed_scale=float(bridge_raw.get("speed_scale", 0.3)),
                settle_frames=int(bridge_raw.get("settle_frames", 15)),
                source=bridge_raw.get("source", "auto"))
        else:
            bridge = BridgeConfig()
        steps.append(PipelineStep(
            policy_id=s["policy_id"], name=s.get("name", f"step_{i}"),
            transition=transition, warmup_frames=int(s.get("warmup_frames", 12)),
            speed_scale=float(s.get("speed_scale", 1.0)),
            temporal_ensemble_coeff=(
                float(s["temporal_ensemble_coeff"])
                if s.get("temporal_ensemble_coeff") is not None else None),
            bridge=bridge))
    return PipelineConfig(
        name=name, steps=steps, active_arms=body.get("active_arms", []),
        loop_hz=int(body.get("loop_hz", 30)), safety_overrides=body.get("safety_overrides"))

@router.post("/load")
async def load_pipeline(request: Request):
    rt = _rt()
    if not rt: return _no_rt()
    try:
        config = parse_pipeline_config(await request.json())
        warnings = rt.load(config)
        return {"status": "loaded", "total_steps": len(config.steps),
                "alignment_warnings": [dataclasses.asdict(w) for w in warnings],
                "start_poses": rt.get_start_poses()}
    except (RuntimeError, ValueError) as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/start")
def start_pipeline():
    rt = _rt()
    if not rt: return _no_rt()
    try:
        rt.start()
        return {"status": "started"}
    except RuntimeError as e:
        return JSONResponse(status_code=400, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/stop")
def stop_pipeline():
    rt = _rt()
    if not rt: return _no_rt()
    try:
        rt.stop()
        return {"status": "stopped"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/estop")
def estop_pipeline():
    rt = _rt()
    if not rt: return _no_rt()
    try:
        rt.estop()
        return {"status": "estop"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.post("/trigger")
def trigger_pipeline():
    rt = _rt()
    if not rt: return _no_rt()
    try:
        if rt.trigger_manual():
            return {"status": "triggered"}
        return JSONResponse(status_code=400, content={"error": "No manual trigger on current step"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/status")
def pipeline_status():
    rt = _rt()
    if not rt: return {"state": "idle", "total_steps": 0}
    return dataclasses.asdict(rt.get_status())

@router.post("/configs/save")
async def save_config(request: Request):
    try:
        body = await request.json()
        name = body.get("name")
        if not name or not isinstance(name, str):
            return JSONResponse(status_code=400, content={"error": "'name' required"})
        safe = re.sub(r'[^\w-]', '', name).strip('-_')
        if not safe:
            return JSONResponse(status_code=400, content={"error": "Invalid config name"})
        CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
        path = CONFIGS_DIR / (safe + ".json")
        path.write_text(json.dumps(body, indent=2))
        return {"status": "saved", "path": str(path)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@router.get("/configs")
def list_configs():
    if not CONFIGS_DIR.exists(): return []
    return [{"name": p.stem, "path": str(p)} for p in sorted(CONFIGS_DIR.glob("*.json"))]

@router.get("/configs/{name}")
def get_config(name: str):
    path = CONFIGS_DIR / f"{name}.json"
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": f"Config '{name}' not found"})
    return JSONResponse(content=json.loads(path.read_text()))

@router.delete("/configs/{name}")
def delete_config(name: str):
    path = CONFIGS_DIR / f"{name}.json"
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": f"Config '{name}' not found"})
    path.unlink()
    return {"status": "deleted"}
