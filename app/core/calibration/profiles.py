import logging
from typing import Any, Dict, List

from app.core.config import CALIBRATION_DIR

logger = logging.getLogger(__name__)


class CalibrationProfiles:
    """Calibration profile persistence — save/load/list/delete/restore.

    The active_profiles dict lives on this instance (lazily initialized).
    Accesses shared state via self._svc (the parent CalibrationService).
    """

    def __init__(self, svc):
        self._svc = svc

    def _get_arm_calibration_dir(self, arm_id: str, arm=None):
        """Resolve calibration directory from arm instance, falling back to arm_id.

        Prefers the arm's configured calibration_dir (set via connection.py) so the
        profile system always targets the same directory as _save_calibration().
        Falls back to CALIBRATION_DIR / arm_id for disconnected arms.
        """
        if arm is None:
            arm, _ = self._svc.get_arm_context(arm_id)
        if arm and hasattr(arm, 'calibration_dir'):
            return arm.calibration_dir
        return CALIBRATION_DIR / arm_id

    def ensure_active_profiles_init(self):
        if not hasattr(self, "active_profiles"):
            self.active_profiles = {}
            self._load_persistent_active_profiles()

    def _get_persistence_path(self):
        CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        return CALIBRATION_DIR / "active_profiles.json"

    def _load_persistent_active_profiles(self):
        fpath = self._get_persistence_path()
        if fpath.exists():
            import json
            try:
                with open(fpath, "r") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.active_profiles = data
                        logger.info(f"Loaded persistent active profiles: {self.active_profiles}")
            except Exception as e:
                logger.error(f"Failed to load persistent active profiles: {e}")

    def _save_persistent_active_profiles(self):
        fpath = self._get_persistence_path()
        import json
        try:
            with open(fpath, "w") as f:
                json.dump(self.active_profiles, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save persistent active profiles: {e}")

    def restore_active_profiles(self):
        """Attempts to load the calibration files for the active profiles."""
        self.ensure_active_profiles_init()
        logger.info("Restoring active calibration profiles...")

        for arm_id, filename in self.active_profiles.items():
            logger.info(f"Restoring {arm_id} -> {filename}")
            self.load_calibration_file(arm_id, filename)

    def list_calibration_files(self, arm_id: str) -> List[Dict[str, Any]]:
        arm, _ = self._svc.get_arm_context(arm_id)
        if not arm:
            return []

        self.ensure_active_profiles_init()

        # Use arm's configured calibration directory
        base_dir = self._get_arm_calibration_dir(arm_id, arm)

        if not base_dir.exists():
            base_dir.mkdir(parents=True, exist_ok=True)

        files = []
        for f in base_dir.glob("*.json"):
            import datetime
            stats = f.stat()
            dt = datetime.datetime.fromtimestamp(stats.st_mtime)

            is_active = False
            if arm_id in self.active_profiles:
                if self.active_profiles[arm_id] == f.stem:
                    is_active = True

            files.append({
                "name": f.stem,
                "created": dt.strftime("%Y-%m-%d %H:%M"),
                "path": str(f.absolute()),
                "active": is_active
            })

        # Sort by newest first
        files.sort(key=lambda x: x["created"], reverse=True)
        return files

    def load_calibration_file(self, arm_id: str, filename: str):
        arm, _ = self._svc.get_arm_context(arm_id)
        if not arm:
            return False

        self.ensure_active_profiles_init()

        base_dir = self._get_arm_calibration_dir(arm_id, arm)
        fpath = base_dir / f"{filename}.json"

        if not fpath.exists():
            logger.error(f"Calibration file not found: {fpath}")
            return False

        logger.info(f"Loading calibration for {arm_id} from {fpath}")
        try:
            # Load calibration data into memory from the profile file.
            # No need to call _save_calibration() — the profile file on disk
            # IS the persistent copy (calibration_dir already points here).
            arm._load_calibration(fpath)
            # Apply to motors (Dynamixel EEPROM writes require torque off)
            if hasattr(arm.bus, "write_calibration"):
                if self._svc._is_dynamixel_arm(arm_id):
                    arm.bus.disable_torque()
                arm.bus.write_calibration(arm.calibration)
                # Restore gripper spring mode after EEPROM writes
                if self._svc._is_dynamixel_arm(arm_id) and hasattr(arm, 'configure'):
                    arm.configure()

            # Propagate calibrated joint limits to motor bus for runtime enforcement
            if hasattr(arm, 'apply_calibration_limits'):
                arm.apply_calibration_limits()

            # Update Active Profile state
            self.active_profiles[arm_id] = filename
            self._save_persistent_active_profiles()

            logger.info(f"Successfully loaded calibration for {arm_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to load calibration: {e}")
            return False

    def delete_calibration_file(self, arm_id: str, filename: str):
        base_dir = self._get_arm_calibration_dir(arm_id)
        fpath = base_dir / f"{filename}.json"
        if fpath.exists():
            fpath.unlink()
            return True
        return False

    def save_calibration(self, arm_id: str, name: str = None):
        arm, _ = self._svc.get_arm_context(arm_id)
        if not arm:
            return

        # Write calibration to motor hardware (Feetech STS3215 only)
        # Damiao: host-side JSON only (absolute encoders, no offset registers)
        # Dynamixel leaders: host-side JSON only (gripper has torque enabled,
        # can't write EEPROM registers like Homing_Offset/Position_Limits)
        if not self._svc._is_damiao_arm(arm_id) and not self._svc._is_dynamixel_arm(arm_id):
            if self._svc.robot_lock:
                with self._svc.robot_lock:
                    arm.bus.write_calibration(arm.calibration)
            else:
                arm.bus.write_calibration(arm.calibration)

        # Usage of internal save for default file
        if hasattr(arm, "_save_calibration"):
            arm._save_calibration()
        else:
            logger.warning(f"Save calibration not supported for {arm}")

        # If a name is provided, ALSO save to the profile directory
        if name:
            self.ensure_active_profiles_init()
            import dataclasses
            import json

            # Create profile dir
            base_dir = self._get_arm_calibration_dir(arm_id, arm)
            base_dir.mkdir(parents=True, exist_ok=True)

            # Sanitize name
            safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip()
            if not safe_name:
                safe_name = "unnamed"

            fpath = base_dir / f"{safe_name}.json"

            if hasattr(arm, "_save_calibration"):
                arm._save_calibration(fpath)
                logger.info(f"Saved named calibration to {fpath}")

            # Update Active Profile state
            self.active_profiles[arm_id] = safe_name
            self._save_persistent_active_profiles()
