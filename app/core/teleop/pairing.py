import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class PairingContext:
    """Per-pairing state for independent teleop loops.

    Each leader→follower pair gets its own context so that multiple pairs
    can run simultaneously without cross-contaminating mapping, scaling,
    or value-mode state.
    """
    pairing_id: str            # e.g. "aira_zero_leader→aira_zero"
    active_leader: object      # Leader arm instance
    active_robot: object       # Follower arm instance
    joint_mapping: dict        # {leader_key: follower_key}
    follower_value_mode: str   # "float" (Damiao rad), "rad_to_percent" (Dyn→Feetech), "int" (legacy)
    has_damiao_follower: bool
    leader_cal_ranges: dict    # {follower_key: (range_min, range_max)} from leader calibration
    # Mutable per-loop state (reset each start):
    follower_start_pos: dict = field(default_factory=dict)
    leader_start_rad: dict = field(default_factory=dict)
    rad_to_percent_scale: dict = field(default_factory=dict)
    blend_start_time: float | None = None
    filtered_gripper_torque: float = 0.0


# Dynamixel leader uses joint_N names, Damiao follower uses base/linkN names
DYNAMIXEL_TO_DAMIAO_JOINT_MAP = {
    "joint_1": "base",
    "joint_2": "link1",
    "joint_3": "link2",
    "joint_4": "link3",
    "joint_5": "link4",
    "joint_6": "link5",
    "gripper": "gripper",
}


def precompute_mappings(svc):
    """Pre-computes active joint mappings to avoid string ops in the loop."""
    svc.joint_mapping = {}
    svc.assist_groups = {}

    # Detect if any follower arm is Damiao (uses float radians, not int ticks)
    svc._has_damiao_follower = False
    # Value conversion mode: "float" (Damiao rad), "rad_to_percent" (Dyn→Feetech), "int" (legacy)
    svc._follower_value_mode = "int"
    if svc.robot:
        try:
            from lerobot.robots.damiao_follower.damiao_follower import DamiaoFollowerRobot as DamiaoFollower
            if isinstance(svc.robot, DamiaoFollower):
                svc._has_damiao_follower = True
            elif hasattr(svc.robot, 'left_arm') and isinstance(svc.robot.left_arm, DamiaoFollower):
                svc._has_damiao_follower = True
            elif hasattr(svc.robot, 'right_arm') and isinstance(svc.robot.right_arm, DamiaoFollower):
                svc._has_damiao_follower = True
        except ImportError:
            pass

    # Check arm_registry for Damiao follower arms
    if not svc._has_damiao_follower and svc.arm_registry and svc.active_arms:
        for arm_id in svc.active_arms:
            arm = svc.arm_registry.arms.get(arm_id)
            if arm and arm.motor_type == 'damiao' and arm.role.value == 'follower':
                svc._has_damiao_follower = True
                break

    if not svc.leader and not svc.arm_registry:
         return

    # Try pairing-based mapping first (new arm registry system)
    if svc.arm_registry:
        precompute_mappings_from_pairings(svc)
    else:
        # Fallback to legacy side-based mapping
        precompute_mappings_legacy(svc)


def precompute_mappings_from_pairings(svc):
    """Use explicit pairings from arm registry for joint mapping."""
    pairings = svc.arm_registry.get_active_pairings(svc.active_arms)

    for pairing in pairings:
        leader_id = pairing['leader_id']
        follower_id = pairing['follower_id']

        # Only map if both are in active selection (or no selection = all active)
        if svc.active_arms is not None:
            if leader_id not in svc.active_arms or follower_id not in svc.active_arms:
                continue

        # Check if this is a Dynamixel→Damiao pairing (different joint naming)
        # Use .arms dict directly to get ArmDefinition objects (not .get_arm() which returns dicts)
        leader_arm = svc.arm_registry.arms.get(leader_id) if svc.arm_registry else None
        follower_arm = svc.arm_registry.arms.get(follower_id) if svc.arm_registry else None

        is_dynamixel_leader = leader_arm and leader_arm.motor_type in ('dynamixel_xl330', 'dynamixel_xl430')
        is_damiao_follower = follower_arm and follower_arm.motor_type == 'damiao'
        is_feetech_follower = follower_arm and follower_arm.motor_type == 'sts3215'

        if is_dynamixel_leader and is_damiao_follower:
            # Dynamixel→Damiao: direct mapping, float radians passthrough
            for dyn_name, dam_name in DYNAMIXEL_TO_DAMIAO_JOINT_MAP.items():
                svc.joint_mapping[f"{dyn_name}.pos"] = f"{dam_name}.pos"
            svc._has_damiao_follower = True
            svc._follower_value_mode = "float"
        elif is_dynamixel_leader and is_feetech_follower:
            # Dynamixel→Feetech: direct mapping, rad→percent conversion
            for dyn_name, dam_name in DYNAMIXEL_TO_DAMIAO_JOINT_MAP.items():
                svc.joint_mapping[f"{dyn_name}.pos"] = f"{dam_name}.pos"
            svc._follower_value_mode = "rad_to_percent"
            # Precompute leader calibration ranges for absolute rad→percent mapping
            if svc._active_leader and hasattr(svc._active_leader, 'calibration') and svc._active_leader.calibration:
                for dyn_name, dam_name in DYNAMIXEL_TO_DAMIAO_JOINT_MAP.items():
                    if dyn_name == "gripper":
                        continue
                    l_cal = svc._active_leader.calibration.get(dyn_name)
                    if l_cal:
                        f_key = f"{dam_name}.pos"
                        svc._leader_cal_ranges[f_key] = (l_cal.range_min, l_cal.range_max)
                if svc._leader_cal_ranges:
                    print(f"[Teleop] Absolute mapping: {len(svc._leader_cal_ranges)} joints from leader calibration", flush=True)
                else:
                    print("[Teleop] WARNING: Leader calibration missing — falling back to relative delta tracking", flush=True)
        else:
            # Legacy prefix-based mapping for Feetech/same-type arms
            leader_prefix = get_arm_prefix(svc, leader_id)
            follower_prefix = get_arm_prefix(svc, follower_id)
            for name in svc.joint_names_template:
                leader_key = f"{leader_prefix}{name}.pos"
                follower_key = f"{follower_prefix}{name}.pos"
                svc.joint_mapping[leader_key] = follower_key

    logger.info(f"Pairing-based Mapping: {len(svc.joint_mapping)} joints mapped from {len(pairings)} pairings.")
    if svc.joint_mapping:
        for lk, fk in svc.joint_mapping.items():
            logger.info(f"  {lk} → {fk}")


def build_pairing_context(svc, pairing: dict, leader_inst, follower_inst) -> PairingContext:
    """Build a fully independent PairingContext for one leader→follower pair.

    This ensures each pair's mapping, value mode, and scaling are isolated
    from other pairs — preventing the cross-contamination that caused the
    Damiao crash when running simultaneously with Feetech pairs.
    """
    leader_id = pairing['leader_id']
    follower_id = pairing['follower_id']
    pairing_id = f"{leader_id}→{follower_id}"

    joint_mapping = {}
    follower_value_mode = "int"
    has_damiao_follower = False
    leader_cal_ranges = {}

    # Determine arm types
    leader_arm = svc.arm_registry.arms.get(leader_id) if svc.arm_registry else None
    follower_arm = svc.arm_registry.arms.get(follower_id) if svc.arm_registry else None

    is_dynamixel_leader = leader_arm and leader_arm.motor_type in ('dynamixel_xl330', 'dynamixel_xl430')
    is_damiao_follower_arm = follower_arm and follower_arm.motor_type == 'damiao'
    is_feetech_follower = follower_arm and follower_arm.motor_type == 'sts3215'

    if is_dynamixel_leader and is_damiao_follower_arm:
        # Dynamixel→Damiao: direct mapping, float radians passthrough
        for dyn_name, dam_name in DYNAMIXEL_TO_DAMIAO_JOINT_MAP.items():
            joint_mapping[f"{dyn_name}.pos"] = f"{dam_name}.pos"
        has_damiao_follower = True
        follower_value_mode = "float"
    elif is_dynamixel_leader and is_feetech_follower:
        # Dynamixel→Feetech: direct mapping, rad→percent conversion
        for dyn_name, dam_name in DYNAMIXEL_TO_DAMIAO_JOINT_MAP.items():
            joint_mapping[f"{dyn_name}.pos"] = f"{dam_name}.pos"
        follower_value_mode = "rad_to_percent"
        # Precompute leader calibration ranges for absolute rad→percent mapping
        if leader_inst and hasattr(leader_inst, 'calibration') and leader_inst.calibration:
            for dyn_name, dam_name in DYNAMIXEL_TO_DAMIAO_JOINT_MAP.items():
                if dyn_name == "gripper":
                    continue
                l_cal = leader_inst.calibration.get(dyn_name)
                if l_cal:
                    f_key = f"{dam_name}.pos"
                    leader_cal_ranges[f_key] = (l_cal.range_min, l_cal.range_max)
            if leader_cal_ranges:
                print(f"[Teleop] [{pairing_id}] Absolute mapping: {len(leader_cal_ranges)} joints from leader calibration", flush=True)
            else:
                print(f"[Teleop] [{pairing_id}] WARNING: Leader calibration missing — falling back to relative delta tracking", flush=True)
    else:
        # Legacy prefix-based mapping for Feetech/same-type arms
        leader_prefix = get_arm_prefix(svc, leader_id)
        follower_prefix = get_arm_prefix(svc, follower_id)
        for name in svc.joint_names_template:
            leader_key = f"{leader_prefix}{name}.pos"
            follower_key = f"{follower_prefix}{name}.pos"
            joint_mapping[leader_key] = follower_key

    logger.info(f"[{pairing_id}] Built context: mode={follower_value_mode}, damiao={has_damiao_follower}, {len(joint_mapping)} joints")
    for lk, fk in joint_mapping.items():
        logger.info(f"  [{pairing_id}] {lk} → {fk}")

    return PairingContext(
        pairing_id=pairing_id,
        active_leader=leader_inst,
        active_robot=follower_inst,
        joint_mapping=joint_mapping,
        follower_value_mode=follower_value_mode,
        has_damiao_follower=has_damiao_follower,
        leader_cal_ranges=leader_cal_ranges,
    )


def get_arm_prefix(svc, arm_id: str) -> str:
    """Get the joint name prefix for an arm ID."""
    # For legacy IDs like "left_follower", "right_leader" -> extract side
    if arm_id.startswith("left_"):
        return "left_"
    elif arm_id.startswith("right_"):
        return "right_"
    elif arm_id == "damiao_follower" or arm_id == "damiao_leader":
        return ""  # Damiao uses unprefixed joint names
    else:
        # For custom arm IDs, check the arm registry
        if svc.arm_registry:
            arm = svc.arm_registry.get_arm(arm_id)
            if arm:
                # Use the arm ID as prefix for custom arms
                return f"{arm_id}_"
        return ""  # Default to no prefix


def precompute_mappings_legacy(svc):
    """Legacy side-based mapping (left_leader -> left_follower, etc.)"""
    # Helper to check if active
    def is_active(side, group):
        if svc.active_arms is None:
            return True
        id_str = f"{side}_{group}"
        return id_str in svc.active_arms

    # 1. Teleop Mapping (Leader -> Follower)
    # We iterate potential keys and check if they are active
    for side in ["left", "right", "default"]:
        if side == "default":
             prefix = ""
        else:
             prefix = f"{side}_"

        # Check if this side is active for BOTH leader and follower
        leader_active = is_active(side, "leader")
        follower_active = is_active(side, "follower")

        if leader_active and follower_active:
            for name in svc.joint_names_template:
                leader_key = f"{prefix}{name}.pos"
                follower_key = f"{prefix}{name}.pos" # Robot action key (must end in .pos for Umbra)
                svc.joint_mapping[leader_key] = follower_key

    logger.info(f"Legacy Teleop Mapping: {len(svc.joint_mapping)} joints mapped.")

    # 2. Assist Groups (Leader Keys for Assist Calculation)
    if svc.leader_assists:
        for arm_key in svc.leader_assists.keys():
            prefix = f"{arm_key}_" if arm_key != "default" else ""

            # Pre-generate the list of full joint names for this arm
            # This avoids f-string creation in the loop
            arm_joint_names = [f"{prefix}{name}" for name in svc.joint_names_template]
            svc.assist_groups[arm_key] = arm_joint_names

    logger.info(f"Assist Groups Optimized: {list(svc.assist_groups.keys())}")
