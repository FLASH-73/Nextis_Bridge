import numpy as np
import logging
from lerobot.model.kinematics import RobotKinematics

logger = logging.getLogger(__name__)

class GravityCompensationService:
    def __init__(self, urdf_path, joint_names):
        """
        Initializes Gravity Compensation Service using Pinocchio (via RobotKinematics).
        """
        self.urdf_path = urdf_path
        self.joint_names = joint_names

        try:
            # We assume RobotKinematics can be used for GRAVITY (Inverse Dynamics)
            # lerobot.model.kinematics.RobotKinematics mostly does Forward/Inverse KINEMATICS.
            # We need DYNAMICS (RNEA).
            # Let's see if we can access the underlying placo/pinocchio model.
            self.kinematics = RobotKinematics(urdf_path, joint_names=joint_names)
            # Access underlying placo robot wrapper
            self.robot_wrapper = self.kinematics.robot
            logger.info(f"Gravity Compensation Initialized with URDF: {urdf_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Gravity Compensation: {e}")
            self.robot_wrapper = None

        # Motor Parameters
        # TODO: User must tune these!
        self.kt = 0.5 # Torque Constant (Nm/A) - Placeholder
        # STS3215 Stall Torque ~30kg.cm = 3Nm. Max Current ~2A? => Kt ~ 1.5?
        # Let's stick with 0.5 for now as a safe starting point.

        # Friction Compensation
        self.ks = 0.05 # Static Friction Torque (Nm) - Placeholder help
        self.kd = 0.01 # Dynamic Friction Coeff - Placeholder

        # PWM Scaling
        # PWM is +/- 1000. 1000 = Max Current/Torque?
        # Need to map Torque (Nm) -> PWM
        self.max_pwm = 1000.0
        self.max_torque = 3.0 # Approx 3Nm?

    def compute_torques(self, joint_positions_deg, joint_velocities_deg):
        """
        Computes gravity + friction compensation torques.
        Returns: Dict {joint_name: pwm_value}
        """
        if not self.robot_wrapper:
            return {}

        try:
            # 1. Update Model State
            # Positions in radians
            q = np.deg2rad(joint_positions_deg)
            # Velocities in rad/s
            dq = np.deg2rad(joint_velocities_deg)
            # Accelerations (0 for gravity only)
            ddq = np.zeros_like(q)

            # Update wrapper
            # Note: Placo/Pinocchio joint order matches self.joint_names
            for i, name in enumerate(self.joint_names):
                 self.robot_wrapper.set_joint(name, q[i])
                 self.robot_wrapper.set_velocity(name, dq[i])
                 # self.robot_wrapper.set_acceleration(name, 0)

            self.robot_wrapper.update_kinematics()

            # 2. Compute Gravity Torques (Generalized Gravity)
            # Placo might expose this as generalized forces?
            # Or we might need to use pinocchio directly if placo doesn't expose it easily.
            # Let's check placo bindings... assuming self.robot_wrapper.generalized_gravity() exists?
            # Warning: Pure speculation on placo API. If it fails, we default to 0.

            # Fallback/Placeholder: Just rely on Friction Comp if Gravity fails
            gravity_torques = np.zeros_like(q)
            if hasattr(self.robot_wrapper, "generalized_gravity"):
                 gravity_torques = self.robot_wrapper.generalized_gravity()

            # 3. Compute Friction Compensation
            # Coloumb + Viscous
            friction_torques = self.ks * np.sign(dq) + self.kd * dq

            # 4. Total Torque
            total_torque = gravity_torques + friction_torques

            # 5. Convert to PWM
            # Torque = Current * Kt
            # PWM is proportional to Current?
            # 1000 PWM ~ Max Torque
            pwm_values = {}
            for i, name in enumerate(self.joint_names):
                tau = total_torque[i]

                # Check bounds
                tau = max(-self.max_torque, min(self.max_torque, tau))

                # Map to PWM (-1000 to 1000)
                pwm = (tau / self.max_torque) * self.max_pwm
                pwm_values[name] = int(pwm)

            return pwm_values

        except Exception as e:
            # logger.error(f"G-Comp Calc Fail: {e}") # Noisy log
            return {}
