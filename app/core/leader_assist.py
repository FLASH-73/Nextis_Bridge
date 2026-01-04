import numpy as np
import logging
import json
import os
import math

logger = logging.getLogger(__name__)

class LeaderAssistService:
    def __init__(self, arm_id="default", calibration_path=None):
        """
        Initializes Leader Assistance Service.
        Includes Regressed Gravity Model, Friction Assistance, and Haptics.
        """
        self.arm_id = arm_id
        
        # Organize in visibly named folder
        base_dir = os.path.join(os.getcwd(), "calibration_gravity")
        os.makedirs(base_dir, exist_ok=True)
        
        if calibration_path is None:
             self.calibration_path = os.path.join(base_dir, f"gravity_{arm_id}.json")
        else:
             self.calibration_path = calibration_path
             
        self.gravity_weights = {} # {joint_name: [w1, w2, ...]}
        self.is_calibrated = False
        self.load_calibration()
        
        # Friction Assistance Gain
        # Assist torque = K_assist * tanh(velocity / epsilon)
        # This acts as negative damping to cancel out physical friction.
        # Tunable! Start small.
        self.k_assist = 0.5  # Reduced for stability
        self.v_threshold = 2.0 
        self.vel_deadband = 1.0 # Ignore noise (Deadband)
        
        # Haptic Gain
        # Percent of follower torque to reflect back
        self.k_haptic = 0.0 # Disabled by default
        
        # Gravity Gain (Scalar to fine tune the learned model)
        self.k_gravity = 1.0
        
        # Gain for transparency (damping compensation)
        self.k_damping = 0.5 # Default low damping for better transparency
        
        # Calibration State
        self.calibration_mode = False
        self.calibration_data = [] # List of (q, load) tuples
        
        # Max PWM for safety
        self.max_pwm = 400 # Cap at 40% (Safety First)

    def update_gains(self, k_gravity=None, k_assist=None, k_haptic=None, v_threshold=None, k_damping=None):
        if k_gravity is not None: self.k_gravity = float(k_gravity)
        if k_assist is not None: self.k_assist = float(k_assist)
        if k_haptic is not None: self.k_haptic = float(k_haptic)
        if v_threshold is not None: self.v_threshold = float(v_threshold)
        if k_damping is not None: self.k_damping = float(k_damping)
        logger.info(f"Updated Gains: G={self.k_gravity}, F={self.k_assist}, H={self.k_haptic}, D={self.k_damping}")
        
    def load_calibration(self):
        if os.path.exists(self.calibration_path):
            try:
                with open(self.calibration_path, 'r') as f:
                    self.gravity_weights = json.load(f)
                self.is_calibrated = True
                logger.info(f"Loaded Gravity Calibration from {self.calibration_path}")
            except Exception as e:
                logger.error(f"Failed to load calibration: {e}")
        else:
            logger.warning("No Gravity Calibration found. Run calibration routine!")

    def save_calibration(self):
        try:
            with open(self.calibration_path, 'w') as f:
                json.dump(self.gravity_weights, f)
            self.is_calibrated = True
            logger.info(f"Saved Gravity Calibration to {self.calibration_path}")
        except Exception as e:
            logger.error(f"Failed to save calibration: {e}")

    # --- Feature Engineering ---
    def _compute_features(self, q_deg):
        """
        Computes regression features from joint positions (degrees).
        Features: sin(q), cos(q), sin(q1+q2), etc.
        For a simple 6DOF arm, mainly independent terms dominate, but coupling exists.
        Simplified Model: [1, sin(q), cos(q)] per joint (Independent Joint Approximation)
        """
        q = [math.radians(x) for x in q_deg]
        feats = [1.0] # Bias
        for val in q:
            feats.append(math.sin(val))
            feats.append(math.cos(val))
        return np.array(feats)

    # --- Calibration Routine ---
    def start_calibration(self):
        logger.info("Starting Gravity Calibration Mode...")
        self.calibration_mode = True
        self.calibration_data = []
        
    def record_sample(self, positions_deg, loads_raw):
        """
        Records a sample for calibration.
        positions_deg: list of joint angles
        loads_raw: list of current loads (PWM value needed to hold position)
                   In calibration, user holds arm still, so Load ~ Gravity.
        """
        if not self.calibration_mode: return
        self.calibration_data.append((positions_deg, loads_raw))
        logger.info(f"Recorded Calibration Sample {len(self.calibration_data)}")
        
    def compute_weights(self):
        """
        Fits linear regression model W * Features = Load
        """
        if not self.calibration_data:
            logger.error("No data to calibrate!")
            return
            
        logger.info(f"Computing Gravity Weights from {len(self.calibration_data)} samples...")
        
        # Prepare Data Matrices
        # We model each joint independently for simplicity of implementation first.
        # Load_i = W_i * Features
        
        num_joints = len(self.calibration_data[0][1])
        X = [] # Feature Matrix
        Y = [[] for _ in range(num_joints)] # list of target vectors
        
        for q, load in self.calibration_data:
            feat = self._compute_features(q)
            X.append(feat)
            for i in range(num_joints):
                Y[i].append(load[i])
                
        X = np.array(X)
        
        self.gravity_weights = {} # Reset
        
        # Solve Least Squares for each joint
        # W = (X^T X)^-1 X^T Y
        try:
            for i in range(num_joints):
                y_vec = np.array(Y[i])
                # Ridge Regression (add small lambda I) for stability
                lambda_reg = 1e-3
                w = np.linalg.inv(X.T @ X + lambda_reg * np.eye(X.shape[1])) @ X.T @ y_vec
                
                # Store weights
                joint_name = f"joint_{i}" # Generic name, map closer to real names if passed
                self.gravity_weights[joint_name] = w.tolist()
                
            self.save_calibration()
            self.calibration_mode = False
            logger.info("Calibration Complete!")
            
        except Exception as e:
            logger.error(f"Calibration Failed: {e}")

    # --- Runtime Control ---
    def predict_gravity(self, positions_deg):
        """
        Predicts gravity torque (PWM) for given positions.
        Returns list of floats.
        """
        features = self._compute_features(positions_deg)
        num_joints = len(positions_deg)
        gravity_pwm = []
        
        for i in range(num_joints):
            w_key = f"joint_{i}"
            val = 0.0
            if self.is_calibrated and w_key in self.gravity_weights:
                w = np.array(self.gravity_weights[w_key])
                val = np.dot(w, features)
            gravity_pwm.append(val)
            
        return gravity_pwm

    def compute_assist_torque(self, joint_names, positions_deg, velocities_deg, follower_torques=None):
        """
        Computes Total Assist Torque (PWM).
        """
        pwm_values = {}
        
        # Precompute Features
        features = self._compute_features(positions_deg)
        
        for i, name in enumerate(joint_names):
            total_pwm = 0.0
            
            # 1. Gravity Compensation (G)
            w_key = f"joint_{i}"
            if self.is_calibrated and w_key in self.gravity_weights:
                w = np.array(self.gravity_weights[w_key])
                g_pwm = np.dot(w, features)
                total_pwm += g_pwm * self.k_gravity
                
            # 2. Friction Assistance (F)
            # Negative Damping to overcome friction
            # tanh(v / threshold) gives smooth -1 to 1 switch
            vel = velocities_deg[i]
            if abs(vel) > self.vel_deadband and self.k_assist > 0:
                f_assist = self.k_assist * math.tanh(vel / self.v_threshold) * 100.0 
                total_pwm += f_assist
            
            # 3. Haptic Feedback (H)
            # Add follower load (scaled)
            if follower_torques:
                # If follower_torques is dict, lookup by name. If list, assume sync index.
                # Usually dict {link1: val} passed from teleop
                # Note: follower_torques here should be the "External Force" ideally, 
                # but if raw is passed, we just scale it.
                f_load = 0.0
                if isinstance(follower_torques, dict):
                    if name in follower_torques:
                        f_load = follower_torques[name]
                elif i < len(follower_torques):
                     f_load = follower_torques[i]
                     
                h_pwm = f_load * self.k_haptic
                
                # INVERT HAPTIC: If load is +, we want - torque to resist
                total_pwm -= h_pwm
            
            # 4. Damping (Stability)
            # Subtract viscous friction term
            total_pwm -= vel * self.k_damping
            
            # Clamp
            # Safety Cap: 400 (out of 1000) for robustness
            SAFETY_LIMIT = 400
            total_pwm = max(-SAFETY_LIMIT, min(SAFETY_LIMIT, total_pwm))
            
            pwm_values[name] = int(total_pwm)
            
        return pwm_values
