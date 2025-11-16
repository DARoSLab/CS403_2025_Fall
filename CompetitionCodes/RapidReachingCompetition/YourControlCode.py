import mujoco
import numpy as np
import time


class YourCtrl:
  def __init__(self, m: mujoco.MjModel, d: mujoco.MjData, target_points):
    self.m = m
    self.d = d
    self.target_points = target_points  # Shape: (3, num_points)
    self.num_points = target_points.shape[1]
    
    # Track which points we've visited
    self.visited = [False] * self.num_points
    self.visit_threshold = 0.035  # 35mm threshold for considering point "reached"
    self.completed = False  # Flag to track if all points are done
    
    # Get end effector body ID (from lecture examples)
    self.ee_body_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "EE_Frame")
    
    # Task-space control gains (from Lecture 10-11 slide pattern)
    self.kp_task = 600.0  # Proportional gain for position error
    self.kd_task = 50.0   # Derivative gain for velocity damping
    
    # Null-space control gains (from lecture: for redundancy resolution)
    # Keep these small so they don't interfere with task
    self.kp_null = 5.0
    self.kd_null = 2.0
    
    # Desired null-space configuration (comfortable middle position)
    self.q_null = np.array([0.0, -0.5, 0.0, -1.2, 0.0, 0.0])
    
    # Timing
    self.start_time = None  # Will be set when first point is being pursued
    self.completion_time = None

  def _get_ee_position(self):
    """
    Get end effector position in world frame
    From lectures: self.d.xpos[body_id] gives body position
    """
    return self.d.xpos[self.ee_body_id].copy()
  
  def _get_ee_velocity(self):
    """
    Get end effector velocity using Jacobian
    From Lecture 9-11: Linear velocity = Jacobian × joint velocities
    Formula: ẋ = J·q̇
    """
    jacp = np.zeros((3, self.m.nv))
    jacr = np.zeros((3, self.m.nv))
    # From lecture: mj_jacBody computes world frame Jacobian
    mujoco.mj_jacBody(self.m, self.d, jacp, jacr, self.ee_body_id)
    # Velocity = Jacobian @ joint_velocities (from lecture formula)
    return jacp @ self.d.qvel[:6]
  
  def _get_jacobian(self):
    """
    Get position Jacobian for end effector
    From Lecture 10-11: "mujoco.mj_jacBody" returns world frame Jacobian
    """
    jacp = np.zeros((3, self.m.nv))
    jacr = np.zeros((3, self.m.nv))
    mujoco.mj_jacBody(self.m, self.d, jacp, jacr, self.ee_body_id)
    return jacp[:, :6]  # Only first 6 joints
  
  def _select_nearest_target(self):
    """
    Select nearest unvisited target point
    Marks points as visited when within threshold distance
    Returns: (target_idx, distance) tuple
    """
    ee_pos = self._get_ee_position()
    
    # Start timer on first call
    if self.start_time is None:
      self.start_time = time.time()
    
    # First pass: Mark any unvisited points we're close to as visited
    for i in range(self.num_points):
      if not self.visited[i]:
        dist = np.linalg.norm(ee_pos - self.target_points[:, i])
        if dist < self.visit_threshold:
          self.visited[i] = True
          
          # Change color to green to show it's reached
          site_id = self.m.site(f'point{i}').id
          self.m.site_rgba[site_id] = np.array([0.0, 1.0, 0.0, 1.0])  # Green
          
          elapsed = time.time() - self.start_time
          print(f"Point {i} reached! Distance: {dist*1000:.1f}mm | Elapsed: {elapsed:.2f}s")
          print(f"Points remaining: {self.num_points - sum(self.visited)}")
    
    # Check if all points visited
    if all(self.visited):
      if not self.completed:
        self.completion_time = time.time() - self.start_time
        print("\n" + "="*50)
        print("All points reached!")
        print(f"Total points visited: {sum(self.visited)}/{self.num_points}")
        print(f"⏱️  Total completion time: {self.completion_time:.2f} seconds")
        print("="*50 + "\n")
        self.completed = True
      return 0, 0.0  # Special return value indicating completion
    
    # Second pass: Find nearest unvisited point
    min_dist = float('inf')
    nearest_idx = 0
    
    for i in range(self.num_points):
      if not self.visited[i]:
        dist = np.linalg.norm(ee_pos - self.target_points[:, i])
        if dist < min_dist:
          min_dist = dist
          nearest_idx = i
    
    return nearest_idx, min_dist

  def CtrlUpdate(self):
    """
    Main control loop
    Based on lecture patterns for end-effector control
    """
    # Select target (returns index and distance)
    target_idx, dist_to_target = self._select_nearest_target()
    
    # If all points visited, stop at the last point
    if dist_to_target == 0.0:
      # Return zero torques to stop all motion
      return np.zeros(6)
    
    target_pos = self.target_points[:, target_idx]
    
    # Get current end effector state
    ee_pos = self._get_ee_position()
    ee_vel = self._get_ee_velocity()
    
    # (from Lecture 10-11)
    # Position error in Cartesian space
    pos_error = target_pos - ee_pos
    
    # Desired task-space velocity using PD law
    # Formula: ẋ_desired = Kp * error - Kd * velocity
    desired_task_vel = self.kp_task * pos_error - self.kd_task * ee_vel
    
    # Limiting  maximum velocity for stability
    max_task_vel = 1.5  # m/s
    task_vel_norm = np.linalg.norm(desired_task_vel)
    if task_vel_norm > max_task_vel:
      desired_task_vel = desired_task_vel * (max_task_vel / task_vel_norm)
    
    # Jacobian (from Lecture 9-11)
    J = self._get_jacobian()
    
    # Compute damped pseudo-inverse for inverse kinematics (from Lecture 2)
    # For non-square matrix: J^+ = J^T (J J^T + λI)^{-1}
    # Damping prevents singularity issues
    damping = 0.01
    J_damped = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(3))
    
    # Compute desired joint velocities (from lecture: inverse kinematics)
    # q̇_task = J^+ × ẋ_desired (primary task)
    q_dot_task = J_damped @ desired_task_vel
    
    # Null-space control (from Lecture 9-11: redundancy resolution)
    # For 6-DOF arm controlling 3-DOF position, we have 3 extra DOF
    # Use these for secondary objectives without affecting primary task
    
    # Null-space projector: N = I - J^+ J
    null_space_proj = np.eye(6) - J_damped @ J
    
    # Secondary objective: move joints toward comfortable configuration
    q_error_null = self.q_null - self.d.qpos[:6]
    q_dot_null = self.kp_null * q_error_null - self.kd_null * self.d.qvel[:6]
    
    # Combined desired joint velocity (primary + secondary in null-space)
    # q̇_cmd = q̇_task + N q̇_null
    q_dot_cmd = q_dot_task + null_space_proj @ q_dot_null
    
    # Convert to torques using joint-space PD control (from Lecture 10-11)
    kp_joint = 50.0
    kd_joint = 15.0
    
    # Velocity tracking: τ = Kd * (q̇_desired - q̇_current)
    torque = kd_joint * (q_dot_cmd - self.d.qvel[:6])
    
    # Add gravity compensation (from Lecture 15: Newton-Euler dynamics)
    # mj_rne computes inverse dynamics: gravity + Coriolis forces
    qfrc_bias = np.zeros(self.m.nv)
    mujoco.mj_rne(self.m, self.d, 0, qfrc_bias)
    torque += qfrc_bias[:6]
    
    # Torque limits for safety
    torque = np.clip(torque, -50, 50)
    
    return torque
