import numpy as np
import mujoco
from scipy.spatial import transform

# import quaternion

def quat_multiply(quaternion0, quaternion1):
  w0, x0, y0, z0 = quaternion0
  w1, x1, y1, z1 = quaternion1
  v0 = np.array([x0, y0, z0])
  v1 = np.array([x1, y1, z1])
  r = w0 * w1 - v0.dot(v1)
  v = w0 * v1 + w1 * v0 + np.cross(v0, v1)
  return np.array([r, v[0], v[1], v[2]], dtype=float)

def quat_inv(quaternion):
  w0, x0, y0, z0 = quaternion
  return np.array([w0, -x0, -y0, -z0], dtype=float)

def quat2so3(quaternion):
  w, x, y, z = quaternion
  theta = 2 * np.arccos(w)
  if abs(theta) < 0.0001:
    return np.zeros(3)
  else:
    return theta / np.sin(theta/2) * np.array([x, y, z], dtype=float)

def rotationMatrixToQuaternion3(m):
  #q0 = qw
  t = np.matrix.trace(m)
  q = np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float64)

  if(t > 0):
    t = np.sqrt(t + 1)
    q[0] = 0.5 * t
    t = 0.5/t
    q[1] = (m[2,1] - m[1,2]) * t
    q[2] = (m[0,2] - m[2,0]) * t
    q[3] = (m[1,0] - m[0,1]) * t

  else:
    i = 0
    if (m[1,1] > m[0,0]):
      i = 1
    if (m[2,2] > m[i,i]):
      i = 2

    j = (i+1)%3
    k = (j+1)%3

    t = np.sqrt(m[i,i] - m[j,j] - m[k,k] + 1)
    q[i+1] = 0.5 * t
    t = 0.5 / t
    q[0] = (m[k,j] - m[j,k]) * t
    q[j+1] = (m[j,i] + m[i,j]) * t
    q[k+1] = (m[k,i] + m[i,k]) * t

  return q


class CircularMotionCtrl:
  
  def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
    self.m = m
    self.d = d
    self.kp = 150
    self.kd = 10.0 
    self.init_qpos = d.qpos.copy()
    self.c_center = np.array([0.4, -0.6, 0.9])  # Example desired position
    self.c_radius = 0.25

    self.c_axis = np.array([1, -1, 0])
    cx_axis = self.c_axis / np.linalg.norm(self.c_axis)
    cy_axis = np.cross(cx_axis, np.array([0, 0, -1]))
    cy_axis = cy_axis / np.linalg.norm(cy_axis)
    cz_axis = np.cross(cx_axis, cy_axis)
    cz_axis = cz_axis / np.linalg.norm(cz_axis)
    self.c_rot = np.array([cx_axis, cy_axis, cz_axis]).T

    print(self.c_rot)
    self.r_freq = 0.5

  def update(self):
    # target_position = self.c_radius * np.array([np.cos(2*np.pi*self.r_freq*self.d.time), 0, np.sin(2*np.pi*self.r_freq*self.d.time)]) + self.c_center
    target_position = self.c_radius * np.array([0, np.cos(2*np.pi*self.r_freq*self.d.time), np.sin(2*np.pi*self.r_freq*self.d.time)])
    target_position = self.c_rot @ target_position
    target_position += self.c_center
    
    target_ori = np.zeros(4)
    ori_quat = transform.Rotation.from_matrix(self.c_rot).as_quat()
    target_ori[0] = ori_quat[3]
    target_ori[1:4] = ori_quat[0:3]

    # target_ori = rotationMatrixToQuaternion3(self.c_rot)

    # get EE position
    point = np.zeros(3)
    ee_id = mujoco.mj_name2id(self.m, 1, "EE_Frame")

    jacp = np.zeros((3, 6))
    jacr = np.zeros((3, 6))

    initial_jpos = np.copy(self.d.qpos[:6])
    target_jpos = np.copy(initial_jpos)
    for(i) in range(3):
      mujoco.mj_jac(self.m, self.d, jacp, jacr, point, ee_id)
      # mujoco.mj_jacBody(self.m, self.d, jacp, jacr, ee_id)
      EE_pos = self.d.body(ee_id).xpos
      EE_ori = self.d.body(ee_id).xquat

      pos_err = (target_position - EE_pos)
      quat_err = quat_multiply(target_ori, quat_inv(EE_ori)) 
      ori_err = quat2so3(quat_err) 
      pose_err = np.concatenate((pos_err, ori_err))

      J_pose = np.concatenate((jacp, jacr))
      
      # print(J_pose)

      target_jpos += 0.1*np.linalg.pinv(J_pose) @ pose_err

      self.d.qpos[:6] = target_jpos
      mujoco.mj_kinematics(self.m, self.d) 
      # print(pos_err)

    self.d.qpos[:6] = np.copy(initial_jpos)
    jpos_error = target_jpos - self.d.qpos[:6]
    # Calculate joint velocity
    velocity = self.d.qvel[:6]


    # print(self.d.qpos)
    # print(self.d.qfrc_bias)
    # Calculate control signal
    A = np.zeros((6,6))
    mujoco.mj_fullM(self.m, A, self.d.qM)
    control_signal = A @ (self.kp * jpos_error - self.kd * velocity) + self.d.qfrc_bias[:6]
    # print(control_signal)
    # breakpoint()
    # Apply control signal to joints
    self.d.ctrl = control_signal

