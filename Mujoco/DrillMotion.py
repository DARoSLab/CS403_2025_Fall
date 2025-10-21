import mujoco
import numpy as np
import mujoco

def quat2SO3(quaternion):
  w, x, y, z = quaternion
  return np.array([[w*w + x*x - y*y - z*z, 2*(x*y - z*w), 2*(w*y + x*z)],
                   [2*(w*z + x*y), w*w - x*x + y*y - z*z, 2*(y*z - w*x)],
                   [2*(x*z - w*y), 2*(w*x + y*z), w*w - x*x - y*y + z*z]])

def SO3_2_so3(ori_mtx):
  theta = np.arccos((np.trace(ori_mtx) -1)/2)
  if abs(theta) < 0.0001:
    return np.zeros(3)
  else:
    W = 1/(2*np.sin(theta)) * (ori_mtx - ori_mtx.T) 
    return theta * np.array([W[2,1], W[0,2], W[1,0]], dtype=float)

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
  theta = 2 * np.arcsin(np.sqrt(x*x + y*y + z*z))
  # if theta > np.pi:
  #   print("theta: " + str(theta))
  #   theta = 2*np.pi - theta

  if abs(theta) < 0.0000001:
    return np.zeros(3)
  else:
    return theta / np.sin(theta/2) * np.array([x, y, z], dtype=float)

class DrillMotionCtrl:
  
  def __init__(self, m: mujoco.MjModel, d: mujoco.MjData):
    self.m = m
    self.d = d
    self.kp = 150
    self.kd = 10.0 
    self.init_qpos = d.qpos.copy()
    self.drill_displacement = 0.02
    self.hole_center = np.array([0.55, 0, 0.85])  # Example desired position 
    self.r_freq = 0.5
    self.pitch = -np.pi/5
    self.prep_time = 1.5
    # self.pitch = 0
    self.drill_ori = np.array([np.cos(self.pitch/2), 0, np.sin(self.pitch/2), 0])
    self.count_not_converging = 0
  
  def update(self):
    target_position = self.hole_center
    th = 2*np.pi*self.r_freq*self.d.time
    local_ori_change = np.array([np.cos(th/2), np.sin(th/2), 0, 0])
    print("local_ori_change: " + str(local_ori_change))
    target_ori = quat_multiply(self.drill_ori, local_ori_change) 
    # target_ori_mtx = quat2SO3(self.drill_ori) @ quat2SO3(local_ori_change)
    target_ori_mtx = quat2SO3(self.drill_ori)
    target_position += target_ori_mtx @ np.array([0.02*self.drill_displacement*np.sin(self.d.time), 0, 0])
    # target_ori = quat_multiply(local_ori_change, self.drill_ori) 
    # get EE position
    point = np.zeros(3)
    ee_id = mujoco.mj_name2id(self.m, 1, "EE_Frame")

    nv = self.m.nv
    # print("nv: " + str(nv))
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))

    initial_jpos = np.copy(self.d.qpos[:6])
    target_jpos = np.copy(initial_jpos)
    for(i) in range(25):
      # mujoco.mj_jac(self.m, self.d, jacp, jacr, point, ee_id)
      mujoco.mj_jacBody(self.m, self.d, jacp, jacr, ee_id)
      EE_pos = self.d.body(ee_id).xpos
      EE_ori = self.d.body(ee_id).xquat
      EE_ori_mtx = quat2SO3(EE_ori)
      # print("EE pos: " + str(EE_pos))
      # print("EE ori: " + str(EE_ori))
      pos_err = (target_position - EE_pos)
      # computer orientation error in quaternion
      quat_err = quat_multiply(target_ori, quat_inv(EE_ori)) 
      ori_err_quat = quat2so3(quat_err) 

      ori_err_mtx = target_ori_mtx @ EE_ori_mtx.T
      ori_err_so3 = SO3_2_so3(ori_err_mtx)

      # ori_err = ori_err_so3
      ori_err = ori_err_quat

      pose_err = np.concatenate((pos_err, ori_err))

      J_pose = np.concatenate((jacp[:,:6], jacr[:,:6]))
      target_jpos = target_jpos + 0.01 * np.linalg.pinv(J_pose) @ pose_err

      self.d.qpos[:6] = target_jpos
      mujoco.mj_kinematics(self.m, self.d) 

      if(np.linalg.norm(pos_err) < 0.01 and np.linalg.norm(ori_err) < 0.01):
        break
      # if (i > 23):
        # print("Not converging")
        # print("EE pos: " + str(EE_pos))
        # print("target pos: " + str(target_position))
        # print("EE ori: " + str(EE_ori))
        # print("target ori: " + str(target_ori))
        # print("pose err: " + str(pose_err))
        # print("ori_err_quat: " + str(ori_err_quat))
        # print("ori_err_so3: " + str(ori_err_so3))

        # with np.printoptions(precision=4, suppress=True, formatter={'float': '{:0.4f}'.format}, linewidth=100):
          # print("J_pos: " + str(J_pose))
          # print("target ori mtx: " + str(target_ori_mtx))
          # print("quat2SO3(target_ori): " + str(quat2SO3(target_ori)))
          # print("Jacp: " + str(jacp))
          # print("Jacr: " + str(jacr))
        # print("J_pos: " + str(J_pose))
        # self.count_not_converging += 1
        # print(self.count_not_converging)
        # if(self.count_not_converging > 520):
        #   exit()

      # print(pose_err)

    # print("\n")
    self.d.qpos[:6] = np.copy(initial_jpos)
    jpos_error = target_jpos - self.d.qpos[:6]
    # Calculate joint velocity
    velocity = self.d.qvel[:6]


    # print(self.d.qpos)
    # print(self.d.qfrc_bias)
    # Calculate control signal
    A = np.zeros((nv,nv))
    mujoco.mj_fullM(self.m, A, self.d.qM)
    ArmMassMtx = A[:6,:6]
    control_signal = ArmMassMtx @ (self.kp * jpos_error - self.kd * velocity) + self.d.qfrc_bias[:6]
    # print(control_signal)

    # Apply control signal to joints
    self.d.ctrl = control_signal

