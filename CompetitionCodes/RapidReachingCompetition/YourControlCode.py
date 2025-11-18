import mujoco
import numpy as np


class YourCtrl:
  def __init__(self, m:mujoco.MjModel, d: mujoco.MjData, target_points):
    self.m = m
    self.d = d
    self.target_points = target_points

    self.init_qpos = d.qpos.copy()

    # Control gains (using similar values to CircularMotion)
    self.kp = 150.0
    self.kd = 10.0

    self.ee_id = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_BODY, "EE_Frame")

  def getIK(self, goal, initq):
      
    qi = initq.copy
    epsilon = 0.01

    dx = goal-self.d.xpos[self.ee_id].copy()
    while np.linalg.norm(dx) > epsilon:
      jacp =  np.zeros((3,  self.m.nv)) 
      jcar = np.zeros((3,  self.m.nv))
      mujoco.mj_jac(self.m, self.d, jacp, jcar, point=None, body=self.ee_id)
      J = jacp.copy()

      qi = qi+0.1*np.linalg.pinv(J) @ dx
      self.d.qpos[:] = qi
      mujoco.mj_forward(self.m, self.d)
      
      dx = goal-self.d.xpos[self.ee_id].copy()
    return qi

  def CtrlUpdate(self):
      

    jtorque_cmd = np.zeros(6)
    for i in range(6):
        jtorque_cmd[i] = 150.0*(self.init_qpos[i] - self.d.qpos[i])  - 5.2 *self.d.qvel[i]

    return jtorque_cmd



