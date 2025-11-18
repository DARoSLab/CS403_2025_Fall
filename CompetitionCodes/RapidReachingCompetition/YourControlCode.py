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
      
    dataT = mujoco.MjData(self.m)
    intr = 0
    dataT.qpos[:] = initq.copy() 
    mujoco.mj_forward(self.m, dataT)    

    dx = goal-dataT.xpos[self.ee_id].copy()

    while np.linalg.norm(dx) > 0.01 and intr<100:
      jacp =  np.zeros((3, self.m.nv)) 
      jcar = np.zeros((3, self.m.nv))
      mujoco.mj_jac(self.m, dataT, jacp, jcar, dataT.xpos[self.ee_id], self.ee_id)
      J = jacp.copy()
      
      # piJ = J.T @ np.linalg.inv(J@J.T+0.01**2*np.eye(3))
      piJ = np.linalg.pinv(J)

      dataT.qpos[:] = dataT.qpos[:]+0.01*piJ@dx

      mujoco.mj_forward(self.m, dataT)
      
      dx = goal-dataT.xpos[self.ee_id].copy()
      intr+=1
    return dataT.qpos[:]
  

  def CtrlUpdate(self):
  
    #temp
    goal = self.target_points[:, 3]
    q_d = self.getIK(goal, self.d.qpos)

    M = np.zeros((6,6))
    mujoco.mj_fullM(self.m, M, self.d.qM)  

    jtorque_cmd = M@(self.kp*(q_d - self.d.qpos) - self.kd*self.d.qvel)+ self.d.qfrc_bias

    # for i in range(6):
    #     jtorque_cmd[i] = self.kp*(q_d[i] - self.d.qpos[i])  - self.kd *self.d.qvel[i]

    return jtorque_cmd



