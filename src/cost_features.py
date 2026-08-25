"""Cost feature library.

Crocoddyl residual models for the terms the IRL recovers -- joint torque,
velocity, acceleration, torque change, mechanical power and kinetic energy --
plus the task residuals. Each residual is one column of the feature vector phi.
"""
import pinocchio as pin
import numpy as np
import crocoddyl

'''
Residual Model for Joint Torque: 
Cost = (1/2) * tau^T * tau
R = tau
Rx = dtau_dx --> From actuation model in crocoddyl
Ru = dtau_du --> From actuation model in crocoddyl
'''
class ResidualModelTorque(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, actuation):
        self.unone = np.zeros(state.nv)
        self.model = state.pinocchio
        self.data = self.model.createData()
        self.actuation = actuation
        self.actuation_data = self.actuation.createData()
        super().__init__(state = state, 
                         nr = actuation.nu, 
                         nu = actuation.nu, 
                         q_dependent= True, 
                         v_dependent= True, 
                         u_dependent= True)

    def calc(self, data, x, u = None):
        if u is None:
            u = self.unone
        pin.forwardKinematics(self.model, self.data, x[:self.model.nq], x[self.model.nq:])
        pin.updateFramePlacements(self.model, self.data)
        self.actuation.calc(self.actuation_data, x, u)
        data.r = self.actuation_data.tau
        

    def calcDiff(self, data, x, u = None):
        if u is None:
            u = self.unone
        pin.forwardKinematics(self.model, self.data, x[:self.model.nq], x[self.model.nq:])
        pin.updateFramePlacements(self.model, self.data)
        self.actuation.calcDiff(self.actuation_data, x, u)
        a = pin.aba(self.model, self.data, x[:self.model.nq], x[self.model.nq:], u)
        pin.computeRNEADerivatives(self.model, self.data, x[:self.model.nq], x[self.model.nq:], a)
        dtau_dq = self.data.dtau_dq
        dtau_dv = self.data.dtau_dv
        data.Rx = np.hstack([dtau_dq, dtau_dv])
        data.Ru = np.eye(self.nu)

'''
Residual Model for Joint Velocity: 
Cost = (1/2) * q_dot^T * q_dot
R = q_dot
Rx = I
Ru = d(q_dot)/du = 0
'''
class ResidualModelJointVelocity(crocoddyl.ResidualModelAbstract):
    def __init__(self, state):
        self.unone = np.zeros(state.nv)
        self.model = state.pinocchio
        self.data = self.model.createData()
        super().__init__(state = state, 
                         nr = state.nv, 
                         nu = state.nv, 
                         q_dependent=False, 
                         v_dependent= True, 
                         u_dependent=False)

    def calc(self, data, x, u = None):
        if u is None:
            u = self.unone
        pin.forwardKinematics(self.model, self.data, x[:self.model.nq], x[self.model.nq:])
        pin.updateFramePlacements(self.model, self.data)
        data.r = x[self.nr:]
        

    def calcDiff(self, data, x, u = None):
        if u is None:
            u = self.unone
        data.Rx = np.eye(self.nr)
        data.Ru = np.zeros((self.nr, self.nu))

'''
Residual Model for Robot States: 
Cost = (1/2) * x^T * x
R = x
Rx = d(x)/dx = I
Ru = d(x)/du = 0
'''
class ResidualModelState(crocoddyl.ResidualModelAbstract):
    def __init__(self, state):
        self.unone = np.zeros(state.nv)
        self.model = state.pinocchio
        self.data = self.model.createData()
        super().__init__(state = state, 
                         nr = state.ndx, 
                         nu = state.nv, 
                         q_dependent= True, 
                         v_dependent= True, 
                         u_dependent= False)

    def calc(self, data, x, u = None):
        if u is None:
            u = self.unone
        pin.forwardKinematics(self.model, self.data, x[:self.model.nq], x[self.model.nq:])
        pin.updateFramePlacements(self.model, self.data)
        data.r = x
        

    def calcDiff(self, data, x, u = None):
        if u is None:
            u = self.unone
        data.Rx = np.eye(self.state.ndx)
        data.Ru = np.zeros((self.nr, self.nu))

'''
Residual Model for Joint Acceleration: 
Cost = (1/2) * q_ddot^T * q_ddot
R = q_ddot --> from pin.aba
Rx = d(q_ddot)/dx = [d(q_ddot)/dq, d(q_ddot)/d(q_dot)] = [ddq_dq, ddq_dv] --> from pin.computeABADerivatives
Ru = d(q_ddot)/du = M^{-1} --> from pin.computeABADerivatives
'''
class ResidualModelJointAcceleration(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, actuation):
        self.unone = np.zeros(state.nv)
        self.actuation = actuation
        self.actuation_data = self.actuation.createData()
        self.model = state.pinocchio
        self.data = self.model.createData()
        super().__init__(state = state, 
                         nr = state.nv, 
                         nu = actuation.nu, 
                         q_dependent= True, 
                         v_dependent= True, 
                         u_dependent= True)

    def calc(self, data, x, u = None):
        if u is None:
            u = self.unone
        pin.forwardKinematics(self.model, self.data, x[:self.model.nq], x[self.model.nq:])
        pin.updateFramePlacements(self.model, self.data)
        self.actuation.calc(self.actuation_data, x, u)
        pin.aba(self.model, self.data, x[:self.model.nq], x[self.model.nv:], self.actuation_data.tau)
        data.r = self.data.ddq

    def calcDiff(self, data, x, u = None):
        if u is None:
            u = self.unone
        pin.computeABADerivatives(self.model, self.data, x[:self.model.nq], x[self.model.nv:] ,self.actuation_data.tau)
        data.Rx = np.concatenate([self.data.ddq_dq.T, self.data.ddq_dv.T]).T
        data.Ru = self.data.Minv

'''
Residual Model for Frame Velocity: (only for X and Z axes)
Cost = (1/2) * V^T * V   s.t.   V = [x_dot, 0, z_dot]
R = V --> from pin.getFrameVelocity
Rx = d(V)/dx = [d(V)/dq, d(V)/d(q_dot)] --> from pin.getFrameVelocityDerivatives
Ru = d(V)/du = 0
'''

class ResidualModelFrameVelocity(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, frame_id, fr_type):
        self.unone = np.zeros(state.nv)
        self.model = state.pinocchio
        self.data = self.model.createData()
        self.frame_id = frame_id
        self.fr_type = fr_type
        super().__init__(state = state, 
                         nr = 2, 
                         nu = state.nv, 
                         q_dependent= True, 
                         v_dependent= True, 
                         u_dependent= False)

    def calc(self, data, x, u = None):
        if u is None:
            u = self.unone
        pin.forwardKinematics(self.model, self.data, x[:self.model.nq], x[self.model.nq:])
        pin.updateFramePlacements(self.model, self.data)
        v = pin.getFrameVelocity(self.model, self.data, self.frame_id, self.fr_type).vector
        data.r = np.array([v[0], v[2]])
        

    def calcDiff(self, data, x, u = None):
        if u is None:
            u = self.unone
        a = pin.aba(self.model, self.data, x[:self.model.nq], x[self.model.nq:], u)
        pin.computeForwardKinematicsDerivatives(self.model, self.data, x[:self.model.nq], x[self.model.nq:], a)
        Rq, Rv = pin.getFrameVelocityDerivatives(self.model, self.data, self.frame_id, self.fr_type)
        Rx = np.hstack([Rq, Rv])
        data.Rx[0, :] = Rx[0, :]
        data.Rx[1, :] = Rx[2, :]
        data.Ru = np.zeros((self.nr, self.nu))
        

'''
Residual Model for Joint Torque Change:
Cost = (1/2) * tau_dot^T * tau_dot
R = tau_dot = dtau/dt = (dtau/dq)*(dq/dt) =  dtau_dq * q_dot--> from crocoddyl actuation model (calc, calcDiff)
Rx and Ru are computed by numerical differentiation
'''
class ResidualModelTorqueChange(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, actuation):
        self.unone = np.zeros(state.nv)
        self.model = state.pinocchio
        self.data = self.model.createData()
        self.actuation = actuation
        self.actuation_data = self.actuation.createData()
        super().__init__(state = state, 
                         nr = actuation.nu, 
                         nu = actuation.nu, 
                         q_dependent= True, 
                         v_dependent= True, 
                         u_dependent= True)

    def calc(self, data, x, u = None):
        if u is None:
            u = self.unone
        pin.forwardKinematics(self.model, self.data, x[:self.model.nq], x[self.model.nq:])
        pin.updateFramePlacements(self.model, self.data)
        self.actuation.calc(self.actuation_data, x, u)
        self.actuation.calcDiff(self.actuation_data, x, u)
        q_dot = x[self.model.nq:]
        a = pin.aba(self.model, self.data, x[:self.model.nq], x[self.model.nq:], u)
        pin.computeRNEADerivatives(self.model, self.data, x[:self.model.nq], x[self.model.nq:], a)
        dtau_dq = self.data.dtau_dq
        data.r = np.squeeze(dtau_dq@q_dot[:,None])

'''
Residual Model for Energy:
Cost = Sum (q_dot(i)*tau(i))**2
R = q_dot^T @ I @ tau
Rx = [q_dot^T dtau_dq , q_dot^T @ dtau_ddq + tau]
Ru = [q_dot]
'''

class ResidualModelEnergy(crocoddyl.ResidualModelAbstract):
    """Energy cost: r[i] = q̇_i · u_i for each joint i in `joint_indices`.

    With Crocoddyl's default quadratic activation, the cost is
        0.5 · Σ_{i ∈ joint_indices} (q̇_i · u_i)²
    i.e. squared per-joint power summed over the selected joint subset.

    Pass `joint_indices=None` (default) to include ALL actuator DOFs (original
    behaviour). Pass a list of indices (e.g. [0] for "thoracic only") to get
    a per-joint-group energy feature — useful for IRL analyses that want to
    distinguish which body region carries the work.
    """
    def __init__(self, state, actuation, joint_indices=None):
        self.unone = np.zeros(state.nv)
        self.model = state.pinocchio
        self.data = self.model.createData()
        self.actuation = actuation
        self.actuation_data = self.actuation.createData()
        if joint_indices is None:
            self._joint_indices = np.arange(actuation.nu, dtype=int)
        else:
            self._joint_indices = np.asarray(joint_indices, dtype=int)
        super().__init__(state = state,
                         nr = len(self._joint_indices),
                         nu = actuation.nu,
                         q_dependent= True,
                         v_dependent= True,
                         u_dependent= True)

    def calc(self, data, x, u = None):
        if u is None:
            u = self.unone
        pin.forwardKinematics(self.model, self.data, x[:self.model.nq], x[self.model.nq:])
        pin.updateFramePlacements(self.model, self.data)
        self.actuation.calc(self.actuation_data, x, u)
        tau = u
        q_dot = x[self.state.nq:]
        idx = self._joint_indices
        r = q_dot[idx] * tau[idx]
        if np.ndim(data.r) == 0:
            data.r = float(r[0])
        elif data.r.shape[0] == 1 and r.shape[0] == 1:
            data.r[0] = float(r[0])
        else:
            data.r[:] = r

    def calcDiff(self, data, x, u = None):
        if u is None:
            u = self.unone
        q_ddot = pin.aba(self.model, self.data, x[:self.model.nq], x[self.model.nq:], u)
        pin.computeRNEADerivatives(self.model, self.data, x[:self.model.nq], x[self.model.nq:], q_ddot)
        self.actuation.calcDiff(self.actuation_data, x, u)
        tau = u
        dtau_dq = self.data.dtau_dq
        dtau_dv = self.data.dtau_dv
        q_dot = x[self.model.nq:]
        idx = self._joint_indices
        nv = self.model.nv
        e_idx = np.eye(nv)[idx]
        Rx_q = np.diag(q_dot[idx]) @ dtau_dq[idx, :]
        Rx_v = np.diag(q_dot[idx]) @ dtau_dv[idx, :] + np.diag(tau[idx]) @ e_idx
        Ru   = np.diag(q_dot[idx]) @ e_idx
        if data.Rx.ndim == 1:
            data.Rx[:nv] = Rx_q[0]
            data.Rx[nv:] = Rx_v[0]
            data.Ru[:]   = Ru[0]
        else:
            data.Rx[:, :nv] = Rx_q
            data.Rx[:, nv:] = Rx_v
            data.Ru[:, :]   = Ru
        

'''
Residual Model for Geodesic cost feature:
Cost = q_dot^T * M * q_dot 
R = q_dot^T * M^{1/2}    s.t.   if U and D are cholesky decomposition of M (mass matrix) --> M^{1/2} = U * sqrt(diag(D))
Rx and Ru are computed by numerical differentiation
'''

class ResidualModelGeodesic(crocoddyl.ResidualModelAbstract):
    def __init__(self, state):
        self.unone = np.zeros(state.nv)
        self.model = state.pinocchio
        self.data = self.model.createData()
        super().__init__(state = state, 
                         nr = state.nq, 
                         nu = state.nv, 
                         q_dependent= True, 
                         v_dependent= True, 
                         u_dependent= False)

    def calc(self, data, x, u = None):
        if u is None:
            u = self.unone
        pin.forwardKinematics(self.model, self.data, x[:self.model.nq], x[self.model.nq:])
        pin.updateFramePlacements(self.model, self.data)
        q_dot = x[self.state.nq:][:,None]
        pin.crba(self.model, self.data, x[:self.model.nq])
        pin.cholesky.decompose(self.model, self.data)
        U = self.data.U
        DS = np.diag(np.sqrt(self.data.D))
        M_cholesky = U@DS
        data.r = np.squeeze(q_dot.T@M_cholesky)

'''
Residual Model for X axis position:
R = x
Rx = dR/dX = [dR/dq , dR/dq_dot] = [J[0,:nq] , 0]
Ru = 0
Note: This is used for constraint model on terminal X value
'''

class ResidualModelX(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, frame_id, x_ref=None):
        self.unone = np.zeros(state.nv)
        self.frame_id = frame_id
        self.x_ref = x_ref
        if self.x_ref is None:
            self.x_ref = 0.0
        self.model = state.pinocchio
        self.data = self.model.createData()
        super().__init__(state = state, 
                         nr = 1, 
                         nu = 2, 
                         q_dependent= True, 
                         v_dependent= False, 
                         u_dependent= False)

    def calc(self, data, x, u = None):
        if u is None:
            u = self.unone
        pin.forwardKinematics(self.model, self.data, x[:self.model.nq], x[self.model.nq:])
        pin.updateFramePlacements(self.model, self.data)
        x = self.data.oMf[self.frame_id].translation[0]
        data.r[0] = x - self.x_ref

    def calcDiff(self, data, x, u = None):
        if u is None:
            u = self.unone
        pin.computeJointJacobians(self.model, self.data, x[:self.model.nq])
        J = pin.getFrameJacobian(self.model, self.data, self.frame_id, pin.LOCAL_WORLD_ALIGNED)
        data.Rx[:self.nu] = J[0,:]

'''
Constraint Residual on X axis position, given the goal X (param['pxf']):
H = x - pxf
Hx = Rx (<-- from the X residual)
Hu = 0
'''

class ConstraintModelResidualFrameTranslationX(crocoddyl.ConstraintModelAbstract):
    def __init__(self, state, residual, lb, ub):
        self.unone = np.zeros(state.nv)
        self.model = state.pinocchio
        super().__init__(state = state, 
                         residual = residual,
                         nh = 0, 
                         ng = 1)

    def calc(self, data, x, u = None):
        if u is None:
            u = self.unone
        res_data = self.residual.createData(crocoddyl.DataCollectorAbstract())
        self.residual.calc(res_data, x, u)
        data.g = res_data.r
        

    def calcDiff(self, data, x, u = None):
        if u is None:
            u = self.unone
        res_data = self.residual.createData(crocoddyl.DataCollectorAbstract())
        self.residual.calcDiff(res_data, x, u)
        data.Gx = res_data.Rx

'''
Residual Model for Numerical Differentiation
Direct mimic of the C++ codes of crocoddyl's NumDiff package (residual.cpp and residual.hpp)
Note: The python binding of the residual function numerical differentiation of crocoddyl is missing.
'''
class ResidualModelNumDiff(crocoddyl.ResidualModelAbstract):
    def __init__(self, res_model):
        self.e_jac = 1e-07
        self.model = res_model.model
        self.res_model = res_model
        self.unone = np.zeros(self.res_model.state.nv)
        super().__init__(state = self.res_model.state, 
                         nr = self.res_model.nr, 
                         nu = self.res_model.nu, 
                         q_dependent= self.res_model.q_dependent, 
                         v_dependent= self.res_model.v_dependent, 
                         u_dependent= self.res_model.u_dependent)
        self.nx = self.state.nx
        
        self.appendData()

    def calc(self, data, x, u = None):
        if u is None:
            u = self.unone
        self.res_model.calc(self.data_0, x, u)
        data.r = self.data_0.r

    def calcDiff(self, data, x, u = None):
        if u is not None:
            data.Rx = np.zeros((self.nu, self.nx))
            data.Ru = np.zeros((self.nu, self.nu))
            self.dx = np.zeros(self.nx)
            self.du = np.zeros(self.nu)
            
            r0 = data.r
            self.dx = self.state.diff(self.state.zero(), x)
            self.x_norm = np.linalg.norm(self.dx)
            self.dx = np.zeros_like(self.dx)
    
            self.xh_jac = self.e_jac * np.max([1.0, self.x_norm])
            if self.xh_jac == 0:
                    self.xh_jac = self.e_jac
            for ix in range(self.state.ndx):
                self.dx[ix] = self.xh_jac
                self.xp = self.state.integrate(x, self.dx)
                self.res_model.calc(self.data_x[ix], self.xp, u)
                data.Rx[:,ix] = (self.data_x[ix].r - r0) / self.xh_jac
                self.dx[ix] = 0.
            
            self.uh_jac = self.e_jac * np.max([1.0, np.linalg.norm(u)])
            for iu in range(self.nu):
                self.du[iu] = self.uh_jac
                self.up = u + self.du
                self.res_model.calc(self.data_u[iu], x, self.up)
                data.Ru[:,iu] = (self.data_u[iu].r - r0) / self.uh_jac
                self.du[iu] = 0.
        else:
            r0 = data.r
            self.dx = np.zeros_like(self.dx)
            for ix in range(self.state.ndx):
                self.dx[ix] = self.xh_jac
                self.xp = self.state.integrate(x, self.dx)
                self.res_model.calc(self.data_x[ix], self.xp)
                if self.xh_jac == 0:
                    self.xh_jac = self.e_jac
                data.Rx[:,ix] = (self.data_x[ix].r - r0) / self.xh_jac
                self.dx[ix] = 0.

    def appendData(self):
        self.data_0 = self.model.createData()
        self.data_x = []
        self.data_u = []
        for i in range(self.state.ndx):
            self.data_x.append(self.model.createData())
        for i in range(self.nu):
            self.data_u.append(self.model.createData())
        self.dx = np.zeros(self.state.ndx)
        self.du = np.zeros(self.nu)
        self.x_norm = 0.0
        self.xh_jac = 0.0
        self.uh_jac = 0.0
        self.xp = np.zeros(self.state.nx)
        self.up = np.zeros(self.nu)

        
            
    
            
