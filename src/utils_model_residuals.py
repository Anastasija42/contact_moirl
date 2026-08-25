"""Residual models beyond the core feature library.

Contact-aware torque-change, actuation and collision-distance residuals, the
numerical-difference fallback, and the constrained-acceleration helpers used to
project recorded motion onto contact-consistent dynamics.
"""
import math
import os
import sys
import numpy as np
import pinocchio as pin
import crocoddyl
import mujoco
try:
    import friction_lib as _friction_lib
except Exception:
    _friction_lib = None

class ResidualModelNumDiff(crocoddyl.ResidualModelAbstract):
    def __init__(self, res_model):
        self.e_jac = 1e-07
        self.model = res_model.pin_model
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

class ResidualModelTorqueChange(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, actuation):
        """
        Computes a partial measure of torque change: r = (dtau_dq) * v.
        This represents the "Stiffness" contribution to the torque change.
        """
        self.pin_model = state.pinocchio
        self.pin_data = self.pin_model.createData()
        self.actuation = actuation
        
        super().__init__(state=state, 
                         nr=actuation.nu, 
                         nu=actuation.nu, 
                         q_dependent=True, 
                         v_dependent=True, 
                         u_dependent=True)

    def _get_stiffness(self, q, v, u):
        """Helper to compute dtau_dq"""
        pin.aba(self.pin_model, self.pin_data, q, v, u)
        
        pin.computeRNEADerivatives(
            self.pin_model, 
            self.pin_data, 
            q, 
            v, 
            self.pin_data.ddq
        )
        return self.pin_data.dtau_dq

    def calc(self, data, x, u=None):
        if u is None: u = np.zeros(self.nu)
        
        q = x[:self.state.nq]
        v = x[self.state.nq:]
        
        dtau_dq = self._get_stiffness(q, v, u)
        data.r[:] = dtau_dq @ v

    def calcDiff(self, data, x, u=None):
        if u is None: u = np.zeros(self.nu)

        eps = 1e-6
        dx = np.zeros(self.state.ndx)
        du = np.zeros(self.nu)
        
        q = x[:self.state.nq]
        v = x[self.state.nq:]
        dtau_dq_0 = self._get_stiffness(q, v, u)
        r0 = dtau_dq_0 @ v
        
        for i in range(self.state.ndx):
            dx[i] = eps
            x_plus = self.state.integrate(x, dx)
            
            q_plus = x_plus[:self.state.nq]
            v_plus = x_plus[self.state.nq:]
            
            dtau_dq_plus = self._get_stiffness(q_plus, v_plus, u)
            r_plus = dtau_dq_plus @ v_plus
            
            data.Rx[:, i] = (r_plus - r0) / eps
            dx[i] = 0.0

        for i in range(self.nu):
            du[i] = eps
            u_plus = u + du
            
            dtau_dq_plus = self._get_stiffness(q, v, u_plus)
            r_plus = dtau_dq_plus @ v
            
            data.Ru[:, i] = (r_plus - r0) / eps
            du[i] = 0.0

class ResidualCollisionDistance(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, geom_model, pair_id):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 1)
        self.geom_model = geom_model
        self.pair_id = pair_id
        
    def calc(self, data, x, u=None):
        q = x[:self.state.nq]
        
        if not hasattr(data, 'pin_data'):
            data.pin_data = self.state.pinocchio.createData()
            data.geom_data = self.geom_model.createData()
            
        pin.framesForwardKinematics(self.state.pinocchio, data.pin_data, q)
        pin.updateGeometryPlacements(self.state.pinocchio, data.pin_data, self.geom_model, data.geom_data, q)
        
        pin.computeDistance(self.geom_model, data.geom_data, self.pair_id)
        
        dist = data.geom_data.distanceResults[self.pair_id].min_distance
        data.r[0] = dist

    def calcDiff(self, data, x, u=None):
        
        eps = 1e-6
        ndx = self.state.ndx
        dist0 = data.r[0] 
        
        if not hasattr(data, 'pin_data_eps'):
            data.pin_data_eps = self.state.pinocchio.createData()
            data.geom_data_eps = self.geom_model.createData()

        for i in range(ndx):
            dx = np.zeros(ndx)
            dx[i] = eps
            x_eps = self.state.integrate(x, dx)
            q_eps = x_eps[:self.state.nq]
            
            pin.framesForwardKinematics(self.state.pinocchio, data.pin_data_eps, q_eps)
            pin.updateGeometryPlacements(self.state.pinocchio, data.pin_data_eps, self.geom_model, data.geom_data_eps, q_eps)
            pin.computeDistance(self.geom_model, data.geom_data_eps, self.pair_id)
            dist_eps = data.geom_data_eps.distanceResults[self.pair_id].min_distance
            
            data.Rx[i] = (dist_eps - dist0) / eps
"""            
class ResidualModelEnergy(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, nu):

        super().__init__(state=state, 
                         nr=state.nv, 
                         nu=nu, 
                         q_dependent=False, 
                         v_dependent=True, 
                         u_dependent=True)

    def calc(self, data, x, u=None):
        if u is None: u = np.zeros(self.nu)
        v = x[self.state.nq:]
        data.r[:] = u * v

    def calcDiff(self, data, x, u=None):
        if u is None: u = np.zeros(self.nu)
        v = x[self.state.nq:]
        # d(u*v)/dq = 0
        data.Rx[:, :self.state.nv] = 0.0
        # d(u*v)/dv = diag(u)
        data.Rx[:, self.state.nv:] = np.diag(u)
        # d(u*v)/du = diag(v)
        data.Ru[:, :] = np.diag(v)
"""

class ResidualModelActuationTau(crocoddyl.ResidualModelAbstract):
    """Contact-aware joint-torque residual: r = actuation.tau = u + J_c^T·f_fric.

    Unlike crocoddyl's ResidualModelControl (r = u), this includes the
    friction-overcoming torque injected by the friction-aware actuation model,
    so the OCP's Tau cost equals the IRL Tau feature 0.5·‖tau_act‖² computed in
    HumanCrocoddyl.get_traj_features (which reads actuation.tau). Use only when
    args['contact_aware_cost'] is set.

    Derivatives are NUMERICAL. friction_lib's ActuationModelFriction.calcDiff
    sets dtau_du = I and dtau_dx = 0 (it linearises as if tau = u), so the
    analytic act_data.dtau_* are inconsistent with the contact-aware *value*
    and would break Gauss-Newton. Finite-differencing actuation.calc keeps
    calc/calcDiff self-consistent (same pattern as ResidualModelGeodesic /
    ResidualModelTorqueChange in this file).
    """
    def __init__(self, state, actuation, eps=1e-6):
        self.actuation = actuation
        self.actuation_data = actuation.createData()
        self.eps = eps
        super().__init__(state=state,
                         nr=actuation.nu,
                         nu=actuation.nu,
                         q_dependent=True,
                         v_dependent=True,
                         u_dependent=True)

    def _tau(self, x, u):
        self.actuation.calc(self.actuation_data, x, u)
        return np.asarray(self.actuation_data.tau).copy()

    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)
        data.r[:] = self._tau(x, u)

    def calcDiff(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)
        eps = self.eps
        r0 = self._tau(x, u)
        dx = np.zeros(self.state.ndx)
        for i in range(self.state.ndx):
            dx[i] = eps
            xp = self.state.integrate(x, dx)
            data.Rx[:, i] = (self._tau(xp, u) - r0) / eps
            dx[i] = 0.0
        du = np.zeros(self.nu)
        for i in range(self.nu):
            du[i] = eps
            data.Ru[:, i] = (self._tau(x, u + du) - r0) / eps
            du[i] = 0.0

def jtc_id_residual(model, mdata, q, v, a):
    """Inverse-dynamics torque-rate residual (jerk term dropped):

        r = dtau_dq·v + dtau_dv·a ,   tau_id(q,v,a) = M(q)a + C(q,v)v + g(q) = rnea(q,v,a)

    Differentiating tau_id along the trajectory gives
    d(tau_id)/dt = dtau_dq·v + dtau_dv·a + M·ȧ, so r is the full torque rate
    minus only the jerk term M·ȧ. With `a` = the CONTACT-CONSTRAINED
    acceleration, tau_id is the total joint generalized force
    (= tau_act + J_c^T·λ_contact = muscle + friction + contact reaction), so r
    captures movement AND friction per-node, no state augmentation.

    SHARED by the OCP cost (ResidualModelTorqueChangeContactAware, a = DAM xout
    read from shared.pinocchio.ddq) and the IRL feature
    (HumanCrocoddyl.get_traj_features, a = Δv/dt). These accelerations are equal
    at any dynamically-feasible trajectory (Euler: v_{t+1}=v_t+dt·xout), so the
    solver cost == the feature at the solution by construction. Returns an
    nv-vector.
    """
    pin.computeRNEADerivatives(model, mdata, q, v, a)
    return mdata.dtau_dq @ v + mdata.dtau_dv @ a

def constrained_accel_1d(model, mdata, q, v, tau, frame_id, R_axis, ref=pin.LOCAL):
    """Contact-constrained acceleration under the model's 1D point contact
    (the z-axis of R_axis at `frame_id`), via pin.forwardDynamics. Drift-only
    (Baumgarte stabilisation omitted — negligible once the contact is tracked).
    Used ONLY to recompute `a` at perturbed states for the numdiff gradient of
    ResidualModelTorqueChangeContactAware; the cost VALUE uses the exact DAM
    acceleration (shared.pinocchio.ddq), so parity does not depend on this
    matching the DAM bit-for-bit.
    """
    pin.computeAllTerms(model, mdata, q, v)
    J6 = pin.getFrameJacobian(model, mdata, frame_id, ref)
    Jc = (R_axis.T @ J6[:3])[2:3, :]
    pin.forwardKinematics(model, mdata, q, v, np.zeros(model.nv))
    drift = pin.getFrameClassicalAcceleration(model, mdata, frame_id, ref).linear
    gamma = (R_axis.T @ drift)[2:3]
    return np.asarray(pin.forwardDynamics(model, mdata, q, v, tau, Jc, gamma)).copy()

class ResidualModelTorqueChangeContactAware(crocoddyl.ResidualModelAbstract):
    """Contact-aware per-node torque-rate cost via the inverse-dynamics form:

        r = dtau_dq·v + dtau_dv·a  (jtc_id_residual), a = constrained acceleration.

    Replaces the rnea stiffness proxy ResidualModelTorqueChange when the OCP is
    run with args['contact_aware_cost']. Captures movement + friction + contact
    reaction per-node (drops only the jerk term), no state augmentation. The IRL
    JTC feature uses the SAME jtc_id_residual with a = Δv/dt, so cost == feature
    at the solution.

    calc reads the EXACT constrained acceleration from the DAM
    (data.shared.pinocchio.ddq) → exact value-parity with the feature.
    calcDiff is numerical and self-contained: it recomputes the constrained `a`
    at each perturbed state via constrained_accel_1d (the analytic rnea/aba
    derivatives in the original residual assume FREE dynamics + ActuationFull
    and would be inconsistent here).
    """
    def __init__(self, state, actuation, contact_frame_id, R_axis,
                 ref=pin.LOCAL, eps=1e-6):
        self.model = state.pinocchio
        self.data = self.model.createData()
        self.actuation = actuation
        self.actuation_data = actuation.createData()
        self.contact_frame_id = None if contact_frame_id is None else int(contact_frame_id)
        self.R_axis = None if R_axis is None else np.asarray(R_axis, dtype=float)
        self.ref = ref
        self.eps = eps
        super().__init__(state=state,
                         nr=state.nv,
                         nu=actuation.nu,
                         q_dependent=True,
                         v_dependent=True,
                         u_dependent=True)

    def _tau(self, x, u):
        self.actuation.calc(self.actuation_data, x, u)
        return np.asarray(self.actuation_data.tau).copy()

    def _r_fd(self, x, u):
        """Self-contained residual (constrained `a` via forwardDynamics) — used
        for the numdiff gradient so the acceleration's dependence on (x,u) is
        captured. Free phase (no contact frame): plain forward dynamics."""
        nq, nv = self.model.nq, self.model.nv
        q = np.asarray(x[:nq]); v = np.asarray(x[nq:nq + nv])
        tau = self._tau(x, u)
        if self.contact_frame_id is None:
            a = pin.aba(self.model, self.data, q, v, tau)
        else:
            a = constrained_accel_1d(self.model, self.data, q, v, tau,
                                     self.contact_frame_id, self.R_axis, self.ref)
        return jtc_id_residual(self.model, self.data, q, v, a)

    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)
        nq, nv = self.model.nq, self.model.nv
        q = np.asarray(x[:nq]); v = np.asarray(x[nq:nq + nv])
        a = np.asarray(data.shared.pinocchio.ddq).copy()
        data.r[:] = jtc_id_residual(self.model, self.data, q, v, a)

    def _jtc_analytic(self, data, x, u):
        """ANALYTIC JTC Jacobian. r = g(q,v,a), a = constrained accel:
          dr/dx = dg/dx|_a + dg/da . da/dx_total ,  da/dx_total = da/dx_direct + da/dtau . dtau/dx.
        da/dx via KKT sensitivity (ONE forwardDynamics, no per-perturbation solve):
          da/dtau = Minv - Minv Jc^T (1/K) Jc Minv   (constrained inverse inertia)
          [M Jc^T; Jc 0][da/dq; -dl/dq] = [-dtau_dq + dJc^T/dq.lam ; -d(Jc.a)/dq - dgamma/dq]
          [M Jc^T; Jc 0][da/dv; -dl/dv] = [-dtau_dv ; -dgamma/dv]
        dg via 2nd-order RNEA (friction_lib.jtc_dg); dtau/dx from the friction map
        (calcDiff_analytic). Verified vs the FD to solver precision."""
        model = self.model; d = self.data; nv = model.nv; nq = model.nq; eps = self.eps
        fid = self.contact_frame_id; Rax = self.R_axis; ref = self.ref
        q = np.asarray(x[:nq]); v = np.asarray(x[nq:nq + nv])
        tau = self._tau(x, u)
        a0 = constrained_accel_1d(model, d, q, v, tau, fid, Rax, ref)
        pin.computeAllTerms(model, d, q, v); M = d.M.copy(); Minv = np.linalg.inv(M)
        Jc = (Rax.T @ pin.getFrameJacobian(model, d, fid, ref)[:3])[2:3, :]
        KKT = np.block([[M, Jc.T], [Jc, np.zeros((1, 1))]])
        K = float(Jc @ Minv @ Jc.T)
        a_free = pin.aba(model, d, q, v, tau)
        def gam(qq, vv):
            pin.forwardKinematics(model, d, qq, vv, np.zeros(nv))
            return float((Rax.T @ pin.getFrameClassicalAcceleration(model, d, fid, ref).linear)[2])
        g0 = gam(q, v); lam = -(float(Jc @ a_free) + g0) / K
        pin.computeRNEADerivatives(model, d, q, v, a0)
        dtau_dq = d.dtau_dq.copy(); dtau_dv = d.dtau_dv.copy()
        def JcTl(qq):
            pin.computeAllTerms(model, d, qq, v)
            return ((Rax.T @ pin.getFrameJacobian(model, d, fid, ref)[:3])[2:3, :]).ravel() * lam
        def Jca(qq):
            pin.computeAllTerms(model, d, qq, v)
            return float((Rax.T @ pin.getFrameJacobian(model, d, fid, ref)[:3])[2:3, :] @ a0)
        JcTl0 = JcTl(q); Jca0 = Jca(q); ei = np.eye(nv)
        dJcTl = np.column_stack([(JcTl(pin.integrate(model, q, eps*ei[i])) - JcTl0)/eps for i in range(nv)])
        dJca  = np.array([[(Jca(pin.integrate(model, q, eps*ei[i])) - Jca0)/eps for i in range(nv)]])
        dgq   = np.array([[(gam(pin.integrate(model, q, eps*ei[i]), v) - g0)/eps for i in range(nv)]])
        dgv   = np.array([[(gam(q, v + eps*ei[i]) - g0)/eps for i in range(nv)]])
        da_dq_dir = np.linalg.solve(KKT, np.vstack([-dtau_dq + dJcTl, -dJca - dgq]))[:nv]
        da_dv_dir = np.linalg.solve(KKT, np.vstack([-dtau_dv, -dgv]))[:nv]
        da_dtau = Minv - (Minv @ Jc.T @ Jc @ Minv) / K
        self.actuation.calcDiff_analytic(self.actuation_data, x, u)
        dtaudx = np.asarray(self.actuation_data.dtau_dx); dtaudu = np.asarray(self.actuation_data.dtau_du)
        da_dq = da_dq_dir + da_dtau @ dtaudx[:, :nv]
        da_dv = da_dv_dir + da_dtau @ dtaudx[:, nv:]
        da_du = da_dtau @ dtaudu
        dg_dq, dg_dv, dg_da = [np.asarray(m) for m in self.actuation.jtc_dg(self.actuation_data, x, a0)]
        data.Rx[:, :nv] = dg_dq + dg_da @ da_dq
        data.Rx[:, nv:] = dg_dv + dg_da @ da_dv
        data.Ru[:, :]   = dg_da @ da_du

    def calcDiff(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)
        global _friction_lib
        if _friction_lib is None:
            try:
                import friction_lib as _friction_lib
            except Exception:
                _friction_lib = None
        if (self.contact_frame_id is not None and _friction_lib is not None
                and hasattr(self.actuation, 'jtc_analytic_calcDiff')):
            try:
                Rx, Ru = self.actuation.jtc_analytic_calcDiff(
                    self.actuation_data, np.asarray(x, float), np.asarray(u, float),
                    int(self.contact_frame_id), np.asarray(self.R_axis, float), float(self.eps))
                data.Rx[:, :] = np.asarray(Rx)
                data.Ru[:, :] = np.asarray(Ru)
                return
            except Exception:
                pass
        if (self.contact_frame_id is not None and _friction_lib is not None
                and hasattr(self.actuation, 'calcTauJacobian')):
            Rx, Ru = _friction_lib.jtc_contact_calcDiff(
                self.actuation, self.actuation_data,
                np.asarray(x, float), np.asarray(u, float),
                int(self.contact_frame_id), np.asarray(self.R_axis, float),
                float(self.eps))
            data.Rx[:, :] = np.asarray(Rx)
            data.Ru[:, :] = np.asarray(Ru)
            return
        eps = self.eps
        r0 = self._r_fd(x, u)
        dx = np.zeros(self.state.ndx)
        for i in range(self.state.ndx):
            dx[i] = eps
            data.Rx[:, i] = (self._r_fd(self.state.integrate(x, dx), u) - r0) / eps
            dx[i] = 0.0
        du = np.zeros(self.nu)
        for i in range(self.nu):
            du[i] = eps
            data.Ru[:, i] = (self._r_fd(x, u + du) - r0) / eps
            du[i] = 0.0

class ResidualModelEnergy(crocoddyl.ResidualModelAbstract):
    """Energy cost: r[i] = q̇_i · τ_i for each joint i in `joint_indices`.

    Pass `joint_indices=None` (default) to include all DOFs (original
    behaviour). Pass a list (e.g. [0]) to restrict to a joint subset for
    per-joint-group energy features.

    `contact_aware` selects which torque enters the residual:
      * False (default) — τ = u (raw control), the original behaviour. Note the
        original calcDiff also used the RNEA dynamics dtau_dq/dtau_dv for the
        dr/dx term while using τ = u for the direct term; that is preserved
        verbatim for backward compatibility.
      * True  — τ = actuation.tau = u + J_c^T·f_fric, matching the IRL Eng
        feature in HumanCrocoddyl.get_traj_features. Derivatives are NUMERICAL
        (see ResidualModelActuationTau for why friction_lib's analytic
        actuation derivatives can't be used here).
    """
    def __init__(self, state, actuation, joint_indices=None, contact_aware=False):
        self.unone = np.zeros(state.nv)
        self.model = state.pinocchio
        self.data = self.model.createData()
        self.actuation = actuation
        self.actuation_data = self.actuation.createData()
        self.contact_aware = contact_aware
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

    def _resid(self, x, u):
        """Residual vector r[i] = q̇_i · τ_i over the joint subset."""
        self.actuation.calc(self.actuation_data, x, u)
        tau = np.asarray(self.actuation_data.tau) if self.contact_aware else u
        q_dot = x[self.state.nq:]
        idx = self._joint_indices
        return q_dot[idx] * tau[idx]

    def calc(self, data, x, u = None):
        if u is None:
            u = self.unone
        if not self.contact_aware:
            pin.forwardKinematics(self.model, self.data, x[:self.model.nq], x[self.model.nq:])
            pin.updateFramePlacements(self.model, self.data)
        r = self._resid(x, u)
        if np.ndim(data.r) == 0:
            data.r = float(r[0])
        elif data.r.shape[0] == 1 and r.shape[0] == 1:
            data.r[0] = float(r[0])
        else:
            data.r[:] = r

    def calcDiff(self, data, x, u = None):
        if u is None:
            u = self.unone
        nv = self.model.nv
        idx = self._joint_indices

        if self.contact_aware:
            if hasattr(self.actuation, 'calcDiff_analytic'):
                self.actuation.calcDiff_analytic(self.actuation_data, x, u)
                dtau_dx = np.asarray(self.actuation_data.dtau_dx)
                dtau_du = np.asarray(self.actuation_data.dtau_du)
                self.actuation.calc(self.actuation_data, x, u)
                tau = np.asarray(self.actuation_data.tau)
                q_dot = x[self.model.nq:]
                e_idx = np.eye(nv)[idx]
                Rx = np.diag(q_dot[idx]) @ dtau_dx[idx, :]
                Rx[:, nv:] += np.diag(tau[idx]) @ e_idx
                Ru = np.diag(q_dot[idx]) @ dtau_du[idx, :]
                if data.Rx.ndim == 1:
                    data.Rx[:] = Rx[0]
                    data.Ru[:] = Ru[0]
                else:
                    data.Rx[:, :] = Rx
                    data.Ru[:, :] = Ru
                return
            eps = 1e-6
            r0 = self._resid(x, u)
            Rx = np.zeros((len(idx), self.state.ndx))
            dx = np.zeros(self.state.ndx)
            for i in range(self.state.ndx):
                dx[i] = eps
                xp = self.state.integrate(x, dx)
                Rx[:, i] = (self._resid(xp, u) - r0) / eps
                dx[i] = 0.0
            Ru = np.zeros((len(idx), self.nu))
            du = np.zeros(self.nu)
            for i in range(self.nu):
                du[i] = eps
                Ru[:, i] = (self._resid(x, u + du) - r0) / eps
                du[i] = 0.0
            if data.Rx.ndim == 1:
                data.Rx[:] = Rx[0]
                data.Ru[:] = Ru[0]
            else:
                data.Rx[:, :] = Rx
                data.Ru[:, :] = Ru
            return

        q_ddot = pin.aba(self.model, self.data, x[:self.model.nq], x[self.model.nq:], u)
        pin.computeRNEADerivatives(self.model, self.data, x[:self.model.nq], x[self.model.nq:], q_ddot)
        self.actuation.calcDiff(self.actuation_data, x, u)
        tau = u
        dtau_dq = self.data.dtau_dq
        dtau_dv = self.data.dtau_dv
        q_dot = x[self.model.nq:]
        e_idx = np.eye(nv)[idx]
        Rx_q = np.diag(q_dot[idx]) @ dtau_dq[idx, :]
        Rx_v = np.diag(q_dot[idx]) @ dtau_dv[idx, :] + np.diag(tau[idx]) @ e_idx
        Ru   = np.diag(q_dot[idx]) @ e_idx
        if data.Rx.ndim == 1:
            data.Rx[:nv]  = Rx_q[0]
            data.Rx[nv:]  = Rx_v[0]
            data.Ru[:]    = Ru[0]
        else:
            data.Rx[:, :nv] = Rx_q
            data.Rx[:, nv:] = Rx_v
            data.Ru[:, :]   = Ru

class ResidualModelGroupEffort(crocoddyl.ResidualModelAbstract):
    """Merged per-joint-group effort: torque magnitude (force/holding) AND
    mechanical power (work) for a joint group, combined into ONE cost.

        r = [ tau_act[idx] ; (q_dot * tau_act)[idx] ]   (length 2*|idx|)
        0.5||r||^2 = 0.5(||tau[idx]||^2 + ||power[idx]||^2)
                   = Tau_group + Eng_group .

    A single weight per group (a metabolic-style effort), replacing the
    separate global Tau + per-group Eng with a symmetric per-group structure.
    Contact-aware (tau_act = u + J_c^T f_fric); numerical derivatives
    (friction_lib zeroes the analytic actuation derivatives — see
    ResidualModelActuationTau). Activated simply by putting `Effort_<group>`
    keys in w_run (the HumanCrocoddyl builders add this cost when present).
    """
    def __init__(self, state, actuation, joint_indices, eps=1e-6):
        self.actuation = actuation
        self.actuation_data = actuation.createData()
        self.idx = np.asarray(joint_indices, dtype=int)
        self.eps = eps
        super().__init__(state=state,
                         nr=2 * len(self.idx),
                         nu=actuation.nu,
                         q_dependent=True,
                         v_dependent=True,
                         u_dependent=True)

    def _r(self, x, u):
        self.actuation.calc(self.actuation_data, x, u)
        tau = np.asarray(self.actuation_data.tau)
        v = np.asarray(x[self.state.nq:])
        idx = self.idx
        return np.concatenate([tau[idx], v[idx] * tau[idx]])

    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)
        data.r[:] = self._r(x, u)

    def calcDiff(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)
        eps = self.eps
        r0 = self._r(x, u)
        dx = np.zeros(self.state.ndx)
        for i in range(self.state.ndx):
            dx[i] = eps
            data.Rx[:, i] = (self._r(self.state.integrate(x, dx), u) - r0) / eps
            dx[i] = 0.0
        du = np.zeros(self.nu)
        for i in range(self.nu):
            du[i] = eps
            data.Ru[:, i] = (self._r(x, u + du) - r0) / eps
            du[i] = 0.0

class ResidualModelGroupTau(crocoddyl.ResidualModelAbstract):
    """Per-joint-group torque (load / holding effort) for a joint group:

        r = tau_act[idx]            (length |idx|)
        0.5||r||^2 = Tau_group .

    The load-only counterpart of ResidualModelGroupEffort (which also adds the
    work term). One weight per group → a per-segment torque split of the global
    Tau. Contact-aware (tau_act = u + J_c^T f_fric); numerical derivatives
    (friction_lib zeroes the analytic actuation derivatives — see
    ResidualModelActuationTau). Activated by putting `Tau_<group>` keys in w_run.
    """
    def __init__(self, state, actuation, joint_indices, eps=1e-6):
        self.actuation = actuation
        self.actuation_data = actuation.createData()
        self.idx = np.asarray(joint_indices, dtype=int)
        self.eps = eps
        super().__init__(state=state,
                         nr=len(self.idx),
                         nu=actuation.nu,
                         q_dependent=True,
                         v_dependent=True,
                         u_dependent=True)

    def _r(self, x, u):
        self.actuation.calc(self.actuation_data, x, u)
        tau = np.asarray(self.actuation_data.tau)
        return np.atleast_1d(tau[self.idx])

    def calc(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)
        data.r[:] = self._r(x, u)

    def calcDiff(self, data, x, u=None):
        if u is None:
            u = np.zeros(self.nu)
        if hasattr(self.actuation, 'calcDiff_analytic'):
            self.actuation.calcDiff_analytic(self.actuation_data, x, u)
            Rx = np.asarray(self.actuation_data.dtau_dx)[self.idx, :]
            Ru = np.asarray(self.actuation_data.dtau_du)[self.idx, :]
            data.Rx[:] = Rx.reshape(data.Rx.shape)
            data.Ru[:] = Ru.reshape(data.Ru.shape)
            return
        eps = self.eps
        r0 = self._r(x, u)
        Rx = np.zeros((self.nr, self.state.ndx))
        dx = np.zeros(self.state.ndx)
        for i in range(self.state.ndx):
            dx[i] = eps
            Rx[:, i] = (self._r(self.state.integrate(x, dx), u) - r0) / eps
            dx[i] = 0.0
        Ru = np.zeros((self.nr, self.nu))
        du = np.zeros(self.nu)
        for i in range(self.nu):
            du[i] = eps
            Ru[:, i] = (self._r(x, u + du) - r0) / eps
            du[i] = 0.0
        data.Rx[:] = Rx.reshape(data.Rx.shape)
        data.Ru[:] = Ru.reshape(data.Ru.shape)

class ResidualModelGeodesic(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, nu):
        """
        Residual r = L^T * v, where M = L * L^T
        """
        self.pin_model = state.pinocchio
        self.pin_data = self.pin_model.createData()
        super().__init__(state=state, 
                         nr=state.nv, 
                         nu=nu, 
                         q_dependent=True, 
                         v_dependent=True, 
                         u_dependent=False)

    def _get_L(self, q):
        pin.crba(self.pin_model, self.pin_data, q)
        pin.cholesky.decompose(self.pin_model, self.pin_data)
        U = self.pin_data.U
        D_sqrt = np.diag(np.sqrt(self.pin_data.D))
        A = D_sqrt @ U.T
        return A

    def calc(self, data, x, u=None):
        q = x[:self.state.nq]
        v = x[self.state.nq:]
        A = self._get_L(q)
        data.r[:] = A @ v

    def calcDiff(self, data, x, u=None):
        q = x[:self.state.nq]
        v = x[self.state.nq:]
        
        A = self._get_L(q)
        data.Rx[:, self.state.nv:] = A

        eps = 1e-6
        dx = np.zeros(self.state.ndx)
        for i in range(self.state.nv):
            dx[i] = eps
            x_plus = self.state.integrate(x, dx)
            q_plus = x_plus[:self.state.nq]
            r_plus = self._get_L(q_plus) @ v 
            
            dx[i] = -eps
            x_minus = self.state.integrate(x, dx)
            q_minus = x_minus[:self.state.nq]
            r_minus = self._get_L(q_minus) @ v
            
            data.Rx[:, i] = (r_plus - r_minus) / (2 * eps)
            dx[i] = 0.0
        data.Ru[:, :] = 0.0

class ResidualRail(crocoddyl.ResidualModelAbstract):
    def __init__(self, state, frame_id, p_start, p_end, nu):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 3, nu, True, True, False)
        self.frame_id = frame_id
        self.p_start = p_start
        self.p_end = p_end

        rail_vec = p_end - p_start
        rail_dir = rail_vec / np.linalg.norm(rail_vec)
        self.R_rail = pin.Quaternion.FromTwoVectors(
            np.array([0.0, 0.0, 1.0]), rail_dir
        ).matrix()

    def calc(self, data, x, u=None):
        if not hasattr(data, "pin_data"):
            data.pin_data = self.state.pinocchio.createData()

        q = x[: self.state.nq]
        pin.forwardKinematics(self.state.pinocchio, data.pin_data, q)
        pin.updateFramePlacements(self.state.pinocchio, data.pin_data)

        p_hand = data.pin_data.oMf[self.frame_id].translation
        diff_world = p_hand - self.p_start
        diff_rail = self.R_rail.T @ diff_world

        data.r[:] = diff_rail

    def calcDiff(self, data, x, u=None):
        if not hasattr(data, "pin_data"):
            data.pin_data = self.state.pinocchio.createData()

        q = x[: self.state.nq]
        pin.forwardKinematics(self.state.pinocchio, data.pin_data, q)
        pin.updateFramePlacements(self.state.pinocchio, data.pin_data)

        J = pin.computeFrameJacobian(
            self.state.pinocchio,
            data.pin_data,
            q,
            self.frame_id,
            pin.LOCAL_WORLD_ALIGNED,
        )
        J_trans = J[:3, :]
        J_rail = self.R_rail.T @ J_trans

        data.Arr_Rx.fill(0.0)
        data.Arr_Rx[:, : self.state.nq] = J_rail

        if u is not None:
            data.Arr_Ru.fill(0.0)

class ResidualRailLateral(crocoddyl.ResidualModelAbstract):
    """Lateral (off-rail) deviation of a contact frame from the rail line.
    Returns a 2D residual in the plane perpendicular to the rail direction.
    Cost = (1/2) * w * ||lateral_deviation||^2
    """
    def __init__(self, state, frame_id, p_start, rail_unit, nu):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 2, nu, True, False, False)
        self.frame_id  = frame_id
        self.p_start   = np.array(p_start)
        self.rail_unit = np.array(rail_unit)
        e = np.array([1., 0., 0.])
        if abs(np.dot(self.rail_unit, e)) > 0.9:
            e = np.array([0., 1., 0.])
        perp1 = e - np.dot(e, self.rail_unit) * self.rail_unit
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(self.rail_unit, perp1)
        self.P_lat = np.stack([perp1, perp2])

    def calc(self, data, x, u=None):
        if not hasattr(data, 'pin_data'):
            data.pin_data = self.state.pinocchio.createData()
        q = x[:self.state.nq]
        pin.forwardKinematics(self.state.pinocchio, data.pin_data, q)
        pin.updateFramePlacements(self.state.pinocchio, data.pin_data)
        pos  = data.pin_data.oMf[self.frame_id].translation
        diff = pos - self.p_start
        data.r[:] = self.P_lat @ diff

    def calcDiff(self, data, x, u=None):
        if not hasattr(data, 'pin_data'):
            data.pin_data = self.state.pinocchio.createData()
        q = x[:self.state.nq]
        pin.forwardKinematics(self.state.pinocchio, data.pin_data, q)
        pin.updateFramePlacements(self.state.pinocchio, data.pin_data)
        J = pin.computeFrameJacobian(
            self.state.pinocchio, data.pin_data, q,
            self.frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        data.Rx.fill(0.0)
        data.Rx[:, :self.state.nq] = self.P_lat @ J[:3, :]

class ResidualLateralClearance(crocoddyl.ResidualModelAbstract):
    """Signed lateral clearance of a frame from a fixed reference point along a
    fixed lateral axis:  r = axis . (p_frame - p_ref)   (1D).

    Constrain r >= R (a hard lower bound) to keep e.g. the forearm/elbow on the
    lateral side of the trunk by a margin R, i.e. out of the torso. This is the
    only self-collision avoidance in the model: it is needed once a body's
    medially-rotated elbow (low humeral torsion) would otherwise route the limb
    through its own trunk, which the human geometry never does.
    """
    def __init__(self, state, frame_id, p_ref, axis, nu):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 1, nu, True, False, False)
        self.frame_id = frame_id
        self.p_ref = np.asarray(p_ref, dtype=float)
        a = np.asarray(axis, dtype=float)
        self.axis = a / (np.linalg.norm(a) + 1e-12)

    def calc(self, data, x, u=None):
        if not hasattr(data, 'pin_data'):
            data.pin_data = self.state.pinocchio.createData()
        q = x[:self.state.nq]
        pin.forwardKinematics(self.state.pinocchio, data.pin_data, q)
        pin.updateFramePlacements(self.state.pinocchio, data.pin_data)
        pos = data.pin_data.oMf[self.frame_id].translation
        data.r[...] = self.axis @ (pos - self.p_ref)

    def calcDiff(self, data, x, u=None):
        if not hasattr(data, 'pin_data'):
            data.pin_data = self.state.pinocchio.createData()
        q = x[:self.state.nq]
        pin.forwardKinematics(self.state.pinocchio, data.pin_data, q)
        pin.updateFramePlacements(self.state.pinocchio, data.pin_data)
        J = pin.computeFrameJacobian(
            self.state.pinocchio, data.pin_data, q,
            self.frame_id, pin.LOCAL_WORLD_ALIGNED)
        g = self.axis @ J[:3, :]
        data.Rx[...] = 0.0
        if data.Rx.ndim == 2:
            data.Rx[0, :self.state.nq] = g
        else:
            data.Rx[:self.state.nq] = g

class ActivationModelLinear(crocoddyl.ActivationModelAbstract):
    """Linear activation: a(r) = sum(r). Matches MPPI's per-feature cost
    `w · phi` (linear in feature). Used to make the OCP's `progress_vel`
    cost equivalent to MPPI's instead of Crocoddyl's default `0.5 · r²`.

    Important: the cost Hessian contribution from this activation is 0,
    so the OCP's overall Hessian must be positive-definite from OTHER
    costs (Tau, JA, JV, …). With small init weights this works fine.
    """
    def __init__(self, nr):
        crocoddyl.ActivationModelAbstract.__init__(self, nr)

    def calc(self, data, r):
        data.a_value = float(np.sum(r))

    def calcDiff(self, data, r):
        data.Ar[:] = 1.0

    def createData(self):
        return crocoddyl.ActivationDataAbstract(self)

class ResidualRailVelocity(crocoddyl.ResidualModelAbstract):
    """Along-rail velocity tracking: r = target_vel − (v_frame · rail_unit).
    With Crocoddyl's default quadratic activation, cost = 0.5·w·(v_target − v_rail)²
    has its minimum at v_rail = target_vel. Setting target_vel > 0 makes the
    OCP reward forward motion (without using a non-PD linear activation).
    """
    def __init__(self, state, frame_id, rail_unit, nu, target_vel=0.5):
        crocoddyl.ResidualModelAbstract.__init__(self, state, 1, nu, True, True, False)
        self.frame_id  = frame_id
        self.rail_unit = np.array(rail_unit)
        self.target_vel = float(target_vel)

    def calc(self, data, x, u=None):
        if not hasattr(data, 'pin_data'):
            data.pin_data = self.state.pinocchio.createData()
        q = x[:self.state.nq]
        v = x[self.state.nq:]
        pin.forwardKinematics(self.state.pinocchio, data.pin_data, q, v)
        pin.updateFramePlacements(self.state.pinocchio, data.pin_data)
        frame_vel = pin.getFrameVelocity(
            self.state.pinocchio, data.pin_data,
            self.frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        v_rail = float(np.dot(frame_vel.linear, self.rail_unit))
        data.r[0] = self.target_vel - v_rail

    def calcDiff(self, data, x, u=None):
        if not hasattr(data, 'pin_data'):
            data.pin_data = self.state.pinocchio.createData()
        q  = x[:self.state.nq]
        v  = x[self.state.nq:]
        a0 = np.zeros(self.state.nv)
        pin.computeForwardKinematicsDerivatives(
            self.state.pinocchio, data.pin_data, q, v, a0
        )
        Rq, Rv = pin.getFrameVelocityDerivatives(
            self.state.pinocchio, data.pin_data,
            self.frame_id, pin.LOCAL_WORLD_ALIGNED
        )
        data.Rx.fill(0.0)
        data.Rx[:self.state.nq] = -self.rail_unit @ Rq[:3, :]
        data.Rx[self.state.nq:] = -self.rail_unit @ Rv[:3, :]
