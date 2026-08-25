"""HumanCrocoddyl — Crocoddyl CSQP solver for human-with-tool."""
from repo_paths import REPO
import sys
sys.path.insert(0, str(REPO / "friction_lib/build"))
# so crocoddyl's bindings MUST be imported first (else the base wrapper "has not been created
import crocoddyl  # noqa: F401
try:
    import friction_lib
except Exception:
    pass

from cost_features import ResidualModelJointAcceleration
from utils_model_residuals import *
from utils_slice_trajectories import remove_legs_from_visual_model
import mujoco.viewer
import mim_solvers
import imageio
import hppfcl
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf
from final_models.human_base import HumanBase

_ENG_GROUPS_9DOF = {
    "thoracic": [0],
    "clavicle": [1],
    "shoulder": [2, 3, 4],
    "elbow":    [5, 6],
    "wrist":    [7, 8],
}

_ENG_GROUPS_FOR_NU = {
    9:  _ENG_GROUPS_9DOF,
    10: _ENG_GROUPS_9DOF,
}

class HumanCrocoddyl(HumanBase):
    """Crocoddyl (CSQP) solver."""

    def _post_init(self, args):
        """Crocoddyl-specific __init__ tail."""
        self.update_models()
        self.contact_frame_id = self.pin_model.getFrameId(self.args['contact'])

        self.pin_data = self.pin_model.createData()

        self.w_run       = args['w_run']
        self.w_term      = args['w_term']
        self.dt          = args['dt']
        self.T           = args['T']
        self.solver_type = args['solver_type']
        self.contact_aware_cost = bool(args.get('contact_aware_cost', False))
        self.press_in_actuation = bool(args.get('press_in_actuation', False))
        self.press_normal_dual   = bool(args.get('press_normal_dual', True))
        self.press_friction_dual = bool(args.get('press_friction_dual', True))
        self._lagged_fn = None

        self.viz       = None
        self.R_surface = None
        self._R_rock_ref = None 
        self.target_force_profile = None

        self.stick_static = args.get('stick_static', True)
        self._stick_displacements = None
        if not self.stick_static:
            q_traj = args.get('q_traj', None)
            if q_traj is not None:
                self._stick_displacements = self._estimate_stick_displacement(q_traj)
            else:
               print("[WARN] stick_static=False but no q_traj — falling back to static stick.")
               
        self.tool_length = args.get('tool_length', None)
        _win = args.get('contact_window', None)
        _T_contact = args.get('T_contact', None)
        _contact_frac = args.get('contact_frac', None)

        def _to_step(v):
            v = float(v)
            return int(round(v * self.T)) if 0.0 <= v <= 1.0 else int(round(v))

        self.T_contact_start = 0
        if _win is not None:
            s, e = _win
            self.T_contact_start = int(np.clip(_to_step(s), 0, self.T - 1))
            self.T_contact = int(np.clip(_to_step(e), self.T_contact_start + 1, self.T))
        elif _T_contact is not None:
            self.T_contact = int(np.clip(int(_T_contact), 1, self.T))
        elif _contact_frac is not None:
            self.T_contact = int(np.clip(round(float(_contact_frac) * self.T), 1, self.T))
        elif self.tool_length is not None:
            self.T_contact = self._estimate_T_contact()
        else:
            self.T_contact = self.T

        self.solver = self.create_solver()

        self._cache_solver_info()

        _jn = 'rock_fixed_joint'
        if self.pin_model.existJointName(_jn):
            jid = self.pin_model.getJointId(_jn)
            idx_q = self.pin_model.joints[jid].idx_q
            idx_v = self.pin_model.joints[jid].idx_v
            eps = 1e-4
            self.x_lb[idx_q] = -eps
            self.x_ub[idx_q] =  eps
            self.x_lb[self.nq + idx_v] = -eps
            self.x_ub[self.nq + idx_v] =  eps

        if self.args.get('lock_thorax', False) and self.pin_model.existJointName('middle_thoracic_X'):
            _iq = int(self.pin_model.joints[self.pin_model.getJointId('middle_thoracic_X')].idx_q)
            self._locked_q_idx = sorted(
                set(getattr(self, '_locked_q_idx', []) or []) | {_iq})
            print(f"[HumanCrocoddyl] thorax TRACKS demo path per-node (q idx {_iq}); "
                  f"q_norm excludes {self._locked_q_idx}")

    def _estimate_T_contact(self):
        """Estimate T_contact from q_traj kinematics. Projects the contact
        frame position onto the rail axis at each demo frame and finds the
        last frame where the projection is still within [0, tool_length].
        Falls back to full T if q_traj is unavailable.
        Mirrors human_ocp_mppi.HumanOCPMPPI._estimate_T_contact."""
        q_traj = self.args.get('q_traj', None)
        if self.args.get('own_task', False):
            return self.T
        if q_traj is None or self.tool_length is None:
            return self.T
        try:
            rail_vec = self.p_end_world - self.p_start_world
            rail_len = np.linalg.norm(rail_vec)
            if rail_len < 1e-9:
                return self.T
            rail_dir = rail_vec / rail_len
            tool_len = float(self.tool_length)

            temp_data = self.pin_model.createData()
            last_contact = 0
            for i, q in enumerate(q_traj):
                pin.framesForwardKinematics(self.pin_model, temp_data, q)
                pin.updateFramePlacements(self.pin_model, temp_data)
                p = temp_data.oMf[self.contact_frame_id].translation
                proj = float(np.dot(p - self.p_start_world, rail_dir))
                if 0.0 <= proj <= tool_len:
                    last_contact = i

            frac = last_contact / max(len(q_traj) - 1, 1)
            T_contact = int(round(frac * self.T))
            return max(1, min(T_contact, self.T))
        except Exception as e:
            return self.T

    def _estimate_stick_displacement(self, q_traj):
        """
        Compute per-timestep perpendicular displacement of the stick center
        relative to self.p_stick. The stick follows the contact point's drift
        normal to the stick axis (same logic as human_mppi._estimate_stick_trajectory).
        """
        z_stick = self.p_end_world - self.p_start_world
        z_stick = z_stick / np.linalg.norm(z_stick)

        temp_data = self.pin_model.createData()
        contact_positions = []
        for q in q_traj:
            pin.framesForwardKinematics(self.pin_model, temp_data, q)
            pin.updateFramePlacements(self.pin_model, temp_data)
            contact_positions.append(
                temp_data.oMf[self.contact_frame_id].translation.copy()
            )
        contact_positions = np.array(contact_positions)

        p_c0 = contact_positions[0]
        deltas = contact_positions - p_c0
        deltas_perp = deltas - (deltas @ z_stick)[:, None] * z_stick
        return deltas_perp

    def project_start_configuration(self):
        """
        Adjusts self.q0 using Inverse Kinematics so that the 
        contact frame exactly matches self.p_start_world.
        Enforces joint limits during the projection.
        """
        
        data = self.pin_model.createData()
        q = self.q0.copy()
        
        target_pos = self.p_start_world
        contact_frame_idx = self.pin_model.getFrameId(self.args['contact'])
        
        q_min = self.pin_model.lowerPositionLimit
        q_max = self.pin_model.upperPositionLimit

        for i in range(50):
            pin.framesForwardKinematics(self.pin_model, data, q)
            pin.updateFramePlacements(self.pin_model, data)
            
            curr_pos = data.oMf[contact_frame_idx].translation
            err = curr_pos - target_pos
            
            if np.linalg.norm(err) < 1e-5:
                break
            
            J = pin.computeFrameJacobian(
                self.pin_model, data, q, 
                contact_frame_idx, pin.LOCAL_WORLD_ALIGNED
            )
            J_trans = J[:3, :] 
            
            v = -J_trans.T @ np.linalg.inv(J_trans @ J_trans.T + 1e-6 * np.eye(3)) @ err
            
            q_next = pin.integrate(self.pin_model, q, v)
           
            q_next = np.minimum(np.maximum(q_next, q_min), q_max)
            
            q = q_next
            
        pin.framesForwardKinematics(self.pin_model, data, q)
        pin.updateFramePlacements(self.pin_model, data)
        final_err = np.linalg.norm(data.oMf[contact_frame_idx].translation - target_pos)
        
        if final_err > 1e-3:
            pass
        else:
            pass

        self.q0 = q
        
        self.p_start_world = data.oMf[contact_frame_idx].translation.copy()
        
        self.x0[:self.nq] = self.q0
        _qt0 = self.args.get('q_traj', None)
        if _qt0 is not None and len(_qt0) > 1 and self.nq == self.nv:
            self.x0[self.nq:self.nq + self.nv] = (
                np.asarray(_qt0[1], float) - np.asarray(_qt0[0], float)) / self.dt

    def _add_run_costs(self, costModel, state, actuation, nu, t):
        """Hook for subclasses to add extra running costs to each timestep."""
        pass

    def _compute_rock_ref_rotation(self):
        """FK-snapshot the rock frame's rotation at q0 → reference for rock_ori
        cost. Mirrors MPPI's `rock_quat_ref` (which is taken at the same q0)."""
        q0 = np.asarray(self.q0)
        pin.forwardKinematics(self.pin_model, self.pin_data, q0)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        return self.pin_data.oMf[self.tool_frame_id].rotation.copy()

    def _cache_solver_info(self):
        # keys_run must be the UNION across all running IAMs — free-phase
        base_keys = list(self.solver.problem.runningModels[0].differential.costs.costs.todict().keys())
        extra = []
        for m in self.solver.problem.runningModels[1:]:
            for k in m.differential.costs.costs.todict().keys():
                if k not in base_keys and k not in extra:
                    extra.append(k)
        self.keys_run = sorted(base_keys + extra)
        self.nr_run = len(self.keys_run)
        self.nr_term = len(self.solver.problem.terminalModel.differential.costs.costs.todict().keys())
        self.nr = self.nr_run + self.nr_term
        self.keys_term = list(self.solver.problem.terminalModel.differential.costs.costs.todict().keys())

    def update_models(self):
        if self.mj_model is not None and self.mj_data is not None:
            q_mj = self.q0.copy()
            if self.nq >= 7:
                q_mj[3:7] = [self.q0[6], self.q0[3], self.q0[4], self.q0[5]]
            self.mj_data.qpos = q_mj
            self.mj_data.qvel = self.v0
            mujoco.mj_step(self.mj_model, self.mj_data)
        pin.forwardKinematics(self.pin_model, self.pin_data, self.q0)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

    def _setup_solver_geometry(self):
        """Build state/actuation and compute rail geometry.
        Sets self.M_rail, self.R_surface, self.p_end_world (projected).
        Returns (state, actuation, nu).
        """
        state = crocoddyl.StateMultibody(self.pin_model)
        state.ub = self.x_ub
        state.lb = self.x_lb

        actuation = friction_lib.ActuationModelFriction(
            state, self.contact_frame_id, self.mu, self.target_force
        )
        nu = actuation.nu

        vec_along_stick = self.p_end_world - self.p_start_world
        vec_along_stick = vec_along_stick / np.linalg.norm(vec_along_stick)

        vec_start_to_end = self.p_end_world - self.p_start_world
        self.p_end_world = self.p_start_world + np.dot(vec_start_to_end, vec_along_stick) * vec_along_stick

        vec_diff   = self.p_start_world - self.p_stick
        vec_radial = vec_diff - np.dot(vec_diff, vec_along_stick) * vec_along_stick
        norm_r     = np.linalg.norm(vec_radial)
        vec_radial = vec_radial / norm_r if norm_r > 1e-6 else np.array([0., 1., 0.])

        valid_R = np.eye(3)
        valid_R[:, 0] = -vec_along_stick
        valid_R[:, 2] =  vec_radial
        valid_R[:, 1] =  np.cross(vec_radial, -vec_along_stick)

        self.M_rail    = pin.SE3(valid_R, self.p_start_world)
        self.R_surface = valid_R

        q_traj = self.args.get('q_traj', None)
        if q_traj is not None and not self.stick_static:
            self._R_surface_tv = self._compute_tv_contact_frames(q_traj)
        else:
            self._R_surface_tv = None

        self._contact_normal_ref = None
        if q_traj is not None and self.args.get('contact_normal_track', False):
            self._contact_normal_ref = self._compute_contact_normal_ref(q_traj)

        return state, actuation, nu

    def project_kinematics_to_surface(self, q_traj, dq_traj, thresh=1e-3, max_it=25):
        """Minimal-disturbance IK: for each frame where the recorded contact point
        PENETRATES the rigid surface (normal offset < -thresh), shift q as little as
        possible so the stick tip lies ON the surface (offset 0) and remove the normal
        component of velocity (so the contact does not break/explode in the rollout).
        Lifted frames (offset > 0, contact inactive) are left untouched. This makes the
        recorded kinematics consistent with the rigid contact WITHOUT a spring — the
        press force then comes from joint torque via transmission at the corrected
        position. Returns (q_new (T+1,nq), v_new (T+1,nv), report dict).

        Per-frame QP  min ||q_new - q_rec||^2  s.t.  FK_pos(q_new)·n = 0, and then
        v_new = v_rec - Jn^+ (Jn·v_rec)  (project out the normal velocity component).
        Solved by damped least squares (minimal-norm Jacobian steps)."""
        data = self.pin_model.createData()
        q_out, v_out = [], []
        n_proj = 0
        pen_before, pen_after = [], []
        for t in range(len(q_traj)):
            q = np.asarray(q_traj[t], float).copy()
            v = np.asarray(dq_traj[t], float).copy()
            n = (self._R_surface_tv[t][:, 2] if self._R_surface_tv is not None
                 and t < len(self._R_surface_tv) else self.R_surface[:, 2])
            pin.forwardKinematics(self.pin_model, data, q)
            pin.updateFramePlacements(self.pin_model, data)
            e0 = float((data.oMf[self.contact_frame_id].translation - self.p_start_world) @ n)
            pen_before.append(e0)
            active = e0 < -thresh
            if active:
                n_proj += 1
                for _ in range(max_it):
                    pin.forwardKinematics(self.pin_model, data, q)
                    pin.updateFramePlacements(self.pin_model, data)
                    e = float((data.oMf[self.contact_frame_id].translation
                               - self.p_start_world) @ n)
                    if abs(e) < 1e-5:
                        break
                    J = pin.computeFrameJacobian(self.pin_model, data, q,
                            self.contact_frame_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
                    Jn = n @ J
                    dq = -Jn * (e / (float(Jn @ Jn) + 1e-9))
                    q = pin.integrate(self.pin_model, q, dq)
                pin.forwardKinematics(self.pin_model, data, q)
                pin.updateFramePlacements(self.pin_model, data)
                J = pin.computeFrameJacobian(self.pin_model, data, q,
                        self.contact_frame_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
                Jn = n @ J
                v = v - Jn * (float(Jn @ v) / (float(Jn @ Jn) + 1e-9))
            pin.forwardKinematics(self.pin_model, data, q)
            pin.updateFramePlacements(self.pin_model, data)
            pen_after.append(float((data.oMf[self.contact_frame_id].translation
                                    - self.p_start_world) @ n))
            q_out.append(q); v_out.append(v)
        report = {"n_projected": n_proj, "T": len(q_traj),
                  "pen_before_mm": float(np.min(pen_before) * 1e3),
                  "pen_after_mm": float(np.min(pen_after) * 1e3),
                  "q_shift_deg": float(np.degrees(np.sqrt(np.mean(
                      (np.stack(q_out) - np.stack([np.asarray(q) for q in q_traj]))**2))))}
        return np.stack(q_out), np.stack(v_out), report

    def _compute_contact_normal_ref(self, q_traj):
        """Signed contact-point offset along the surface normal per node (negative =
        into the rock). Used as the ContactModel1D xref so the contact tracks the
        recorded penetration instead of rigidly resisting it. Uses the same per-node
        normal the contact axis uses (R_surface_tv[t] if moving, else R_surface)."""
        temp = self.pin_model.createData()
        ref = []
        for t, q in enumerate(q_traj):
            pin.framesForwardKinematics(self.pin_model, temp, q)
            pin.updateFramePlacements(self.pin_model, temp)
            cp = temp.oMf[self.contact_frame_id].translation
            n = (self._R_surface_tv[t][:, 2] if self._R_surface_tv is not None
                 and t < len(self._R_surface_tv) else self.R_surface[:, 2])
            ref.append(float((cp - self.p_start_world) @ n))
        return np.asarray(ref)

    def _compute_tv_contact_frames(self, q_traj):
        if self.args.get('own_task', False):
            return None
        """Per-timestep contact frame R_surface[t] for the moving stick.

        The 1D rail contact constrains the press NORMAL (the frame's Z-axis).
        Default (good) behaviour: the normal is HELD at the fixed rock-surface
        normal self.R_surface[:,2] — a real stick presses the rock in a roughly
        constant direction, so the press normal must NOT rotate with the stroke.
        Only the in-plane sliding axis (X) follows the trajectory tangent.

        The OLD behaviour (args['tv_normal_follows_tangent']=True) built the
        normal perpendicular to the trajectory tangent + radial from a fixed
        stick centre; over a curved down-stroke that swung the normal ~90 deg
        (Z: +0.40 -> -0.97), rotating the contact-constraint axis and spiking
        the 1D contact force at the crossover node. Kept only for reference."""
        follow_tangent = bool(self.args.get('tv_normal_follows_tangent', False))
        temp_data = self.pin_model.createData()
        contact_positions = []
        for q in q_traj:
            pin.framesForwardKinematics(self.pin_model, temp_data, q)
            pin.updateFramePlacements(self.pin_model, temp_data)
            contact_positions.append(
                temp_data.oMf[self.contact_frame_id].translation.copy()
            )
        contact_positions = np.array(contact_positions)

        tangents = np.gradient(contact_positions, axis=0)
        z_fixed = self.R_surface[:, 2]

        R_tv = []
        for t in range(len(q_traj)):
            tang = tangents[t]
            tang_norm = np.linalg.norm(tang)
            if tang_norm < 1e-9:
                R_tv.append(self.R_surface)
                continue
            x_tan = -tang / tang_norm

            if follow_tangent:
                x_axis = x_tan
                vec_diff = contact_positions[t] - self.p_stick
                vec_radial = vec_diff - np.dot(vec_diff, x_axis) * (-x_axis)
                radial_norm = np.linalg.norm(vec_radial)
                z_axis = self.R_surface[:, 2] if radial_norm < 1e-9 else vec_radial / radial_norm
            else:
                z_axis = z_fixed
                x_axis = x_tan - np.dot(x_tan, z_axis) * z_axis
                xn = np.linalg.norm(x_axis)
                if xn < 1e-6:
                    R_tv.append(self.R_surface)
                    continue
                x_axis /= xn

            y_axis = np.cross(z_axis, x_axis)
            y_norm = np.linalg.norm(y_axis)
            if y_norm < 1e-9:
                R_tv.append(self.R_surface)
                continue
            y_axis /= y_norm

            R = np.eye(3)
            R[:, 0] = x_axis
            R[:, 1] = y_axis
            R[:, 2] = z_axis
            R_tv.append(R)

        return R_tv

    def _thorax_track_bounds(self, t):
        """Per-node state bounds that make the thoracic (trunk) DOF TRACK the
        demo trajectory q_traj[t] +/- tol, leaving every other DOF unbounded.
        Returns (lb, ub) for a ConstraintModelResidual, or None if lock_thorax
        is off/unavailable. The trunk is thus an imposed postural INPUT that
        follows the human's real ~14deg trunk motion -- it neither sweeps (free)
        nor forces the shoulder to fake that motion (static hold)."""
        if not self.args.get('lock_thorax', False):
            return None
        if not self.pin_model.existJointName('middle_thoracic_X'):
            return None
        q_traj = self.args.get('q_traj', None)
        if q_traj is None:
            return None
        jid = self.pin_model.getJointId('middle_thoracic_X')
        iq = int(self.pin_model.joints[jid].idx_q)
        _ti = 0 if self.args.get('thorax_static', False) else min(int(t), len(q_traj) - 1)
        qt = np.asarray(q_traj[_ti], dtype=float)
        tol = float(self.args.get('lock_thorax_tol', 1e-3))
        BIG = 1e6
        lb = np.full(self.nq + self.nv, -BIG)
        ub = np.full(self.nq + self.nv,  BIG)
        lb[iq] = float(qt[iq]) - tol
        ub[iq] = float(qt[iq]) + tol
        self._locked_q_idx = sorted(
            set(getattr(self, '_locked_q_idx', []) or []) | {iq})
        return lb, ub

    def _add_thorax_track(self, constraintModel, state, nu, t):
        """Attach the per-node thorax-tracking constraint if lock_thorax is on."""
        _tb = self._thorax_track_bounds(t)
        if _tb is not None:
            _tres = crocoddyl.ResidualModelState(state, state.zero(), nu)
            constraintModel.addConstraint(
                "thorax_track",
                crocoddyl.ConstraintModelResidual(state, _tres, _tb[0], _tb[1]))

    def _build_contact_iam(self, state, actuation, nu, t):
        """Build one contact-phase IntegratedActionModel for timestep t."""
        R_t = self._R_surface_tv[t] if self._R_surface_tv is not None and t < len(self._R_surface_tv) else self.R_surface

        contactModel = crocoddyl.ContactModelMultiple(state, nu)
        _gains = np.asarray(self.args.get('contact_gains', [1, 1]), dtype=float)
        _cns = float(self.args.get('contact_normal_slack', 0.0))
        if _cns > 0.0:
            _gains = _gains * max(1.0 - _cns, 1e-3)
        _xref = (float(self._contact_normal_ref[min(t, len(self._contact_normal_ref) - 1)])
                 if getattr(self, '_contact_normal_ref', None) is not None else 0.0)
        contact_1d   = crocoddyl.ContactModel1D(
            state, self.contact_frame_id, _xref,
            pin.LOCAL, R_t, nu, _gains
        )
        contactModel.addContact("rail_sliding", contact_1d)

        costModel = crocoddyl.CostModelSum(state, nu)

        _two_cost = bool(self.args.get('force_two_cost', False))
        if 'press_force' in self.w_run:
            if _two_cost:
                fn_t = 0.0
            else:
                profile = getattr(self, 'target_force_profile', None)
                fn_t = (float(profile[t]) if profile is not None and t < len(profile)
                        else float(self.target_force))
            ref_force  = pin.Force(np.array([0.0, 0.0, fn_t]), np.zeros(3))
            force_res  = crocoddyl.ResidualModelContactForce(state, self.contact_frame_id, ref_force, 1, nu)
            force_cost = crocoddyl.CostModelResidual(state, force_res)
            costModel.addCost("press_force", force_cost, self.w_run.get('press_force', 1.0))
        if 'press_capacity' in self.w_run:
            _fmax    = float(self.args.get('force_max', 80.0))
            ref_cap  = pin.Force(np.array([0.0, 0.0, _fmax]), np.zeros(3))
            cap_res  = crocoddyl.ResidualModelContactForce(state, self.contact_frame_id, ref_cap, 1, nu)
            cap_cost = crocoddyl.CostModelResidual(state, cap_res)
            costModel.addCost("press_capacity", cap_cost, self.w_run.get('press_capacity', 1.0))

        if 'press_peak' in self.w_run:
            _prof = getattr(self, 'target_force_profile', None)
            if _prof is not None and np.asarray(_prof).size:
                _p = np.asarray(_prof, float).ravel()
                _pk_node = int(np.argmax(_p)); _f_pk = float(_p[_pk_node])
            else:
                _pk_node, _f_pk = -1, float(self.target_force)
            if t == _pk_node:
                ref_pk = pin.Force(np.array([0.0, 0.0, _f_pk]), np.zeros(3))
                pk_res = crocoddyl.ResidualModelContactForce(state, self.contact_frame_id, ref_pk, 1, nu)
                costModel.addCost("press_peak", crocoddyl.CostModelResidual(state, pk_res),
                                  self.w_run.get('press_peak', 1.0))

        if 'progress_vel' in self.w_run:
            rail_vec  = self.p_end_world - self.p_start_world
            rail_unit = rail_vec / np.linalg.norm(rail_vec)
            target_mode = bool(self.args.get('progress_vel_target_mode', False))
            if target_mode:
                target_vel = float(self.args.get('target_rail_vel', 0.5))
                if target_vel < 0:
                    rail_dist  = float(np.linalg.norm(self.p_end_world - self.p_start_world))
                    target_vel = rail_dist / (self.T * self.dt)
                progress_res = ResidualRailVelocity(
                    state, self.contact_frame_id, rail_unit, nu,
                    target_vel=target_vel)
                progress_cost = crocoddyl.CostModelResidual(state, progress_res)
            else:
                progress_res = ResidualRailVelocity(
                    state, self.contact_frame_id, rail_unit, nu,
                    target_vel=0.0)
                progress_cost = crocoddyl.CostModelResidual(
                    state, ActivationModelLinear(1), progress_res)
            costModel.addCost("progress_vel", progress_cost,
                              self.w_run['progress_vel'])

        if 'rail_lat' in self.w_run:
            rail_vec  = self.p_end_world - self.p_start_world
            rail_unit = rail_vec / np.linalg.norm(rail_vec)
            rail_lat_res  = ResidualRailLateral(
                state, self.contact_frame_id, self.p_start_world, rail_unit, nu)
            rail_lat_cost = crocoddyl.CostModelResidual(state, rail_lat_res)
            costModel.addCost("rail_lat", rail_lat_cost,
                              self.w_run['rail_lat'])

        _rrw = float(self.args.get('rock_rate_w', 0.0))
        if _rrw > 0.0:
            _rate_res = crocoddyl.ResidualModelFrameVelocity(
                state, self.tool_frame_id, pin.Motion.Zero(), pin.LOCAL, nu)
            _act_ang = crocoddyl.ActivationModelWeightedQuad(
                np.array([0., 0., 0., 1., 1., 1.]))
            costModel.addCost("rock_rate",
                              crocoddyl.CostModelResidual(state, _act_ang, _rate_res), _rrw)

        if 'rock_ori' in self.w_run:
            if self._R_rock_ref is None:
                self._R_rock_ref = self._compute_rock_ref_rotation()
            rock_res  = crocoddyl.ResidualModelFrameRotation(
                state, self.tool_frame_id, self._R_rock_ref, nu)
            rock_cost = crocoddyl.CostModelResidual(state, rock_res)
            costModel.addCost("rock_ori", rock_cost,
                              self.w_run['rock_ori'])

        if self.contact_aware_cost:
            u_res = ResidualModelActuationTau(state, actuation)
        else:
            u_res = crocoddyl.ResidualModelControl(state, nu)
        u_cost = crocoddyl.CostModelResidual(state, u_res)
        if 'Tau' in self.w_run:
            costModel.addCost("Tau", u_cost, self.w_run['Tau'])

        diag_weights_jv = np.concatenate([np.zeros(self.nv), np.ones(self.nv)])
        act_jv  = crocoddyl.ActivationModelWeightedQuad(diag_weights_jv)
        jv_res  = crocoddyl.ResidualModelState(state, state.zero(), nu)
        jv_cost = crocoddyl.CostModelResidual(state, act_jv, jv_res)
        if 'JV' in self.w_run:
            costModel.addCost("JV", jv_cost, self.w_run['JV'])

        x_reg_res  = crocoddyl.ResidualModelState(state, self.x0, nu)
        x_reg_cost = crocoddyl.CostModelResidual(state, x_reg_res)
        if 'xReg' in self.w_run:
            costModel.addCost("xReg", x_reg_cost, self.w_run['xReg'])

        _eng_groups = _ENG_GROUPS_FOR_NU.get(actuation.nu, {})
        if _eng_groups:
            for grp_name, grp_idxs in _eng_groups.items():
                key = f"Eng_{grp_name}"
                if key in self.w_run:
                    eng_res = ResidualModelEnergy(state, actuation, joint_indices=grp_idxs,
                                                  contact_aware=self.contact_aware_cost)
                    eng_cost = crocoddyl.CostModelResidual(state, eng_res)
                    costModel.addCost(key, eng_cost, self.w_run[key])
        if 'Eng' in self.w_run:
            eng_res  = ResidualModelEnergy(state, actuation,
                                           contact_aware=self.contact_aware_cost)
            eng_cost = crocoddyl.CostModelResidual(state, eng_res)
            costModel.addCost("Eng", eng_cost, self.w_run['Eng'])

        _tau_groups = _ENG_GROUPS_FOR_NU.get(actuation.nu, {})
        for grp_name, grp_idxs in _tau_groups.items():
            key = f"Tau_{grp_name}"
            if key in self.w_run:
                tau_res = ResidualModelGroupTau(state, actuation, grp_idxs)
                costModel.addCost(key, crocoddyl.CostModelResidual(state, tau_res),
                                  self.w_run[key])

        _eff_groups = _ENG_GROUPS_FOR_NU.get(actuation.nu, {})
        for grp_name, grp_idxs in _eff_groups.items():
            key = f"Effort_{grp_name}"
            if key in self.w_run:
                eff_res = ResidualModelGroupEffort(state, actuation, grp_idxs)
                costModel.addCost(key, crocoddyl.CostModelResidual(state, eff_res),
                                  self.w_run[key])

        if 'Geo' in self.w_run:
            geo_res  = ResidualModelGeodesic(state, nu)
            geo_cost = crocoddyl.CostModelResidual(state, geo_res)
            costModel.addCost("Geo", geo_cost, self.w_run['Geo'])

        if 'JA' in self.w_run:
            ja_res  = crocoddyl.ResidualModelJointAcceleration(state)
            ja_cost = crocoddyl.CostModelResidual(state, ja_res)
            costModel.addCost("JA", ja_cost, self.w_run['JA'])

        if 'JTC' in self.w_run:
            if self.contact_aware_cost:
                jtc_res = ResidualModelTorqueChangeContactAware(
                    state, actuation, self.contact_frame_id, R_t, ref=pin.LOCAL)
            else:
                jtc_res = ResidualModelTorqueChange(state, actuation)
            jtc_cost = crocoddyl.CostModelResidual(state, jtc_res)
            costModel.addCost("JTC", jtc_cost, self.w_run['JTC'])

        self._add_run_costs(costModel, state, actuation, nu, t)

        constraintModel = crocoddyl.ConstraintModelManager(state, nu)
        x_res       = crocoddyl.ResidualModelState(state, state.zero(), nu)
        x_lim_const = crocoddyl.ConstraintModelResidual(state, x_res, self.x_lb, self.x_ub)
        constraintModel.addConstraint("limits", x_lim_const)
        self._add_thorax_track(constraintModel, state, nu, t)

        self._add_effort_constraint(constraintModel, state, nu)
        self._add_self_collision_constraint(constraintModel, state, nu)
        self._add_rock_ori_constraint(constraintModel, state, nu)

        if self.args.get('hard_rail', False):
            _rail_vec  = self.p_end_world - self.p_start_world
            _rail_unit = _rail_vec / np.linalg.norm(_rail_vec)
            _lat_res   = ResidualRailLateral(
                state, self.contact_frame_id, self.p_start_world, _rail_unit, nu)
            _tol = float(self.args.get('hard_rail_tol', 1e-3))
            _lat_const = crocoddyl.ConstraintModelResidual(
                state, _lat_res, -_tol * np.ones(2), _tol * np.ones(2))
            constraintModel.addConstraint("rail_lat_hard", _lat_const)

        _slack_frac = float(self.args.get('force_strict_slack_frac', 0.0))
        if _slack_frac > 0.0 and self.target_force > 0.0:
            profile = getattr(self, 'target_force_profile', None)
            fn_t = (float(profile[t]) if profile is not None and t < len(profile)
                    else float(self.target_force))
            _ref_force = pin.Force(np.array([0.0, 0.0, fn_t]), np.zeros(3))
            _force_res = crocoddyl.ResidualModelContactForce(
                state, self.contact_frame_id, _ref_force, 1, nu)
            _slack = _slack_frac * fn_t
            _lb = np.full(3, -_slack)
            _ub = np.full(3,  _slack)
            _force_const = crocoddyl.ConstraintModelResidual(
                state, _force_res, _lb, _ub)
            constraintModel.addConstraint("force_strict", _force_const)

        dam = crocoddyl.DifferentialActionModelContactFwdDynamics(
            state, actuation, contactModel, costModel, constraintModel, 1e-3, True
        )
        return crocoddyl.IntegratedActionModelEuler(dam, self.dt)

    def _build_free_iam(self, state, actuation, nu, t):
        """Build one free-phase IAM (rock off the stick). No contact
        constraint, no press_force, no progress — otherwise mirrors
        _build_contact_iam so IRL features remain comparable."""
        costModel = crocoddyl.CostModelSum(state, nu)

        if self.contact_aware_cost:
            u_res = ResidualModelActuationTau(state, actuation)
        else:
            u_res = crocoddyl.ResidualModelControl(state, nu)
        u_cost = crocoddyl.CostModelResidual(state, u_res)
        if 'Tau' in self.w_run:
            costModel.addCost("Tau", u_cost, self.w_run['Tau'])

        diag_weights_jv = np.concatenate([np.zeros(self.nv), np.ones(self.nv)])
        act_jv  = crocoddyl.ActivationModelWeightedQuad(diag_weights_jv)
        jv_res  = crocoddyl.ResidualModelState(state, state.zero(), nu)
        jv_cost = crocoddyl.CostModelResidual(state, act_jv, jv_res)
        if 'JV' in self.w_run:
            costModel.addCost("JV", jv_cost, self.w_run['JV'])

        x_reg_res  = crocoddyl.ResidualModelState(state, self.x0, nu)
        x_reg_cost = crocoddyl.CostModelResidual(state, x_reg_res)
        if 'xReg' in self.w_run:
            costModel.addCost("xReg", x_reg_cost, self.w_run['xReg'])

        _eng_groups = _ENG_GROUPS_FOR_NU.get(actuation.nu, {})
        if _eng_groups:
            for grp_name, grp_idxs in _eng_groups.items():
                key = f"Eng_{grp_name}"
                if key in self.w_run:
                    eng_res = ResidualModelEnergy(state, actuation, joint_indices=grp_idxs,
                                                  contact_aware=self.contact_aware_cost)
                    eng_cost = crocoddyl.CostModelResidual(state, eng_res)
                    costModel.addCost(key, eng_cost, self.w_run[key])
        if 'Eng' in self.w_run:
            eng_res  = ResidualModelEnergy(state, actuation,
                                           contact_aware=self.contact_aware_cost)
            eng_cost = crocoddyl.CostModelResidual(state, eng_res)
            costModel.addCost("Eng", eng_cost, self.w_run['Eng'])

        _tau_groups = _ENG_GROUPS_FOR_NU.get(actuation.nu, {})
        for grp_name, grp_idxs in _tau_groups.items():
            key = f"Tau_{grp_name}"
            if key in self.w_run:
                tau_res = ResidualModelGroupTau(state, actuation, grp_idxs)
                costModel.addCost(key, crocoddyl.CostModelResidual(state, tau_res),
                                  self.w_run[key])

        _eff_groups = _ENG_GROUPS_FOR_NU.get(actuation.nu, {})
        for grp_name, grp_idxs in _eff_groups.items():
            key = f"Effort_{grp_name}"
            if key in self.w_run:
                eff_res = ResidualModelGroupEffort(state, actuation, grp_idxs)
                costModel.addCost(key, crocoddyl.CostModelResidual(state, eff_res),
                                  self.w_run[key])

        if 'Geo' in self.w_run:
            geo_res  = ResidualModelGeodesic(state, nu)
            geo_cost = crocoddyl.CostModelResidual(state, geo_res)
            costModel.addCost("Geo", geo_cost, self.w_run['Geo'])

        if 'JA' in self.w_run:
            ja_res  = crocoddyl.ResidualModelJointAcceleration(state)
            ja_cost = crocoddyl.CostModelResidual(state, ja_res)
            costModel.addCost("JA", ja_cost, self.w_run['JA'])

        if 'JTC' in self.w_run:
            if self.contact_aware_cost:
                jtc_res = ResidualModelTorqueChangeContactAware(
                    state, actuation, None, None)
            else:
                jtc_res = ResidualModelTorqueChange(state, actuation)
            jtc_cost = crocoddyl.CostModelResidual(state, jtc_res)
            costModel.addCost("JTC", jtc_cost, self.w_run['JTC'])

        self._add_run_costs(costModel, state, actuation, nu, t)

        constraintModel = crocoddyl.ConstraintModelManager(state, nu)
        x_res       = crocoddyl.ResidualModelState(state, state.zero(), nu)
        x_lim_const = crocoddyl.ConstraintModelResidual(state, x_res, self.x_lb, self.x_ub)
        constraintModel.addConstraint("limits", x_lim_const)
        self._add_thorax_track(constraintModel, state, nu, t)

        self._add_effort_constraint(constraintModel, state, nu)
        self._add_self_collision_constraint(constraintModel, state, nu)
        self._add_rock_ori_constraint(constraintModel, state, nu)

        dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, costModel, constraintModel
        )
        return crocoddyl.IntegratedActionModelEuler(dam, self.dt)

    def _add_effort_constraint(self, constraintModel, state, nu):
        """Optional HARD actuation-effort constraint (gated by args['effort_limits']).

        Box-bounds each actuated joint's commanded torque ``u`` to the per-joint
        URDF effort ceiling carried by ``self.pin_model.effortLimit``. Because every
        species' reduced model carries its OWN (O'Neill-scaled) effort limits, turning
        this on makes actuation capacity a genuine per-morphology channel rather than
        only entering through the fixed-reward Tau cost. Auxiliary DOFs with no real
        muscle ceiling (effortLimit <= 0 or non-finite, e.g. the rock-on-rail joint)
        are left unconstrained. ``args['effort_limit_scale']`` (default 1.0) uniformly
        relaxes/tightens the box for sensitivity sweeps. Default OFF so existing
        soft-Tau-only runs (and the MPPI/CSQP parity path) are unchanged.
        """
        if not self.args.get('effort_limits', False):
            return
        eff = np.asarray(self.pin_model.effortLimit, dtype=float).copy()
        eff *= float(self.args.get('effort_limit_scale', 1.0))
        BIG = 1e6
        eff = np.where((eff <= 0.0) | ~np.isfinite(eff), BIG, eff)
        eff = np.minimum(eff, BIG)
        if eff.shape[0] != nu:
            print(f"[WARN] effort_limits: effortLimit dim {eff.shape[0]} != nu {nu}; "
                  "effort constraint skipped")
            return
        u_res   = crocoddyl.ResidualModelControl(state, nu)
        u_const = crocoddyl.ConstraintModelResidual(state, u_res, -eff, eff)
        constraintModel.addConstraint("effort", u_const)

    def _add_self_collision_constraint(self, constraintModel, state, nu):
        """Optional lateral self-collision avoidance (gated by args['self_collision']).

        Keep the right forearm on the lateral side of the trunk by a margin R, so a
        medially-rotated-elbow morphology (low humeral torsion, e.g. H. naledi) cannot
        route the limb through its own torso. The trunk reference point and the lateral
        axis (trunk->shoulder, horizontal) are taken at q0 -- the body base is fixed.
        Margin via args['self_collision_margin'] (default 0.10 m). Default OFF.
        """
        if not self.args.get('self_collision', False):
            return
        if not hasattr(self, '_selfcol'):
            self._selfcol = None
            m, d = self.pin_model, self.pin_model.createData()
            pin.forwardKinematics(m, d, np.asarray(self.q0, float))
            pin.updateFramePlacements(m, d)
            if (m.existFrame('right_lowerarm') and m.existJointName('middle_thoracic_X')
                    and m.existJointName('right_shoulder_Z')):
                p_ref = d.oMi[m.getJointId('middle_thoracic_X')].translation.copy()
                axis = d.oMi[m.getJointId('right_shoulder_Z')].translation - p_ref
                axis[2] = 0.0
                self._selfcol = (m.getFrameId('right_lowerarm'), p_ref, axis)
        if self._selfcol is None:
            return
        fid, p_ref, axis = self._selfcol
        R = float(self.args.get('self_collision_margin', 0.10))
        res = ResidualLateralClearance(state, fid, p_ref, axis, nu)
        const = crocoddyl.ConstraintModelResidual(state, res,
                                                  np.array([R]), np.array([1e6]))
        constraintModel.addConstraint("self_collision", const)

    def _add_rock_ori_constraint(self, constraintModel, state, nu):
        """Optional HARD bound on how far the rock may rotate (args['rock_ori_bound'], deg).

        rock_ori is a quadratic penalty against one q0 snapshot, so the solver trades
        attitude away node by node and the error accumulates: measured 30 deg of rotation
        for the small hominins where the demonstration turns the stone 11.9 deg, ending
        23-26 deg from where they started. A scrape cannot be claimed with the tool rolling
        that far, and no weight makes a soft penalty a guarantee -- the force and the rail
        are already hard bounds here, attitude was the one part of the tool state left soft.

        Bounds |log(R_ref^T R(t))| per axis, R_ref being the attitude at q0. Default OFF.
        """
        bound_deg = float(self.args.get('rock_ori_bound', 0.0))
        if bound_deg <= 0.0:
            return
        if self._R_rock_ref is None:
            self._R_rock_ref = self._compute_rock_ref_rotation()
        b = np.deg2rad(bound_deg)
        res = crocoddyl.ResidualModelFrameRotation(
            state, self.tool_frame_id, self._R_rock_ref, nu)
        const = crocoddyl.ConstraintModelResidual(
            state, res, -b * np.ones(3), b * np.ones(3))
        constraintModel.addConstraint("rock_ori_bound", const)

    def _build_terminal_model(self, state, actuation, nu):
        """Build the terminal IntegratedActionModel (free dynamics, zero dt)."""
        termCostModel       = crocoddyl.CostModelSum(state, nu)
        termConstraintModel = crocoddyl.ConstraintModelManager(state, nu)

        u_res = crocoddyl.ResidualModelControl(state, nu)
        Tau   = crocoddyl.CostModelResidual(state, u_res)

        diag_weights_jv = np.concatenate([np.zeros(self.nv), np.ones(self.nv)])
        act_jv = crocoddyl.ActivationModelWeightedQuad(diag_weights_jv)
        jv_res = crocoddyl.ResidualModelState(state, state.zero(), nu)
        JV     = crocoddyl.CostModelResidual(state, act_jv, jv_res)

        eng_res = ResidualModelEnergy(state, actuation)
        Eng     = crocoddyl.CostModelResidual(state, eng_res)

        geo_res = ResidualModelGeodesic(state, nu)
        Geo     = crocoddyl.CostModelResidual(state, geo_res)

        ja_res = crocoddyl.ResidualModelJointAcceleration(state)
        JA     = crocoddyl.CostModelResidual(state, ja_res)

        jtc_res = ResidualModelTorqueChange(state, actuation)
        JTC     = crocoddyl.CostModelResidual(state, jtc_res)

        for key, cost_obj in zip(['Tau', 'JV', 'JA', 'JTC', 'Eng', 'Geo'],
                                  [Tau,   JV,   JA,   JTC,   Eng,   Geo]):
            if key in self.w_term:
                termCostModel.addCost(key, cost_obj, self.w_term[key])

        term_pos_res  = crocoddyl.ResidualModelFrameTranslation(state, self.contact_frame_id, self.p_end_world)
        term_pos_cost = crocoddyl.CostModelResidual(state, term_pos_res)
        if 'term_pos' in self.w_term:
            termCostModel.addCost('term_pos', term_pos_cost, self.w_term['term_pos'])

        term_pos_slack = self.args.get('term_pos_slack', 0.02)
        if term_pos_slack is not None:
            slack_vec        = np.full(3, term_pos_slack)
            term_pos_const   = crocoddyl.ConstraintModelResidual(state, term_pos_res, -slack_vec, slack_vec)
            termConstraintModel.addConstraint("term_pos", term_pos_const)
        self._add_thorax_track(termConstraintModel, state, nu, self.T)

        term_dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, termCostModel, termConstraintModel)
        return crocoddyl.IntegratedActionModelEuler(term_dam, 0.0)

    def _make_solver(self, problem):
        """Instantiate the solver from args['solver_type']."""
        if self.solver_type == 'CSQP':
            solver = mim_solvers.SolverCSQP(problem)
            solver.termination_tolerance = self.args.get('termination_tolerance', 1e-5)
            solver.max_qp_iters          = self.args.get('max_qp_iters', 1000)
            solver.eps_abs               = self.args.get('eps_abs', 1e-10)
            solver.eps_rel               = self.args.get('eps_rel', 0.0)
            solver.use_filter_line_search = True
            solver.reset_rho = True
            solver.reset_y   = True
        elif self.solver_type == 'SQP':
            solver = mim_solvers.SolverSQP(problem)
        else:
            solver = crocoddyl.SolverDDP(problem)
        return solver

    def _compute_contact_mask(self):
        """Per-timestep contact schedule (bool array, length T).

        Default: the single coarse window [T_contact_start, T_contact) — contact
        for every step inside it. With args['contact_mask_from_force'] and a force
        profile set, contact is restricted WITHIN that window to the steps where
        the per-step target force actually exceeds a threshold (a fraction of the
        profile peak). This turns the lift-off tail and any mid-stroke release into
        FREE phases instead of phantom contact (the model would otherwise enforce
        a rail constraint + press-force target where the demo isn't pressing,
        poisoning the end windows of the recovered W(t))."""
        mask = np.zeros(self.T, dtype=bool)
        mask[self.T_contact_start:self.T_contact] = True
        profile = getattr(self, 'target_force_profile', None)
        if (self.args.get('contact_mask_from_force', False)
                and profile is not None and len(profile)):
            prof = np.asarray(profile, dtype=float)
            peak = float(np.max(prof)) if prof.size else 0.0
            thr  = max(float(self.args.get('contact_mask_thresh', 0.15)) * peak, 1e-9)
            fmask = np.array([(float(prof[t]) if t < len(prof) else float(prof[-1])) > thr
                              for t in range(self.T)], dtype=bool)
            mask = mask & fmask
            if not mask.any():                  # safety: never produce an all-free schedule
                mask[self.T_contact_start:self.T_contact] = True
        return mask

    def create_solver(self):
        state, default_actuation, nu = self._setup_solver_geometry()
        profile = getattr(self, 'target_force_profile', None)
        self._contact_mask = self._compute_contact_mask()
        if self.args.get('contact_mask_from_force', False):
            _nc = int(self._contact_mask.sum())
            print(f"[HumanCrocoddyl] force-thresholded contact mask: {_nc}/{self.T} "
                  f"steps in contact (window was "
                  f"[{self.T_contact_start},{self.T_contact}) = "
                  f"{self.T_contact - self.T_contact_start} steps)")

        running_models = []
        for t in range(self.T):
            if self._contact_mask[t]:
                if profile is None:
                    actuation_t = default_actuation
                else:
                    fn_t = float(profile[t]) if t < len(profile) else float(profile[-1])
                    if self.press_normal_dual:
                        # Corrected scheme: DO NOT inject the normal -> the contact
                        fn_fric = fn_t
                        if self.press_friction_dual and self._lagged_fn is not None \
                                and t < len(self._lagged_fn):
                            fn_fric = float(self._lagged_fn[t])
                        R_t = (self._R_surface_tv[t]
                               if self._R_surface_tv is not None and t < len(self._R_surface_tv)
                               else (self.R_surface if self.R_surface is not None else np.eye(3)))
                        actuation_t = friction_lib.ActuationModelFriction(
                            state, self.contact_frame_id, self.mu, fn_fric,
                            np.asarray(R_t, dtype=float))
                        actuation_t.inject_normal = False
                    elif self.press_in_actuation:
                        R_t = (self._R_surface_tv[t]
                               if self._R_surface_tv is not None and t < len(self._R_surface_tv)
                               else (self.R_surface if self.R_surface is not None else np.eye(3)))
                        actuation_t = friction_lib.ActuationModelFriction(
                            state, self.contact_frame_id, self.mu, fn_t,
                            np.asarray(R_t, dtype=float))
                    else:
                        actuation_t = friction_lib.ActuationModelFriction(
                            state, self.contact_frame_id, self.mu, fn_t)
                running_models.append(self._build_contact_iam(state, actuation_t, nu, t))
            else:
                running_models.append(self._build_free_iam(state, default_actuation, nu, t))

        terminal_model = self._build_terminal_model(state, default_actuation, nu)
        problem        = crocoddyl.ShootingProblem(self.x0, running_models, terminal_model)
        return self._make_solver(problem)

    def get_traj_features(self, xs, us):
        """IRL interface — mirrors model_mocap.Human.get_traj_features.

        Reads cost values straight from the Crocoddyl problem after a manual
        calc()/calcDiff() pass, instead of routing through the per-feature
        helpers HumanBase uses (those are HumanMPPI-flavored).

        JA is computed KINEMATICALLY from the trajectory's velocities
        (ddq_t = (v_{t+1} - v_t)/dt) rather than via Crocoddyl's
        ResidualModelJointAcceleration which routes through aba+actuation.
        Kinematic ddq is a property of the trajectory alone — same for demo
        and OCP, no friction/contact/dynamics asymmetry.
        """
        Phi          = np.zeros(self.nr)
        Phi_int      = np.zeros(self.nr)
        Phis_Cum     = []
        Phis_Cum_Int = []
        Phis         = []
        T_model = len(self.solver.problem.runningModels)
        T_input = len(us)
        T = min(T_model, T_input)

        xs_arr = np.asarray(xs)
        v_arr  = xs_arr[:, self.nq:self.nq + self.nv]
        ddq_kin = np.diff(v_arr, axis=0) / self.dt
        ja_idx_run  = (self.keys_run.index('JA')
                       if 'JA' in self.keys_run else -1)
        w_ja_run    = float(self.w_run.get('JA', 1.0))
        ja_idx_term = (self.keys_term.index('JA')
                       if 'JA' in self.keys_term else -1)
        w_ja_term   = float(self.w_term.get('JA', 1.0))

        tau_act = np.zeros((T, self.nv))
        for i in range(T):
            run_model = self.solver.problem.runningModels[i]
            run_data  = self.solver.problem.runningDatas[i]
            run_model.differential.actuation.calc(
                run_data.differential.multibody.actuation, xs[i], us[i])
            tau_act[i] = np.asarray(
                run_data.differential.multibody.actuation.tau).copy()

        if bool(self.args.get('press_in_effort', False)):
            _cf = self.get_contact_forces()
            _nvj = min(self.nu, self.nv)
            for i in range(T):
                _fn = float(np.linalg.norm(_cf[i])) if i < len(_cf) else 0.0
                if _fn <= 1e-9:
                    continue
                _q = np.asarray(xs[i][:self.nq], float)
                _Jloc = pin.computeFrameJacobian(
                    self.pin_model, self.pin_data, _q, self.contact_frame_id, pin.LOCAL)
                _Rt = (np.asarray(self._R_surface_tv[i], float)
                       if (self._R_surface_tv is not None and i < len(self._R_surface_tv))
                       else (np.asarray(self.R_surface, float)
                             if self.R_surface is not None else np.eye(3)))
                _Jc = (_Rt @ _Jloc[:3])[2]
                tau_act[i, :_nvj] += (_Jc[:_nvj] * _fn)
        dtau_act = (np.diff(tau_act, axis=0) / self.dt
                    if len(tau_act) > 1 else np.zeros((0, self.nv)))

        if self.contact_aware_cost and ('JTC' in self.keys_run or 'JTC' in self.keys_term):
            from utils_model_residuals import jtc_id_residual
            n_jtc = len(ddq_kin)
            jtc_vec = np.zeros((n_jtc, self.nv))
            for i in range(n_jtc):
                jtc_vec[i] = jtc_id_residual(
                    self.pin_model, self.pin_data,
                    np.asarray(xs[i][:self.nq]), v_arr[i], ddq_kin[i])
        else:
            jtc_vec = dtau_act

        w_tau_run  = float(self.w_run.get('Tau',  1.0)) if 'Tau' in self.keys_run else 0.0
        w_jtc_run  = float(self.w_run.get('JTC',  1.0)) if 'JTC' in self.keys_run else 0.0
        w_jtc_term = float(self.w_term.get('JTC', 1.0)) if 'JTC' in self.keys_term else 0.0
        _eng_groups_for_phi = _ENG_GROUPS_FOR_NU.get(self.nu, {})

        X = xs[-1]
        for j, k in enumerate(self.keys_term):
            if k == 'JA' and len(ddq_kin) > 0:
                ddq_T = ddq_kin[-1]
                cost_ja_term = 0.5 * float(np.dot(ddq_T, ddq_T))
                Phi[j + self.nr_run]     += cost_ja_term
                Phi_int[j + self.nr_run] += cost_ja_term
                continue
            if k == 'JTC' and len(jtc_vec) > 0:
                dtau_T = jtc_vec[-1]
                cost_jtc_term = 0.5 * float(np.dot(dtau_T, dtau_T))
                Phi[j + self.nr_run]     += cost_jtc_term
                Phi_int[j + self.nr_run] += cost_jtc_term
                continue
            cost_model = self.solver.problem.terminalModel.differential.costs.costs[k].cost
            cost_data  = self.solver.problem.terminalData.differential.costs.costs[k]
            cost_model.calc(cost_data, X)
            Phi[j + self.nr_run]     += cost_data.cost
            Phi_int[j + self.nr_run] += cost_data.cost
        Phis_Cum.append(Phi.copy())
        Phis_Cum_Int.append(Phi_int.copy())
        Phis.append(Phi.copy())

        for i in range(T - 1, -1, -1):
            Phi_temp     = np.zeros(self.nr)
            Phi_temp_int = np.zeros(self.nr)
            X = xs[i]; U = us[i]
            run_model = self.solver.problem.runningModels[i]
            run_data  = self.solver.problem.runningDatas[i]
            run_model.differential.actuation.calc(
                run_data.differential.multibody.actuation, X, U)
            run_model.differential.actuation.calcDiff(
                run_data.differential.multibody.actuation, X, U)
            run_model.differential.calc(run_data.differential, X, U)
            costs_dict = run_model.differential.costs.costs.todict()
            for j, k in enumerate(self.keys_run):
                if k not in costs_dict:
                    continue
                if k == 'JA' and i < len(ddq_kin):
                    cost_ja = 0.5 * float(np.dot(ddq_kin[i], ddq_kin[i]))
                    Phi_temp[j]     += cost_ja
                    Phi_temp_int[j] += cost_ja * self.dt
                    continue
                if k == 'Tau' and i < len(tau_act):
                    cost_tau = 0.5 * float(np.dot(tau_act[i], tau_act[i]))
                    Phi_temp[j]     += cost_tau
                    Phi_temp_int[j] += cost_tau * self.dt
                    continue
                if k == 'JTC' and i < len(jtc_vec):
                    cost_jtc = 0.5 * float(np.dot(jtc_vec[i], jtc_vec[i]))
                    Phi_temp[j]     += cost_jtc
                    Phi_temp_int[j] += cost_jtc * self.dt
                    continue
                if (k.startswith('Eng_') and _eng_groups_for_phi
                        and i < len(tau_act)):
                    grp_name = k[4:]
                    grp_idxs = _eng_groups_for_phi.get(grp_name)
                    if grp_idxs is not None:
                        r_eng     = v_arr[i, grp_idxs] * tau_act[i, grp_idxs]
                        cost_eng  = 0.5 * float(np.dot(r_eng, r_eng))
                        Phi_temp[j]     += cost_eng
                        Phi_temp_int[j] += cost_eng * self.dt
                        continue
                if (k.startswith('Tau_') and _eng_groups_for_phi
                        and i < len(tau_act)):
                    grp_name = k[4:]
                    grp_idxs = _eng_groups_for_phi.get(grp_name)
                    if grp_idxs is not None:
                        r_tau    = tau_act[i, grp_idxs]
                        cost_tau = 0.5 * float(np.dot(r_tau, r_tau))
                        Phi_temp[j]     += cost_tau
                        Phi_temp_int[j] += cost_tau * self.dt
                        continue
                cost_model = run_model.differential.costs.costs[k].cost
                cost_data  = run_data.differential.costs.costs[k]
                cost_model.calc(cost_data, X, U)
                cost_model.calcDiff(cost_data, X, U)
                _cv = cost_data.cost
                if k == 'press_peak' or (k in ('press_force', 'press_capacity')
                                         and self.args.get('force_two_cost', False)):
                    _fmax = float(self.args.get('force_max', 80.0))
                    _cv = _cv / (0.5 * _fmax * _fmax + 1e-12)
                Phi_temp[j]     += _cv
                Phi_temp_int[j] += _cv * self.dt
            Phis.append(Phi_temp)
            Phi     += Phi_temp
            Phi_int += Phi_temp_int
            Phis_Cum.append(Phi.copy())
            Phis_Cum_Int.append(Phi_int.copy())

        return Phi, Phis[::-1], Phis_Cum[::-1], Phis_Cum_Int[::-1]

    def get_new_traj_features(self):
        """Compute features for the just-solved OCP trajectory.

        Delegates to `get_traj_features(xs, us)` using `self.solver.xs/us` so
        the IRL signal on the rollout uses the SAME kinematic special-cases
        (ddq_kin for JA, dtau_kin for JTC, the contact-aware Tau path, the
        press_force override) as the demo. Reading dyn residuals directly
        from runningDatas would produce phi values up to 4 orders of
        magnitude larger than the demo (aba+actuation spike at contact
        transitions), giving the IRL an apples-vs-oranges gradient that
        never closes.
        """
        xs = np.asarray(self.solver.xs.tolist().copy())
        us = np.asarray(self.solver.us.tolist().copy())
        return self.get_traj_features(xs, us)

    def get_control(self, qs, qds, qdds):
        """RNEA torques routed through the OCP's actuation model — matches
        model_mocap.Human.get_control."""
        us = []
        for i, (q, qd, qdd) in enumerate(zip(qs, qds, qdds)):
            if i >= len(self.solver.problem.runningModels):
                break
            m = self.solver.problem.runningModels[i]
            d = self.solver.problem.runningDatas[i]
            dummy_d = d.differential.copy()
            dummy_m = m.differential.copy()
            pin.rnea(dummy_m.pinocchio, dummy_d.pinocchio, q, qd, qdd)
            us.append(dummy_d.multibody.actuation.tau)
        return np.stack(us)

    def update_solver_weights(self, w_run_new, w_term_new):
        """Push new run/term cost weights into every running IAM (Crocoddyl
        bakes weights into each IAM at construction; we mutate them here).

        Free-phase IAMs lack contact-only costs (press_force, progress); we
        skip those keys for those timesteps.
        """
        for i in range(self.T):
            costs_dict = self.solver.problem.runningModels[i].differential.costs.costs.todict()
            for key_ in self.keys_run:
                if key_ in w_run_new and key_ in costs_dict:
                    self.solver.problem.runningModels[i].differential.costs.costs[key_].weight \
                        = float(w_run_new[key_])
        for key_ in self.keys_term:
            if key_ in w_term_new:
                self.solver.problem.terminalModel.differential.costs.costs[key_].weight \
                    = float(w_term_new[key_])

    def update_solver_weights_tv(self, w_run_windows, w_term_windows):
        """Per-window run weights pushed into the corresponding IAMs.

        w_run_windows : (n_w, nr_run) — values ordered by self.keys_run.
        w_term_windows: (n_w, nr_term) or None — terminal uses last window.

        Free-phase IAMs lack contact-only costs (press_force, progress); we
        skip those keys for those timesteps.
        """
        n_w = len(w_run_windows)
        window_size = max(1, self.T // n_w)
        for i in range(self.T):
            k = min(i // window_size, n_w - 1)
            costs_dict = self.solver.problem.runningModels[i].differential.costs.costs.todict()
            for j, key_ in enumerate(self.keys_run):
                if key_ not in costs_dict:
                    continue
                self.solver.problem.runningModels[i].differential.costs.costs[key_].weight \
                    = float(w_run_windows[k, j])
        if w_term_windows is not None and np.size(w_term_windows) > 0:
            for j, key_ in enumerate(self.keys_term):
                self.solver.problem.terminalModel.differential.costs.costs[key_].weight \
                    = float(w_term_windows[-1, j])

    def set_force_target_profile(self, profile):
        """Set per-step contact force target.

        profile : (T,) array of f_n values, or None to revert to the scalar
                  self.target_force everywhere.

        Updates both layers and rebuilds the solver:
          - Layer 1: each press_force cost residual's reference becomes
                     pin.Force([0, 0, profile[t]], 0).
          - Layer 2: each running IAM gets its own ActuationModelFriction
                     constructed with f_n=profile[t], so the friction reaction
                     in fwd dynamics tracks per-step.

        Crocoddyl's bindings don't expose the actuation as writable on existing
        IAMs, so set_force_target_profile rebuilds via create_solver(). Any
        per-window weight overrides set via update_solver_weights_tv must be
        re-applied after this call.
        """
        self.target_force_profile = (
            None if profile is None
            else np.asarray(profile, dtype=np.float64)
        )
        if self.target_force_profile is not None and len(self.target_force_profile):
            self.target_force = float(np.mean(self.target_force_profile))
        self.solver = self.create_solver()
        self._cache_solver_info()

    def solve(self, xs_init=None, us_init=None, w_run_windows=None, w_term_windows=None,
              use_given_warmstart=False):
        if w_run_windows is not None:
            self.update_solver_weights_tv(w_run_windows, w_term_windows)

        is_static_guess = np.allclose(xs_init[0], xs_init[-1], atol=1e-3)
        if not use_given_warmstart and not is_static_guess:
            xs_warm, us_warm = self._ik_warmstart()
            xs_warm[0][self.nq:] = self.x0[self.nq:]
            xs_init, us_init = xs_warm, us_warm

        self.solver.problem.x0 = xs_init[0]
        _mx = int(self.args.get('max_iter', 1000))
        self.solver.solve(xs_init, us_init, _mx)

        # Option B (press_friction_dual): the tangential friction mu*f_n must use
        if getattr(self, 'press_friction_dual', False) and getattr(self, 'press_normal_dual', False):
            for _ in range(int(self.args.get('friction_dual_passes', 2))):
                dch = self._update_friction_from_duals()
                if dch < float(self.args.get('friction_dual_tol', 0.5)):
                    break
                self.solver.solve(self.solver.xs.tolist(), self.solver.us.tolist(), _mx)

        xs = np.stack(self.solver.xs.tolist().copy())
        us = np.stack(self.solver.us.tolist().copy())
        return xs, us

    def _update_friction_from_duals(self, damping=None):
        """Option B: set each contact node's friction actuation target_fn to the
        back-solved contact-dual normal, so the tangential friction mu*f_n and the
        free normal share ONE consistent f_n. Returns max |change| for the
        fixed-point early-exit."""
        if damping is None:
            damping = float(self.args.get('friction_dual_damping', 1.0))
        lag = np.zeros(self.T); max_change = 0.0
        cmask = getattr(self, '_contact_mask', None)
        for t in range(self.T):
            rd = self.solver.problem.runningDatas[t]
            try:
                f = rd.differential.multibody.contacts.contacts['rail_sliding'].f.linear
                _fv = np.asarray(f, dtype=float)
                fn = (abs(float(_fv[2])) if self.args.get('friction_dual_normal_only', False)
                      else float(np.linalg.norm(_fv)))
                _cap = float(self.args.get('friction_dual_cap', 0.0))
                if _cap > 0.0:
                    fn = min(fn, _cap)
            except (KeyError, AttributeError):
                fn = 0.0
            lag[t] = fn
            if cmask is not None and t < len(cmask) and not cmask[t]:
                continue
            act = getattr(self.solver.problem.runningModels[t].differential, 'actuation', None)
            try:
                old = float(act.target_fn)
                new = (1.0 - damping) * old + damping * fn
                act.target_fn = new
                max_change = max(max_change, abs(new - old))
            except (AttributeError, TypeError):
                pass
        self._lagged_fn = lag
        return max_change

    def get_real_tau_pinocchio(self, xs, us_ocp, contact_forces=None):
        """
        Convert OCP control inputs to real joint torques via Pinocchio.

        The OCP friction actuation model applies:
            tau_real = u_ocp + J_contact^T @ f_friction

        where f_friction = -mu * f_n * (v_contact / ||v_contact||)

        Parameters
        ----------
        xs       : (T+1, nq+nv) state trajectory
        us_ocp   : (T, nu) OCP control inputs (friction actuation model inputs)
        contact_forces : (T,) measured normal forces, or None (uses target_force)

        Returns
        -------
        us_real  : (T, nu) real joint torques including friction contribution
        """
        T = min(len(us_ocp), len(xs) - 1)
        us_real = np.zeros((T, us_ocp.shape[1]))
        eps = 1e-6

        for t in range(T):
            q = xs[t][:self.nq]
            v = xs[t][self.nq:]

            pin.forwardKinematics(self.pin_model, self.pin_data, q, v)
            pin.updateFramePlacements(self.pin_model, self.pin_data)

            v_frame = pin.getFrameVelocity(
                self.pin_model, self.pin_data,
                self.contact_frame_id, pin.LOCAL)
            vx = v_frame.linear[0]
            vy = v_frame.linear[1]
            v_norm = np.sqrt(vx**2 + vy**2 + eps**2)

            f_n = contact_forces[t] if (contact_forces is not None and t < len(contact_forces)) else self.target_force
            f_fric = np.zeros(6)
            f_fric[0] = -self.mu * f_n * (vx / v_norm)
            f_fric[1] = -self.mu * f_n * (vy / v_norm)

            J = pin.computeFrameJacobian(
                self.pin_model, self.pin_data, q,
                self.contact_frame_id, pin.LOCAL)

            us_real[t] = us_ocp[t] + J.T @ f_fric

        return us_real

    def get_contact_forces(self):
        forces = []
        for t, d in enumerate(self.solver.problem.runningDatas):
            try:
                contact_data = d.differential.multibody.contacts.contacts['rail_sliding']
                forces.append(contact_data.f.linear)
            except (KeyError, AttributeError):
                forces.append(np.zeros(3))
        return np.stack(forces)

    def update_stick_viz(self, t):
        """Move the stick in Meshcat to match stick_traj[t] (for animation)."""
        if (self.viz is None or self._stick_displacements is None
                or self.stick_static):
            return
        t_c = min(t, len(self._stick_displacements) - 1)
        displacement = self._stick_displacements[t_c]

        R_stick = self._stick_R
        p_new = self.p_stick + displacement

        M = np.eye(4)
        M[:3, :3] = R_stick
        M[:3, 3] = p_new
        self.viz.viewer["pinocchio/visuals/stick_visual"].set_transform(M)

    def check_constraints(self, xs, us):
        print("\n--- (Analysis of Force and Costs) ---")
        print(f"{'t':>3} | {'Force (Fx, Fy, Fz) [N]':>22} | {'Active Costs'}")
        print("-" * 100)
        
        for t in range(self.T):
            data_iam = self.solver.problem.runningDatas[t]
            dam_data = data_iam.differential
            
            force_display = "      NAN"
            if hasattr(dam_data.multibody, "contacts"):
                contacts = dam_data.multibody.contacts.contacts.todict()
                if 'rail_sliding' in contacts:
                    f = contacts['rail_sliding'].f.linear
                    if not np.any(np.isnan(f)):
                        force_display = f"[{f[0]:5.2f}, {f[1]:5.2f}, {f[2]:5.2f}]"
            
            costs_dict = dam_data.costs.costs.todict()
            cost_strings = []
            for cname, cdata in costs_dict.items():
                val = cdata.cost
                if val > 1e-6 or np.isnan(val):
                    status = f"{val:.4f}" if not np.isnan(val) else "⚠️ NAN"
                    cost_strings.append(f"{cname}: {status}")
            
            costs_display = " | ".join(cost_strings)
            
            print(f"{t:3d} | {force_display:22} | {costs_display}")

        print("-" * 100)
        term_data = self.solver.problem.terminalData.differential
        term_costs = term_data.costs.costs.todict()
        term_strings = [f"{k}: {v.cost:.4f}" for k, v in term_costs.items()]
        print(f"TERM| {'':22} | {' | '.join(term_strings)}")
        print("-" * 100)

    def check_line_feasibility(self, n_samples: int = 50, ik_tol: float = 1e-3, verbose: bool = False) -> dict:
        """
        Check whether the arm can track the contact frame along a straight line
        from ``p_start_world`` to ``p_end_world`` while staying within joint limits.

        For every sampled point the method runs a damped pseudo-inverse IK
        (warm-started from the previous solution) and then verifies:
          1. IK converged   – position error < ``ik_tol`` [m]
          2. No joint limit is reached (within 1 mrad slack)

        Parameters
        ----------
        n_samples : int
            Number of evenly-spaced points to probe along the line (default 50).
        ik_tol : float
            IK convergence threshold in metres (default 1e-3).
        verbose : bool
            Print a per-sample status line (default True).

        Returns
        -------
        dict
            'feasible'            – True iff every sample passes both checks
            'feasible_fraction'   – fraction of samples that are feasible
            'first_failure_alpha' – α ∈ [0,1] of the first failing sample, or None
            'first_failure_pos'   – world position of the first failure, or None
            'results'             – list of per-sample dicts:
                                    alpha, pos, ik_error, in_limits, feasible,
                                    joint_violations [(name, value, limit_str)]
        """
        data  = self.pin_model.createData()
        q     = self.q0.copy()

        q_min = self.pin_model.lowerPositionLimit
        q_max = self.pin_model.upperPositionLimit

        has_ff    = (self.pin_model.joints[1].shortname() == "JointModelFreeFlyer")
        start_idx = 7 if has_ff else 0

        frame_id = self.contact_frame_id
        alphas   = np.linspace(0.0, 1.0, n_samples)
        results  = []

        first_failure_alpha = None
        first_failure_pos   = None

        total_dist = np.linalg.norm(self.p_end_world - self.p_start_world)

        if verbose:
            print(f"\n{'─'*64}")
            print(f"  Line feasibility check  ({n_samples} samples)")
            print(f"  start : {np.round(self.p_start_world, 4)}")
            print(f"  end   : {np.round(self.p_end_world,   4)}")
            print(f"  length: {total_dist * 100:.1f} cm")
            print(f"{'─'*64}")

        for alpha in alphas:
            target = self.p_start_world + alpha * (self.p_end_world - self.p_start_world)

            q_test = q.copy()
            for _ in range(100):
                pin.framesForwardKinematics(self.pin_model, data, q_test)
                pin.updateFramePlacements(self.pin_model, data)
                err_vec = data.oMf[frame_id].translation - target
                if np.linalg.norm(err_vec) < ik_tol * 0.1:
                    break
                J   = pin.computeFrameJacobian(
                    self.pin_model, data, q_test, frame_id, pin.LOCAL_WORLD_ALIGNED
                )
                J_t = J[:3, :]
                dq  = -J_t.T @ np.linalg.solve(J_t @ J_t.T + 1e-6 * np.eye(3), err_vec)
                q_next = pin.integrate(self.pin_model, q_test, dq)
                q_next[start_idx:] = np.clip(
                    q_next[start_idx:], q_min[start_idx:], q_max[start_idx:]
                )
                q_test = q_next

            pin.framesForwardKinematics(self.pin_model, data, q_test)
            pin.updateFramePlacements(self.pin_model, data)
            ik_err = np.linalg.norm(data.oMf[frame_id].translation - target)

            margin     = 1e-3
            violations = []
            for jidx in range(start_idx, self.pin_model.nq):
                jname = f"q[{jidx}]"
                for ji in range(1, len(self.pin_model.joints)):
                    jt = self.pin_model.joints[ji]
                    if hasattr(jt, "idx_q") and jt.idx_q == jidx:
                        jname = self.pin_model.names[ji]
                        break
                lo, hi, val = q_min[jidx], q_max[jidx], q_test[jidx]
                if val <= lo + margin:
                    violations.append((jname, val, f">= {lo:.4f}"))
                elif val >= hi - margin:
                    violations.append((jname, val, f"<= {hi:.4f}"))

            in_limits = len(violations) == 0
            feasible  = (ik_err < ik_tol) and in_limits

            results.append({
                "alpha":            alpha,
                "pos":              target.copy(),
                "ik_error":         ik_err,
                "in_limits":        in_limits,
                "feasible":         feasible,
                "joint_violations": violations,
            })

            if not feasible and first_failure_alpha is None:
                first_failure_alpha = alpha
                first_failure_pos   = target.copy()

            q = q_test.copy()

            if verbose:
                tick = "✓" if feasible else "✗"
                note = ""
                if ik_err >= ik_tol:
                    note += f"  IK err={ik_err:.4f} m"
                if not in_limits:
                    note += f"  at limit: {[v[0] for v in violations]}"
                print(f"  α={alpha:.2f}  {tick}{note}")

        n_ok     = sum(r["feasible"] for r in results)
        fraction = n_ok / n_samples

        if verbose:
            print(f"{'─'*64}")
            print(f"  Result: {n_ok}/{n_samples} feasible  ({fraction * 100:.1f} %)")
            if first_failure_alpha is not None:
                d_fail = first_failure_alpha * total_dist * 100
                print(f"  First failure at α={first_failure_alpha:.2f}  "
                      f"({d_fail:.1f} cm along rail)  "
                      f"pos={np.round(first_failure_pos, 4)}")
            else:
                print("  Full path is feasible ✓")
            print(f"{'─'*64}\n")

        return {
            "feasible":            first_failure_alpha is None,
            "feasible_fraction":   fraction,
            "first_failure_alpha": first_failure_alpha,
            "first_failure_pos":   first_failure_pos,
            "results":             results,
        }

    def debug_contact_at(self, x):
        q = x[:self.nq]
        v = x[self.nq:]

        pin.forwardKinematics(self.pin_model, self.pin_data, q, v)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        M = self.pin_data.oMf[self.contact_frame_id]
        p = M.translation
        print("contact world position:", p)

        J6 = pin.computeFrameJacobian(
            self.pin_model,
            self.pin_data,
            q,
            self.contact_frame_id,
            pin.LOCAL
        )
        J_trans = J6[:3, :]

        J_contact = self.R_surface.T @ J_trans
        print("J_contact[2,:] norm (normal dir):", np.linalg.norm(J_contact[2, :]))

    def solve_tracking_only(self, n_iter=1000, verbose=True):
        """
        Minimal OCP: just track the straight line start→end with no contact,
        no force constraints. Useful to check if the kinematics are solvable.
        """
        import crocoddyl

        state     = crocoddyl.StateMultibody(self.pin_model)
        actuation = crocoddyl.ActuationModelFull(state)
        nu        = actuation.nu

        running_models = []

        for t in range(self.T):
            alpha          = t / float(self.T)
            current_target = (1 - alpha) * self.p_start_world + alpha * self.p_end_world

            costModel = crocoddyl.CostModelSum(state, nu)

            track_res  = crocoddyl.ResidualModelFrameTranslation(
                state, self.contact_frame_id, current_target)
            track_cost = crocoddyl.CostModelResidual(state, track_res)
            costModel.addCost("track", track_cost, 1e4)

            u_res  = crocoddyl.ResidualModelControl(state, nu)
            u_cost = crocoddyl.CostModelResidual(state, u_res)
            costModel.addCost("ctrl", u_cost, 1e-3)

            x_res = crocoddyl.ResidualModelState(state, state.zero(), nu)
            bounds = crocoddyl.ActivationBounds(self.x_lb, self.x_ub)

            activation = crocoddyl.ActivationModelQuadraticBarrier(bounds)

            x_limit_cost = crocoddyl.CostModelResidual(state, activation, x_res)
            costModel.addCost("state_limit", x_limit_cost, 1e5) 

            dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(
                state, actuation, costModel)
            iam = crocoddyl.IntegratedActionModelEuler(dam, self.dt)
            running_models.append(iam)

        termCost = crocoddyl.CostModelSum(state, nu)
        term_res  = crocoddyl.ResidualModelFrameTranslation(
            state, self.contact_frame_id, self.p_end_world)
        term_cost = crocoddyl.CostModelResidual(state, term_res)
        termCost.addCost("term_pos", term_cost, 1e6)

        term_dam = crocoddyl.DifferentialActionModelFreeFwdDynamics(
            state, actuation, termCost)
        terminal_model = crocoddyl.IntegratedActionModelEuler(term_dam, 0.0)

        problem = crocoddyl.ShootingProblem(self.x0, running_models, terminal_model)

        xs_warm = [self.x0.copy()] * (self.T + 1)
        us_warm = [np.zeros(nu)] * self.T

        for i, x in enumerate(xs_warm):
            if np.any(x < self.x_lb) or np.any(x > self.x_ub):
                print(f"Warmstart state {i} violates bounds!")
                violated_indices = np.where((x < self.x_lb) | (x > self.x_ub))[0]
                for idx in violated_indices:
                    print(f"  State index {idx}: value={x[idx]:.6f}, lb={self.x_lb[idx]:.6f}, ub={self.x_ub[idx]:.6f}")
        
        print("x0:", xs_warm[0])
        print("xT:", xs_warm[-1])
        print("x_lb:", self.x_lb)
        print("x_ub:", self.x_ub)
        print("x0 in bounds:", np.all(xs_warm[0] >= self.x_lb) and np.all(xs_warm[0] <= self.x_ub))
        print("xT in bounds:", np.all(xs_warm[-1] >= self.x_lb) and np.all(xs_warm[-1] <= self.x_ub))

        solver = mim_solvers.SolverCSQP(problem)
        solver.termination_tolerance = 1e-4
        solver.max_qp_iters = 1000
        solver.eps_abs = 1e-5
        solver.eps_rel = 0.0
        solver.use_filter_line_search = True 

        solver.problem.x0 = xs_warm[0]
        solver.solve(xs_warm, us_warm, n_iter)

        problem.calc(solver.xs, solver.us)

        if verbose:
            final_q  = np.array(solver.xs[-1])[:self.nq]
            pin.forwardKinematics(self.pin_model, self.pin_data, final_q)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            final_pos = self.pin_data.oMf[self.contact_frame_id].translation
            err = np.linalg.norm(final_pos - self.p_end_world)
            print(f"Terminal position error: {err*1000:.2f} mm")

        return np.array(solver.xs), np.array(solver.us)

    def _ik_warmstart(self):
        """IK warm-start sa proračunom brzina (v) i konzistentnim frejmovima."""
        q_curr = self.q0.copy()
        xs_warm = []
        us_warm = []
        
        qs = [q_curr.copy()]
        for t in range(self.T):
            alpha = (t + 1) / float(self.T)
            target_pos = (1 - alpha) * self.p_start_world + alpha * self.p_end_world

            for _ in range(40):
                pin.forwardKinematics(self.pin_model, self.pin_data, q_curr)
                pin.updateFramePlacements(self.pin_model, self.pin_data)
                
                curr_pos = self.pin_data.oMf[self.contact_frame_id].translation
                err = curr_pos - target_pos
                if np.linalg.norm(err) < 1e-5: break
                
                J = pin.computeFrameJacobian(self.pin_model, self.pin_data, q_curr, 
                                             self.contact_frame_id, pin.LOCAL_WORLD_ALIGNED)[:3, :]
                dq = -J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(3), err)
                q_curr = pin.integrate(self.pin_model, q_curr, dq)
            qs.append(q_curr.copy())

        _gate = bool(self.args.get('warmstart_gate_contact', False))
        def _normal_jac(_q, _t):
            _Jloc = pin.computeFrameJacobian(self.pin_model, self.pin_data, _q,
                                             self.contact_frame_id, pin.LOCAL)
            _Rt = (np.asarray(self._R_surface_tv[_t], float)
                   if getattr(self, '_R_surface_tv', None) is not None
                   and _t < len(self._R_surface_tv)
                   else np.asarray(self.R_surface if self.R_surface is not None
                                   else np.eye(3), float))
            return (_Rt @ _Jloc[:3])[2]
        for t in range(self.T + 1):
            if t < self.T:
                v_curr = pin.difference(self.pin_model, qs[t], qs[t+1]) / self.dt
            else:
                v_curr = np.zeros(self.nv)
            in_win = (t < self.T) and (self.T_contact_start <= t < self.T_contact)
            _Jn = None
            if _gate and in_win:
                _Jn = _normal_jac(qs[t], t)
                _den = float(_Jn @ _Jn) + 1e-9
                v_curr = v_curr - _Jn * (float(_Jn @ v_curr) / _den)
            xs_warm.append(np.concatenate([qs[t], v_curr]))
            if t < self.T:
                _wp = float(self.args.get('warmstart_press', 0.0) or 0.0)
                if _wp > 0.0 and (in_win if _gate else True):
                    if _Jn is None:
                        _Jn = _normal_jac(qs[t], t)
                    _nvj = min(self.nu, self.nv)
                    _u = np.zeros(self.nu); _u[:_nvj] = -_Jn[:_nvj] * _wp
                    us_warm.append(_u)
                else:
                    us_warm.append(np.zeros(self.nu))

        # init_xs[0], NOT problem.x0) must match the DEMO's v0 (self.x0[nq:]) —
        if len(xs_warm) > 0:
            xs_warm[0][self.nq:] = self.x0[self.nq:]
        return xs_warm, us_warm

    def _neutral_warmstart(self):
        """Genuinely UN-informed warm-start: hold the initial pose q0 static with
        zero velocity for the whole horizon, zero controls. No rail-IK path, no
        press seed -- the challenger must discover the whole reach+press from the
        cost alone (the honest 'no warm-start' IRL). Mirrors _ik_warmstart's return
        signature and the replay-side neutral seed (run_csqp_population_irl.py)."""
        x_static = np.concatenate([self.q0.copy(), np.zeros(self.nv)])
        xs_warm = [x_static.copy() for _ in range(self.T + 1)]
        us_warm = [np.zeros(self.nu) for _ in range(self.T)]
        return xs_warm, us_warm

    def get_contact_point(self, x):
        """
        Vraća svetske koordinate (x, y, z) kontaktne tačke za dato stanje x.
        """
        q = x[:self.nq]
        
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        
        contact_placement = self.pin_data.oMf[self.contact_frame_id]
        
        return contact_placement.translation.copy()
