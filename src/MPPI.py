"""CPU model-predictive path-integral controller.

Sampling-based trajectory generator used as the inner solver alongside the
analytical OCP: rolls out perturbed control sequences and reweights them by
exponentiated cost.
"""
import numpy as np
import mujoco

class MPPIController:
    def __init__(self, model, data, pin_model, pin_data, contact_frame_id,
                 horizon=20, num_samples=500, noise_sigma=0.5, lambda_=0.1,
                 target_force=10.0, p_start_world=None, p_end_world=None, mu=0.3,
                 locked_joint_constraints=None, q_traj=None, x0=None):
        """
        MPPI Controller for tool handling task
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            pin_model: Pinocchio model (for computing frame positions)
            pin_data: Pinocchio data
            contact_frame_id: Frame ID for contact point
            horizon: Planning horizon (H)
            num_samples: Number of rollout samples (K)
            noise_sigma: Exploration noise std dev
            lambda_: Temperature parameter (lower = more aggressive)
            target_force: Desired normal force
            p_end_world: Target end position
            mu: Friction coefficient
        """
        self.model = model
        self.data = data
        self.pin_model = pin_model
        self.pin_data = pin_data
        self.contact_frame_id = contact_frame_id
        
        self.H = horizon
        self.K = num_samples
        self.sigma = noise_sigma
        self.lam = lambda_
        self.noise_taper = 'uniform'
        self._taper = np.ones(self.H)
        self.noise_schedule = self._build_noise_schedule()
        
        self.target_force = target_force
        self.force_traj = None
        self.force_ramp_steps = 0
        self.target_rail_vel = 0.0
        self.p_start_world = p_start_world
        self.p_end_world = p_end_world
        self.q_traj = q_traj
        self.traveled_ref = None
        self.mu = mu
        
        self.nu = model.nu 
        self.nq = model.nq
        self.nv = model.nv

        self.dt = 0.0083
        
        self.U = np.zeros((self.H, self.nu))

        act_qposadr, act_dofadr = [], []
        for i in range(self.nu):
            jnt_id = int(self.model.actuator_trnid[i, 0])
            act_qposadr.append(int(self.model.jnt_qposadr[jnt_id]))
            act_dofadr.append(int(self.model.jnt_dofadr[jnt_id]))
        self.arm_qposadr = np.array(act_qposadr, dtype=int)
        self.arm_dofadr = np.array(act_dofadr, dtype=int)

        # Reference state must be set before warmstart()
        self.x0 = x0 if x0 is not None else np.zeros(model.nq + model.nv)

        self.U = self.warmstart()
        self.u_prev = self.U[0].copy()

        self._per_joint_sigma = None
        self.dial_iters = 1
        self.dial_noise_decay = 0.5

        self.w_run = {
            'Tau': 0.01,
            'JV':  0.01,
        }

        self.w_term = {}
        
        self.q_min = model.jnt_range[:, 0].copy()
        self.q_max = model.jnt_range[:, 1].copy()
        
        self.contact_site_name = "rock_contact_point"  
        self.contact_site_id = None
        try:
            self.contact_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, self.contact_site_name)
            print(f"[MPPI] Found contact site '{self.contact_site_name}' with ID {self.contact_site_id}")
        except:
            print(f"[WARN] Site {self.contact_site_name} not found, using body position")

        self.best_traj = None
        self.last_costs = None
        self.locked_joint_constraints = locked_joint_constraints or []

        self.all_trajs = []

        self.contact_elite = False
        self.contact_elite_min = 10
        self.contact_elite_jitter = 0.3

        self.d_sim_pool = [mujoco.MjData(self.model) for _ in range(self.K)]

        self._stick_radius = 0.02
        self._rock_radius  = 0.022
        self._stick_half_len = 0.4

        self.stick_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stick")
        if self.stick_body_id < 0:
            print("[WARN] 'stick' body not found — contact_attract cost disabled")
            self.stick_body_id = None

        self.rock_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rock")
        self.rock_quat_ref = None
        if self.rock_body_id >= 0:
            d_tmp = mujoco.MjData(model)
            d_tmp.qpos[:] = self.x0[:self.nq]
            d_tmp.qvel[:] = 0.0
            mujoco.mj_forward(model, d_tmp)
            self.rock_quat_ref = d_tmp.xquat[self.rock_body_id].copy()
            print(f"[MPPI] Rock orientation ref: {np.round(self.rock_quat_ref, 4)}")

        self.rock_geom_id = None
        rock_body_id = self.rock_body_id
        if rock_body_id >= 0:
            for g in range(model.body_geomadr[rock_body_id],
                           model.body_geomadr[rock_body_id] + model.body_geomnum[rock_body_id]):
                if model.geom_contype[g] > 0:
                    self.rock_geom_id = g
                    break
        if self.rock_geom_id is None:
            print("[WARN] Rock collision geom not found — contact_attract cost disabled")

        if q_traj is not None and p_start_world is not None and p_end_world is not None:
            self._build_traveled_ref()

        self._stick_traj_ctrl   = None
        self._stick_qposadr_ctrl = None

        self._w_run_windows  = None
        self._w_term_windows = None
        self._tv_window_size = 1

    def set_noise_from_limits(self, fraction=1.0, limits=None):
        """
        Set per-joint exploration noise proportional to torque limits.

            σ_j = fraction * limit_j

        Parameters
        ----------
        fraction : scale factor (e.g. 0.5 → half the torque limit)
        limits   : (nu,) per-joint torque limits; defaults to actuator_ctrlrange
        """
        if limits is None:
            limits = self.model.actuator_ctrlrange[:self.nu, 1].copy()
        self._per_joint_sigma = fraction * np.array(limits)
        print(f"[MPPI] per-joint noise: fraction={fraction}, "
              f"σ range=[{self._per_joint_sigma.min():.1f}, {self._per_joint_sigma.max():.1f}]")

    def set_dial(self, iters=3, noise_decay=0.5):
        """
        Enable DIAL-MPC style multi-iteration noise annealing.

        Each step() runs `iters` MPPI updates. Noise is scaled by
        noise_decay^i at iteration i (so it shrinks as we converge).

        Parameters
        ----------
        iters       : number of MPPI iterations per step
        noise_decay : noise multiplier between iterations (< 1 to shrink)
        """
        self.dial_iters = iters
        self.dial_noise_decay = noise_decay
        print(f"[MPPI] DIAL: {iters} iters/step, noise_decay={noise_decay}")

    def set_stick_traj(self, stick_traj, qposadr):
        """
        Register the pre-computed stick trajectory so that _rollout updates
        the stick body position at each internal timestep.

        stick_traj : (T+1, 3) array of stick center world positions
        qposadr    : int, MuJoCo qpos address of the stick freejoint
        """
        self._stick_traj_ctrl    = np.array(stick_traj)
        self._stick_qposadr_ctrl = int(qposadr)

    def set_tv_weights(self, w_run_windows, w_term_windows=None):
        """
        Register per-window weights used inside _rollout.

        w_run_windows  : list of w_run dicts (one per window), or None to clear
        w_term_windows : list of w_term dicts (one per window), or None
        """
        self._w_run_windows  = list(w_run_windows)  if w_run_windows  else None
        self._w_term_windows = list(w_term_windows) if w_term_windows else None
        if self._w_run_windows:
            self._tv_window_size = max(1, self.H // len(self._w_run_windows))

    def _w_run_at(self, t):
        """Return the w_run dict appropriate for rollout timestep t."""
        if not self._w_run_windows:
            return self.w_run
        k = min(t // self._tv_window_size, len(self._w_run_windows) - 1)
        return self._w_run_windows[k]

    def _w_term_final(self):
        """Return the w_term dict to use for the terminal cost."""
        if self._w_term_windows:
            return self._w_term_windows[-1]
        return self.w_term

    def _build_traveled_ref(self):
        """Precompute along-rail projection of contact site for each frame of q_traj."""
        if self.q_traj is None or self.p_start_world is None or self.p_end_world is None:
            return
        if self.contact_site_id is None:
            return
        rail_vec = self.p_end_world - self.p_start_world
        rail_len = np.linalg.norm(rail_vec)
        if rail_len < 1e-9:
            return
        rail_unit = rail_vec / rail_len
        d_tmp = mujoco.MjData(self.model)
        traveled = []
        for q in self.q_traj:
            mujoco.mj_resetData(self.model, d_tmp)
            self._set_state(d_tmp, q, np.zeros(self.nv))
            mujoco.mj_forward(self.model, d_tmp)
            pos = d_tmp.site_xpos[self.contact_site_id].copy()
            traveled.append(float(np.dot(pos - self.p_start_world, rail_unit)))
        self.traveled_ref = np.array(traveled)
        print(f"[MPPI] traveled_ref built: {len(self.traveled_ref)} frames, "
              f"range [{self.traveled_ref.min():.3f}, {self.traveled_ref.max():.3f}] m")

    def warmstart(self):
        U = np.zeros((self.H, self.nu))
        d_tmp = mujoco.MjData(self.model)
        d_tmp.qpos[:] = self.x0[:self.nq]
        d_tmp.qvel[:] = 0.0
        mujoco.mj_forward(self.model, d_tmp)
        tau_grav = d_tmp.qfrc_bias[self.arm_dofadr]
        for t in range(self.H):
            U[t] = self._clip_u(tau_grav)
        return U

    def warmstart_from_demo(self, q_demo, locked_joint_constraints=None):
        """
        Inverse-dynamics warmstart along the demo trajectory.
        Finite-differences q_demo to get dq, ddq, then uses mj_inverse
        to compute the torques that reproduce that motion (no contact).

        q_demo: (N, nq_arm) joint angles from demo.
        Returns U: (H, nu) clipped.
        """
        H = min(self.H, len(q_demo))
        dt = self.dt

        dq = np.zeros_like(q_demo)
        ddq = np.zeros_like(q_demo)
        dq[1:] = (q_demo[1:] - q_demo[:-1]) / dt
        ddq[1:-1] = (dq[2:] - dq[:-2]) / (2 * dt) if len(dq) > 2 else 0.0

        d_tmp = mujoco.MjData(self.model)
        U = np.zeros((self.H, self.nu))

        for t in range(H):
            mujoco.mj_resetData(self.model, d_tmp)
            for i, (qa, da) in enumerate(zip(self.arm_qposadr, self.arm_dofadr)):
                d_tmp.qpos[qa] = q_demo[t, i]
                d_tmp.qvel[da] = dq[t, i]
                d_tmp.qacc[da] = ddq[t, i]
            if locked_joint_constraints is not None:
                for qpos_id, dof_id, lock_val in locked_joint_constraints:
                    d_tmp.qpos[qpos_id] = lock_val
                    d_tmp.qvel[dof_id] = 0.0
                    d_tmp.qacc[dof_id] = 0.0
            mujoco.mj_inverse(self.model, d_tmp)
            U[t] = self._clip_u(d_tmp.qfrc_inverse[self.arm_dofadr])

        for t in range(H, self.H):
            U[t] = U[H - 1]

        print(f"[MPPI] warmstart_from_demo (inverse dynamics): H={H}, "
              f"tau range=[{U[:H].min():.1f}, {U[:H].max():.1f}] Nm")
        return U

    def _clip_u(self, u):
        if hasattr(self.model, "actuator_ctrllimited") and hasattr(self.model, "actuator_ctrlrange"):
            limited = self.model.actuator_ctrllimited[:self.nu].astype(bool)
            umin = np.full(self.nu, -200.0)
            umax = np.full(self.nu,  200.0)
            if np.any(limited):
                umin[limited] = self.model.actuator_ctrlrange[:self.nu, 0][limited]
                umax[limited] = self.model.actuator_ctrlrange[:self.nu, 1][limited]
            return np.clip(u, umin, umax)
        return np.clip(u, -200.0, 200.0)

    def _set_state(self, d_sim, state_qpos, state_qvel):
        if len(state_qpos) == self.model.nq:
            d_sim.qpos[:] = state_qpos
            if len(state_qvel) == self.model.nv:
                d_sim.qvel[:] = state_qvel
            else:
                d_sim.qvel[:] = 0.0
            return

        d_sim.qpos[:] = self.data.qpos
        d_sim.qvel[:] = 0.0
        n = min(len(state_qpos), len(self.arm_qposadr))
        d_sim.qpos[self.arm_qposadr[:n]] = state_qpos[:n]
        m = min(len(state_qvel), len(self.arm_dofadr))
        d_sim.qvel[self.arm_dofadr[:m]] = state_qvel[:m]

    def _get_contact_position(self, mj_data):
        if self.contact_site_id is not None:
            return mj_data.site_xpos[self.contact_site_id].copy()
        else:
            raise RuntimeError("rock_contact_point site not found in model — cannot get contact position")
        
    def _get_contact_force(self, mj_data):
        """
        Compute normal force at rock <-> stick contact using mj_contactforce.
        Returns scalar normal force magnitude in Newtons.
        """
        total_normal_force = 0.0
        
        for i in range(mj_data.ncon):
            c = mj_data.contact[i]
            
            b1 = self.model.geom_bodyid[c.geom1]
            b2 = self.model.geom_bodyid[c.geom2]
            
            b1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b1)
            b2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b2)
            
            pair = {b1_name, b2_name}
            if pair == {"rock", "stick"}:
                force_contact = np.zeros(6)
                mujoco.mj_contactForce(self.model, mj_data, i, force_contact)
                
                normal_force = force_contact[0]
                total_normal_force += max(0.0, normal_force)
        
        if not hasattr(self, '_force_debug_count'):
            self._force_debug_count = 0
        if self._force_debug_count < 12 and mj_data.ncon > 0:
            rs_contacts = []
            for i in range(mj_data.ncon):
                c = mj_data.contact[i]
                b1 = self.model.geom_bodyid[c.geom1]
                b2 = self.model.geom_bodyid[c.geom2]
                n1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b1)
                n2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, b2)
                if {n1, n2} == {"rock", "stick"}:
                    fb = np.zeros(6)
                    mujoco.mj_contactForce(self.model, mj_data, i, fb)
                    rs_contacts.append({
                        'slot': i, 'dist': c.dist, 'normal_N': fb[0],
                        'efc_addr': int(c.efc_address),
                        'efc_force_at_addr': float(mj_data.efc_force[c.efc_address])
                                            if c.efc_address >= 0 else None,
                    })
            if rs_contacts:   
                sum_compr = sum(max(0.0, fc['normal_N']) for fc in rs_contacts)
                max_abs   = max(abs(fc['normal_N']) for fc in rs_contacts)
                slot0_is_rs = (rs_contacts[0]['slot'] == 0)
                slot0_efc = rs_contacts[0]['efc_force_at_addr'] if slot0_is_rs else \
                            (float(mj_data.efc_force[mj_data.contact[0].efc_address])
                            if mj_data.contact[0].efc_address >= 0 else None)
                print(f"[FORCE-DIAG #{self._force_debug_count}] "
                    f"ncon={mj_data.ncon}  rock_stick={len(rs_contacts)}  "
                    f"slot0_is_rs={slot0_is_rs}  "
                    f"per-rs-contact={[(c['slot'], round(c['dist'],5), round(c['normal_N'],2), c['efc_force_at_addr']) for c in rs_contacts]}  "
                    f"|  demo-SUM={sum_compr:.2f}N  CPU-MAX={max_abs:.2f}N  "
                    f"MJX-slot0-efc={slot0_efc}")
                self._force_debug_count += 1

        return total_normal_force

    def _contact_rail_dist(self, mj_data):
        """
        Lateral distance of the contact point from the rail line p_start→p_end.
        Returns the perpendicular distance (in meters) — zero means perfectly on rail.
        """
        if self.p_start_world is None or self.p_end_world is None:
            return 0.0
        p_contact = self._get_contact_position(mj_data)
        rail_vec = self.p_end_world - self.p_start_world
        rail_len = np.linalg.norm(rail_vec)
        if rail_len < 1e-9:
            return 0.0
        rail_dir = rail_vec / rail_len
        diff = p_contact - self.p_start_world
        along = np.dot(diff, rail_dir) * rail_dir
        perp = diff - along
        return np.linalg.norm(perp)

    def _rock_stick_gap(self, mj_data):
        """
        Signed gap between rock surface and stick surface (negative = penetrating).
        Uses live geom/body positions from mj_data.
        """
        if self.rock_geom_id is None or self.stick_body_id is None:
            return 0.0
        rock_center  = mj_data.geom_xpos[self.rock_geom_id].copy()
        stick_center = mj_data.xpos[self.stick_body_id].copy()
        stick_axis   = mj_data.xmat[self.stick_body_id].reshape(3, 3)[:, 2]
        t = np.clip(np.dot(rock_center - stick_center, stick_axis),
                    -self._stick_half_len, self._stick_half_len)
        nearest = stick_center + t * stick_axis
        dist = np.linalg.norm(rock_center - nearest)
        return dist - (self._rock_radius + self._stick_radius)

    def _compute_running_cost(self, mj_data, u, u_prev, t, w_run=None):
        """
        Running cost matching the OCP create_solver formulation.
        Pass w_run to override self.w_run (used for TV weights inside _rollout).
        """
        w = w_run if w_run is not None else self.w_run
        cost = 0.0

        if w.get('press_force', 0.0) > 0:
            force = self._get_contact_force(mj_data)
            if force > 0.1:
                f_ref = self.force_traj[t] if (self.force_traj is not None and t < len(self.force_traj)) else self.target_force
                if self.force_ramp_steps > 0 and t < self.force_ramp_steps:
                    f_ref *= (t + 1) / self.force_ramp_steps
                cost += w['press_force'] * (force - f_ref) ** 2

        if w.get('rail_pos', 0.0) > 0:
            cost += w['rail_pos'] * self._contact_rail_dist(mj_data) ** 2

        if w.get('Tau', 0.0) > 0:
            cost += w['Tau'] * np.sum(u ** 2)

        if w.get('JV', 0.0) > 0:
            cost += w['JV'] * np.sum(mj_data.qvel ** 2)

        if w.get('Eng', 0.0) > 0:
            v_act = mj_data.qvel[self.arm_dofadr]
            cost += w['Eng'] * np.sum((v_act * u) ** 2)

        if w.get('Geo', 0.0) > 0:
            M_full = np.zeros((self.model.nv, self.model.nv))
            mujoco.mj_fullM(self.model, M_full, mj_data.qM)
            M_arm = M_full[np.ix_(self.arm_dofadr, self.arm_dofadr)]
            v_arm = mj_data.qvel[self.arm_dofadr]
            cost += w['Geo'] * float(v_arm @ M_arm @ v_arm)

        if w.get('JA', 0.0) > 0:
            cost += w['JA'] * np.sum(mj_data.qacc ** 2)

        if w.get('JTC', 0.0) > 0:
            cost += w['JTC'] * np.sum((u - u_prev) ** 2)

        if w.get('rock_ori', 0.0) > 0 and self.rock_body_id >= 0 and self.rock_quat_ref is not None:
            q_cur = mj_data.xquat[self.rock_body_id]
            dot = float(np.dot(q_cur, self.rock_quat_ref))
            cost += w['rock_ori'] * (1.0 - dot ** 2)

        q_per_jnt = mj_data.qpos[self.model.jnt_qposadr]
        limited = self.model.jnt_limited.astype(bool)
        q_violation = np.where(limited,
            np.maximum(0, self.q_min - q_per_jnt) + np.maximum(0, q_per_jnt - self.q_max),
            0.0)
        cost += 10000.0 * np.sum(q_violation ** 2)

        return cost

    def _raw_run_features(self, d_sim, u, u_prev, t, w_run=None):
        """Raw (unweighted) feature values at a single rollout step. Dict keyed by feature name."""
        w = w_run if w_run is not None else self.w_run
        raw = {}
        if 'Tau' in w:
            raw['Tau'] = float(np.sum(u ** 2))
        if 'JV' in w:
            raw['JV'] = float(np.sum(d_sim.qvel ** 2))
        if 'JA' in w:
            raw['JA'] = float(np.sum(d_sim.qacc ** 2))
        if 'JTC' in w:
            raw['JTC'] = float(np.sum((u - u_prev) ** 2))
        if 'Eng' in w:
            v_act = d_sim.qvel[self.arm_dofadr]
            raw['Eng'] = float(np.sum((v_act * u) ** 2))
        if 'Geo' in w:
            M_full = np.zeros((self.model.nv, self.model.nv))
            mujoco.mj_fullM(self.model, M_full, d_sim.qM)
            M_arm = M_full[np.ix_(self.arm_dofadr, self.arm_dofadr)]
            v_arm = d_sim.qvel[self.arm_dofadr]
            raw['Geo'] = float(v_arm @ M_arm @ v_arm)
        if 'press_force' in w:
            force = self._get_contact_force(d_sim)
            if force > 0.1:
                f_ref = self.force_traj[t] if (self.force_traj is not None and t < len(self.force_traj)) else self.target_force
                if self.force_ramp_steps > 0 and t < self.force_ramp_steps:
                    f_ref *= (t + 1) / self.force_ramp_steps
                raw['press_force'] = float((force - f_ref) ** 2)
            else:
                raw['press_force'] = 0.0
        if 'rock_ori' in w and self.rock_body_id >= 0 and self.rock_quat_ref is not None:
            q_cur = d_sim.xquat[self.rock_body_id]
            dot = float(np.dot(q_cur, self.rock_quat_ref))
            raw['rock_ori'] = 1.0 - dot ** 2
        if 'surface' in w:
            raw['surface'] = float(self._rock_stick_gap(d_sim) ** 2)
        if 'progress' in w and self.p_end_world is not None and self.p_start_world is not None:
            pos = self._get_contact_position(d_sim)
            rail_vec = self.p_end_world - self.p_start_world
            rail_len = np.linalg.norm(rail_vec)
            if rail_len > 1e-9:
                traveled = np.dot(pos - self.p_start_world, rail_vec / rail_len)
                raw['progress'] = float(max(0.0, rail_len - traveled))
        vel_key = 'target_vel' if 'target_vel' in w else ('progress_vel' if 'progress_vel' in w else None)
        if vel_key is not None and self.p_start_world is not None and self.p_end_world is not None:
            rail_vec = self.p_end_world - self.p_start_world
            rail_len = np.linalg.norm(rail_vec)
            if rail_len > 1e-9:
                rail_unit = rail_vec / rail_len
                J = np.zeros((3, self.model.nv))
                mujoco.mj_jacSite(self.model, d_sim, J, None, self.contact_site_id)
                v_rail = float(np.dot(J @ d_sim.qvel, rail_unit))
                raw[vel_key] = (v_rail - self.target_rail_vel) ** 2
        if 'rail_lat' in w and self.p_start_world is not None and self.p_end_world is not None:
            pos = self._get_contact_position(d_sim)
            rail_vec = self.p_end_world - self.p_start_world
            rail_len = np.linalg.norm(rail_vec)
            if rail_len > 1e-9:
                rail_unit = rail_vec / rail_len
                diff = pos - self.p_start_world
                lateral = diff - np.dot(diff, rail_unit) * rail_unit
                raw['rail_lat'] = float(np.dot(lateral, lateral))
        return raw

    def _raw_term_features(self, d_sim, u_last=None, w_term=None):
        """Raw (unweighted) terminal feature values."""
        w = w_term if w_term is not None else self.w_term
        raw = {}
        if 'JV' in w:
            raw['JV'] = float(np.sum(d_sim.qvel ** 2))
        if 'JA' in w:
            raw['JA'] = float(np.sum(d_sim.qacc ** 2))
        if 'JTC' in w and u_last is not None:
            raw['JTC'] = float(np.sum(u_last ** 2))
        if 'Geo' in w:
            M_full = np.zeros((self.model.nv, self.model.nv))
            mujoco.mj_fullM(self.model, M_full, d_sim.qM)
            M_arm = M_full[np.ix_(self.arm_dofadr, self.arm_dofadr)]
            v_arm = d_sim.qvel[self.arm_dofadr]
            raw['Geo'] = float(v_arm @ M_arm @ v_arm)
        return raw

    def _compute_terminal_cost(self, mj_data, u_last=None, w_term=None):
        """Terminal cost. Pass w_term to override self.w_term (TV weights)."""
        w = w_term if w_term is not None else self.w_term
        cost = 0.0

        if w.get('JV', 0.0) > 0:
            cost += w['JV'] * np.sum(mj_data.qvel ** 2)

        if w.get('Geo', 0.0) > 0:
            M_full = np.zeros((self.model.nv, self.model.nv))
            mujoco.mj_fullM(self.model, M_full, mj_data.qM)
            M_arm = M_full[np.ix_(self.arm_dofadr, self.arm_dofadr)]
            v_arm = mj_data.qvel[self.arm_dofadr]
            cost += w['Geo'] * float(v_arm @ M_arm @ v_arm)

        if w.get('JA', 0.0) > 0:
            cost += w['JA'] * np.sum(mj_data.qacc ** 2)

        if w.get('JTC', 0.0) > 0 and u_last is not None:
            cost += w['JTC'] * np.sum(u_last ** 2)

        return cost

    def _rollout(self, state_qpos, state_qvel, epsilon_k):
        d_sim = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, d_sim)

        self._set_state(d_sim, state_qpos, state_qvel)

        for qpos_id, dof_id, lock_val in self.locked_joint_constraints:
            d_sim.qpos[qpos_id] = lock_val
            d_sim.qvel[dof_id] = 0.0

        mujoco.mj_forward(self.model, d_sim)

        rollout_cost = 0.0
        u_prev = self.u_prev.copy()
        traj_pos = []
        max_force_this_rollout = 0.0
        min_gap_this_rollout = float('inf')

        keys_run  = sorted(self.w_run.keys())
        keys_term = sorted(self.w_term.keys())
        phi = np.zeros(len(keys_run) + len(keys_term))

        for t in range(self.H):
            for qpos_id, dof_id, lock_val in self.locked_joint_constraints:
                d_sim.qpos[qpos_id] = lock_val
                d_sim.qvel[dof_id] = 0.0

            if self._stick_traj_ctrl is not None and self._stick_qposadr_ctrl is not None:
                t_c = min(t, len(self._stick_traj_ctrl) - 1)
                adr = self._stick_qposadr_ctrl
                d_sim.qpos[adr:adr + 3] = self._stick_traj_ctrl[t_c]

            w_run_t = self._w_run_at(t)

            u = self._clip_u(self.U[t] + epsilon_k[t])

            d_sim.ctrl[:] = 0.0
            d_sim.ctrl[:self.nu] = u
            mujoco.mj_step(self.model, d_sim)

            if not (np.all(np.isfinite(d_sim.qpos)) and np.all(np.isfinite(d_sim.qvel)) and np.all(np.isfinite(d_sim.qacc))):
                rollout_cost += 1e6
                break

            f = self._get_contact_force(d_sim)
            if f > max_force_this_rollout:
                max_force_this_rollout = f

            gap = self._rock_stick_gap(d_sim)
            if gap < min_gap_this_rollout:
                min_gap_this_rollout = gap

            running_cost = self._compute_running_cost(d_sim, u, u_prev, t, w_run=w_run_t)
            rollout_cost += running_cost * self.dt

            if w_run_t.get('surface', 0.0) > 0:
                rollout_cost += w_run_t['surface'] * max(0.0, gap) ** 2
            if w_run_t.get('progress', 0.0) > 0 and self.p_end_world is not None and self.p_start_world is not None:
                pos = self._get_contact_position(d_sim)
                rail_vec = self.p_end_world - self.p_start_world
                rail_len = np.linalg.norm(rail_vec)
                if rail_len > 1e-9:
                    traveled = np.dot(pos - self.p_start_world, rail_vec / rail_len)
                    rollout_cost += w_run_t['progress'] * max(0.0, rail_len - traveled)
            if (w_run_t.get('target_vel', 0.0) > 0 or w_run_t.get('progress_vel', 0.0) > 0) and self.p_start_world is not None and self.p_end_world is not None:
                rail_vec = self.p_end_world - self.p_start_world
                rail_len = np.linalg.norm(rail_vec)
                if rail_len > 1e-9:
                    rail_unit = rail_vec / rail_len
                    J = np.zeros((3, self.model.nv))
                    mujoco.mj_jacSite(self.model, d_sim, J, None, self.contact_site_id)
                    v_rail = float(np.dot(J @ d_sim.qvel, rail_unit))
                    w_vel = w_run_t.get('target_vel', 0.0) or w_run_t.get('progress_vel', 0.0)
                    rollout_cost += w_vel * (v_rail - self.target_rail_vel) ** 2
            if w_run_t.get('rail_lat', 0.0) > 0 and self.p_start_world is not None and self.p_end_world is not None:
                pos = self._get_contact_position(d_sim)
                rail_vec = self.p_end_world - self.p_start_world
                rail_len = np.linalg.norm(rail_vec)
                if rail_len > 1e-9:
                    rail_unit = rail_vec / rail_len
                    diff = pos - self.p_start_world
                    lateral = diff - np.dot(diff, rail_unit) * rail_unit
                    rollout_cost += w_run_t['rail_lat'] * float(np.dot(lateral, lateral))

            traj_pos.append(self._get_contact_position(d_sim))
            raw = self._raw_run_features(d_sim, u, u_prev, t, w_run=w_run_t)
            for j, k in enumerate(keys_run):
                phi[j] += raw.get(k, 0.0) * self.dt
            u_prev = u.copy()

        w_term_t = self._w_term_final()
        rollout_cost += self._compute_terminal_cost(d_sim, u_last=u_prev, w_term=w_term_t)
        raw_term = self._raw_term_features(d_sim, u_last=u_prev, w_term=w_term_t)
        for j, k in enumerate(keys_term):
            phi[len(keys_run) + j] += raw_term.get(k, 0.0)

        return rollout_cost, phi, np.array(traj_pos), max_force_this_rollout, min_gap_this_rollout

    def _build_noise_schedule(self):
        """
        Returns shape (H,) array of per-timestep noise multipliers.

        Decreasing tapers (focus noise on near-term):
          'uniform'       1 everywhere
          'cosine'        0.5*(1 + cos(π·t/H))          1 → 0
          'linear'        1 - t/H                        1 → 0
          'quadratic'     (1 - t/H)²                     1 → 0

        Increasing tapers (more uncertainty in future, DIAL-MPC style):
          'increasing'    t/H                             0 → 1
          'increasing_cos' 0.5*(1 - cos(π·t/H))          0 → 1
        """
        tau = np.linspace(0.0, 1.0, self.H)
        taper = getattr(self, 'noise_taper', 'cosine')
        if taper == 'linear':
            scale = 1.0 - tau
        elif taper == 'quadratic':
            scale = (1.0 - tau) ** 2
        elif taper == 'uniform':
            scale = np.ones(self.H)
        elif taper == 'increasing':
            scale = tau
        elif taper == 'increasing_cos':
            scale = 0.5 * (1.0 - np.cos(np.pi * tau))
        else:
            scale = 0.5 * (1.0 + np.cos(np.pi * tau))
        self._taper = scale
        return self.sigma * scale

    def _resample_contact_elite(self, epsilon, results, state_qpos, state_qvel):
        """
        If too few rollouts made contact, replace non-contact slots with
        resampled perturbations around contact-making elite (tighter noise).
        """
        max_forces = np.array([res[3] for res in results])
        contact_mask = max_forces > 0.1
        n_contact = int(contact_mask.sum())

        if n_contact == 0 or n_contact >= self.contact_elite_min:
            return epsilon, results

        contact_idx = np.where(contact_mask)[0]
        non_contact_idx = np.where(~contact_mask)[0]
        elite_eps = epsilon[contact_idx]

        n_resample = len(non_contact_idx)
        boot_idx = np.random.choice(len(contact_idx), size=n_resample)
        jitter = np.random.normal(0, 1.0, (n_resample, self.H, self.nu))
        jitter *= self.noise_schedule[None, :, None] * self.contact_elite_jitter
        new_eps = elite_eps[boot_idx] + jitter

        new_results = [self._rollout(state_qpos, state_qvel, new_eps[i])
                       for i in range(n_resample)]

        epsilon_out = epsilon.copy()
        results_out = list(results)
        for i, idx in enumerate(non_contact_idx):
            epsilon_out[idx] = new_eps[i]
            results_out[idx] = new_results[i]

        return epsilon_out, results_out

    def _sample_noise(self, noise_scale=1.0):
        """
        Sample exploration noise: (K, H, nu).

        ε_{k,h,j} ~ N(0, noise_scale · σ_j · taper(h))

        - σ_j = per_joint_sigma[j] (from limits) or self.sigma (scalar)
        - taper(h) = horizon shape (increasing for DIAL-MPC)
        - noise_scale = decay^iter (DIAL iteration annealing)
        """
        epsilon = np.random.normal(0, 1.0, (self.K, self.H, self.nu))
        if self._per_joint_sigma is not None:
            epsilon *= (noise_scale
                        * self._per_joint_sigma[None, None, :]
                        * self._taper[None, :, None])
        else:
            epsilon *= noise_scale * self.noise_schedule[None, :, None]
        return epsilon

    def _mppi_update(self, state_qpos, state_qvel, noise_scale=1.0):
        """
        Single MPPI iteration: sample, rollout, update U.
        Returns (u_best, E_phi).
        """
        epsilon = self._sample_noise(noise_scale)
        results = [self._rollout(state_qpos, state_qvel, epsilon[k]) for k in range(self.K)]

        if self.contact_elite:
            epsilon, results = self._resample_contact_elite(
                epsilon, results, state_qpos, state_qvel)

        costs      = np.array([res[0] for res in results])
        phis       = np.array([res[1] for res in results])
        trajs      = [res[2] for res in results]
        max_forces = np.array([res[3] for res in results])

        finite_mask = np.isfinite(costs)
        if not np.any(finite_mask):
            return None, None

        costs = np.where(finite_mask, costs, costs[finite_mask].max())

        self.last_costs = costs
        best_idx = np.argmin(costs)
        self.best_traj = trajs[best_idx]
        self.all_trajs = trajs

        elite_idx = np.arange(len(costs))
        if self.contact_elite:
            contact_mask = max_forces > 0.1
            if contact_mask.sum() >= self.contact_elite_min:
                elite_idx = np.where(contact_mask)[0]

        costs_e   = costs[elite_idx]
        phis_e    = phis[elite_idx]
        epsilon_e = epsilon[elite_idx]

        beta = costs_e.min()
        weights = np.exp(-(costs_e - beta) / self.lam)
        weights /= weights.sum() + 1e-10

        self._last_E_phi = weights @ phis_e
        self._last_phi_keys = sorted(self.w_run.keys()) + sorted(self.w_term.keys())

        for h in range(self.H):
            self.U[h] += np.sum(weights[:, None] * epsilon_e[:, h, :], axis=0)
            self.U[h] = self._clip_u(self.U[h])

        return self._clip_u(self.U[0].copy()), self._last_E_phi

    def step(self, state_qpos, state_qvel):
        u_apply = None
        for i in range(self.dial_iters):
            noise_scale = self.dial_noise_decay ** i
            u, _ = self._mppi_update(state_qpos, state_qvel, noise_scale)
            if u is not None:
                u_apply = u

        if u_apply is None:
            return self._clip_u(self._gravity_compensation(state_qpos))

        self.U = np.roll(self.U, -1, axis=0)
        self.U[-1] = self.U[-2]

        self.u_prev = u_apply.copy()
        return u_apply

    def update(self, state_qpos, state_qvel):
        """
        Single full-horizon MPPI update — no horizon shift.

        Same as step() but U is NOT rolled after the update.
        Use this for full-horizon trajectory optimisation (e.g. IRL rollouts)
        where the nominal trajectory covers the entire motion at once.

        Returns
        -------
        E_phi : (nr,) weighted feature expectation for the current weights.
        """
        E_phi = None
        for i in range(self.dial_iters):
            noise_scale = self.dial_noise_decay ** i
            _, e = self._mppi_update(state_qpos, state_qvel, noise_scale)
            if e is not None:
                E_phi = e

        if E_phi is None:
            return self._last_E_phi if hasattr(self, '_last_E_phi') else np.zeros(1)
        return E_phi

    def _gravity_compensation(self, qpos):
        """Fallback: return gravity compensation torques via MuJoCo bias force."""
        d_tmp = mujoco.MjData(self.model)
        d_tmp.qpos[:] = qpos[:self.nq]
        d_tmp.qvel[:] = 0.0
        mujoco.mj_forward(self.model, d_tmp)
        return d_tmp.qfrc_bias[self.arm_dofadr]

    def set_weights(self, w_run=None, w_term=None):
        """Update cost weights (matching OCP interface)"""
        if w_run is not None:
            self.w_run = dict(w_run)

        if w_term is not None:
            self.w_term = dict(w_term)