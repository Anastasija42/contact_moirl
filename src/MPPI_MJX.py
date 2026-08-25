"""
GPU-accelerated MPPI via MuJoCo MJX + JAX.

Drop-in replacement for MPPIController.  Uses:
  - jax.vmap  : K rollouts in parallel on GPU
  - jax.lax.scan : H-step time loop compiled to XLA
  - jax.jit   : full compilation on first call

Key difference from the CPU version:
  - mj_fullM not needed — MJX already stores qM as a dense (nv,nv) matrix
  - Pinocchio not used anywhere
  - Contact forces from efc_force (same physics, just JAX arrays)
"""

import numpy as np
import mujoco
import mujoco.mjx as mjx
import jax
import jax.numpy as jnp

def _prepare_model_for_mjx(model):
    """
    Make the model MJX-compatible in-place:
      - disable collisions on mesh geoms (human body) — MJX can't handle mesh-cylinder
      - convert remaining cylinder geoms → capsule (same shape, MJX-supported)
    Returns the (modified) model.
    """
    MESH     = int(mujoco.mjtGeom.mjGEOM_MESH)
    CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    CAPSULE  = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    for i in range(model.ngeom):
        if model.geom_type[i] == MESH:
            model.geom_contype[i]     = 0
            model.geom_conaffinity[i] = 0
        elif model.geom_type[i] == CYLINDER:
            model.geom_type[i] = CAPSULE
    return model

def _tile_data(dx, K):
    """Tile a single MJX Data into a batch of K identical copies."""
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x[None], (K,) + x.shape),
        dx,
    )

class MPPIMJXController:
    """
    GPU MPPI controller.  Same public interface as MPPIController:
        step(qpos, qvel)   → u_apply (np.ndarray)
        set_weights(w_run, w_term)
        _last_E_phi        (np.ndarray, nr)
        _last_phi_keys     (list[str])
        warmstart()        → U (np.ndarray H×nu)
    """

    KEYS_RUN  = ['Tau', 'JV', 'JA', 'JTC', 'Eng', 'Geo',
                 'press_force', 'surface', 'progress_vel', 'rail_lat',
                 'rock_ori', 'progress']
    KEYS_TERM = ['JV', 'JA', 'JTC', 'Geo']

    def __init__(self, model, data, pin_model, pin_data, contact_frame_id,
                 horizon=20, num_samples=512, noise_sigma=0.5, lambda_=0.1,
                 target_force=10.0, p_start_world=None, p_end_world=None, mu=0.3,
                 locked_joint_constraints=None, q_traj=None, x0=None):

        self.H   = horizon
        self.K   = num_samples
        self.sigma = noise_sigma
        self.lam = lambda_
        self.target_force = target_force
        self.force_traj   = None
        self.traveled_ref = None
        self.p_start_world = np.array(p_start_world) if p_start_world is not None else None
        self.p_end_world   = np.array(p_end_world)   if p_end_world   is not None else None
        self.mu  = mu
        self.dt  = model.opt.timestep
        self._n_substeps = 1

        self.nu = model.nu
        self.nq = model.nq
        self.nv = model.nv

        act_qposadr, act_dofadr = [], []
        for i in range(self.nu):
            jnt_id = int(model.actuator_trnid[i, 0])
            act_qposadr.append(int(model.jnt_qposadr[jnt_id]))
            act_dofadr.append(int(model.jnt_dofadr[jnt_id]))
        self.arm_qposadr = np.array(act_qposadr, dtype=int)
        self.arm_dofadr  = np.array(act_dofadr,  dtype=int)

        self._u_min = jnp.array(model.actuator_ctrlrange[:, 0].copy())
        self._u_max = jnp.array(model.actuator_ctrlrange[:, 1].copy())

        self._jnt_qposadr = jnp.array(model.jnt_qposadr.copy())
        self._jnt_limited = jnp.array(model.jnt_limited.astype(bool))
        self._q_min = jnp.array(model.jnt_range[:, 0].copy())
        self._q_max = jnp.array(model.jnt_range[:, 1].copy())
        if jnp.all(self._u_min == 0) and jnp.all(self._u_max == 0):
            self._u_min = jnp.full(self.nu, -500.0)
            self._u_max = jnp.full(self.nu,  500.0)

        self.x0 = x0 if x0 is not None else np.zeros(model.nq + model.nv)

        if locked_joint_constraints is None:
            locked_joint_constraints = []
        self.locked_joint_constraints = locked_joint_constraints
        if locked_joint_constraints:
            lq = np.array([c[0] for c in locked_joint_constraints], dtype=int)
            ld = np.array([c[1] for c in locked_joint_constraints], dtype=int)
            lv = np.array([c[2] for c in locked_joint_constraints])
        else:
            lq = np.zeros(0, dtype=int)
            ld = np.zeros(0, dtype=int)
            lv = np.zeros(0)
        self._locked_qpos_ids = jnp.array(lq)
        self._locked_dof_ids  = jnp.array(ld)
        self._locked_vals     = jnp.array(lv)
        self._has_locked      = len(locked_joint_constraints) > 0

        self._contact_site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, "rock_contact_point")
        self._rock_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "rock_sphere")
        self.best_traj = None
        self.all_trajs = []

        self._stick_radius   = 0.02
        self._rock_radius    = 0.02
        self._stick_half_len = 0.4
        self.stick_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stick")
        if self.stick_body_id < 0:
            self.stick_body_id = None

        self._rock_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rock")
        self._rock_quat_ref = None
        if self._rock_body_id >= 0:
            d_tmp = mujoco.MjData(model)
            mujoco.mj_forward(model, d_tmp)
            self._rock_quat_ref = d_tmp.xquat[self._rock_body_id].copy()

        self.force_ramp_steps = 0
        self.target_rail_vel = 0.0
        self._rock_quat_ref_jax = jnp.array(self._rock_quat_ref) if self._rock_quat_ref is not None else None

        import copy
        mjx_model = copy.deepcopy(model)
        _prepare_model_for_mjx(mjx_model)
        self.mx = mjx.put_model(mjx_model)
        self._mj_model = model

        self._arm_dofadr_jax = jnp.array(self.arm_dofadr)

        if self.p_start_world is not None and self.p_end_world is not None:
            rail_vec = self.p_end_world - self.p_start_world
            rail_len = float(np.linalg.norm(rail_vec))
            self._rail_len     = rail_len
            self._rail_dir     = jnp.array(rail_vec / max(rail_len, 1e-9))
            self._p_start_jax  = jnp.array(self.p_start_world)
            self._p_end_jax    = jnp.array(self.p_end_world)
        else:
            self._rail_len = 0.0

        self.w_run  = {}
        self.w_term = {}
        self._w_run_windows  = None
        self._w_term_windows = None

        self._phi_keys_run  = self.KEYS_RUN
        self._phi_keys_term = self.KEYS_TERM
        self.nr_run  = len(self.KEYS_RUN)
        self.nr_term = len(self.KEYS_TERM)
        self.nr      = self.nr_run + self.nr_term

        self.U = self.warmstart()
        self.u_prev = np.zeros(self.nu)

        self._per_joint_sigma = None
        self._noise_sigma_jax = jnp.full(self.nu, self.sigma)
        self.dial_iters = 1
        self.dial_noise_decay = 0.5

        self._last_E_phi    = np.zeros(self.nr)
        self._last_phi_keys = self.KEYS_RUN + self.KEYS_TERM

        self._rng_key = jax.random.PRNGKey(0)

        _dx_tmp = mjx.make_data(self.mx)
        _dx_tmp = mjx.forward(self.mx, _dx_tmp)
        self._dx_template = _dx_tmp

        self._jit_rollouts      = jax.jit(self._batch_rollouts_jax,      static_argnums=(10, 11))
        self._jit_rollouts_full = jax.jit(self._batch_rollouts_full_jax, static_argnums=(10, 11))
        self.last_costs = np.zeros(self.K)
        print("[MPPI-MJX] JIT-compiling on first call…")

    def set_weights(self, w_run=None, w_term=None):
        if w_run  is not None: self.w_run  = dict(w_run)
        if w_term is not None: self.w_term = dict(w_term)

    def set_tv_weights(self, w_run_windows, w_term_windows=None):
        """
        Register per-window weight dicts for time-varying costs.

        w_run_windows  : list of w_run dicts (one per window), or None to clear
        w_term_windows : list of w_term dicts (one per window), or None
        """
        self._w_run_windows  = list(w_run_windows)  if w_run_windows  else None
        self._w_term_windows = list(w_term_windows) if w_term_windows else None

    def _w_run_arr(self):
        return jnp.array([self.w_run.get(k, 0.0)  for k in self.KEYS_RUN])

    def _w_term_arr(self):
        return jnp.array([self.w_term.get(k, 0.0) for k in self.KEYS_TERM])

    def _w_run_schedule(self):
        """
        Build (H, nr_run) weight array for the rollout horizon.
        If TV windows are set, each timestep gets its window's weights;
        otherwise broadcast the single w_run across all H steps.
        """
        H = self.H
        if self._w_run_windows is not None:
            n_w = len(self._w_run_windows)
            window_size = max(1, H // n_w)
            rows = []
            for h in range(H):
                k = min(h // window_size, n_w - 1)
                rows.append([self._w_run_windows[k].get(key, 0.0)
                             for key in self.KEYS_RUN])
            return jnp.array(rows, dtype=jnp.float32)
        else:
            return jnp.tile(self._w_run_arr(), (H, 1))

    def _w_term_final(self):
        """Return the w_term array for the terminal cost (last window)."""
        if self._w_term_windows:
            return jnp.array([self._w_term_windows[-1].get(k, 0.0)
                              for k in self.KEYS_TERM])
        return self._w_term_arr()

    def set_noise_from_limits(self, fraction=1.0, limits=None):
        """Set per-joint noise sigma proportional to torque limits."""
        if limits is None:
            limits = np.array(self._u_max)
        self._per_joint_sigma = fraction * np.array(limits)
        self._noise_sigma_jax = jnp.array(self._per_joint_sigma)
        print(f"[MPPI-MJX] per-joint noise: fraction={fraction}, "
              f"σ range=[{self._per_joint_sigma.min():.1f}, {self._per_joint_sigma.max():.1f}]")

    def set_dial(self, iters=3, noise_decay=0.5):
        """Enable DIAL-MPC style multi-iteration noise annealing."""
        self.dial_iters = iters
        self.dial_noise_decay = noise_decay
        print(f"[MPPI-MJX] DIAL: {iters} iters/step, noise_decay={noise_decay}")

    def warmstart(self):
        U = np.zeros((self.H, self.nu))
        d_tmp = mujoco.MjData(self._mj_model)
        d_tmp.qpos[:] = self.x0[:self.nq]
        d_tmp.qvel[:] = 0.0
        mujoco.mj_forward(self._mj_model, d_tmp)
        tau_grav = d_tmp.qfrc_bias[self.arm_dofadr]
        for t in range(self.H):
            U[t] = np.clip(tau_grav, np.array(self._u_min), np.array(self._u_max))
        return U

    def warmstart_from_demo(self, q_demo, locked_joint_constraints=None,
                            stick_traj=None, stick_qposadr=None):
        """
        Gravity-comp warmstart along the demo trajectory.
        Computes qfrc_bias (gravity + Coriolis) at each demo frame.
        Contact forces are NOT included — MPPI discovers those via the cost.

        q_demo: (N, nq_arm) joint angles from demo.
        Returns U: (H, nu) clipped.
        """
        H = min(self.H, len(q_demo))
        u_min, u_max = np.array(self._u_min), np.array(self._u_max)

        arm_qposadr = []
        for da in self.arm_dofadr:
            for j in range(self._mj_model.njnt):
                if self._mj_model.jnt_dofadr[j] == da:
                    arm_qposadr.append(self._mj_model.jnt_qposadr[j])
                    break
        arm_qposadr = np.array(arm_qposadr)

        d_tmp = mujoco.MjData(self._mj_model)
        U = np.zeros((self.H, self.nu))

        for t in range(H):
            mujoco.mj_resetData(self._mj_model, d_tmp)
            for i, (qa, da) in enumerate(zip(arm_qposadr, self.arm_dofadr)):
                d_tmp.qpos[qa] = q_demo[t, i]
            if locked_joint_constraints is not None:
                for qpos_id, dof_id, lock_val in locked_joint_constraints:
                    d_tmp.qpos[qpos_id] = lock_val
                    d_tmp.qvel[dof_id]  = 0.0
            if stick_traj is not None and stick_qposadr is not None:
                sidx = min(t, len(stick_traj) - 1)
                d_tmp.qpos[stick_qposadr:stick_qposadr + 3] = stick_traj[sidx]
            mujoco.mj_forward(self._mj_model, d_tmp)
            U[t] = np.clip(d_tmp.qfrc_bias[self.arm_dofadr], u_min, u_max)

        for t in range(H, self.H):
            U[t] = U[H - 1]

        print(f"[MPPI] warmstart_from_demo (gravity comp): H={H}, "
              f"tau range=[{U[:H].min():.1f}, {U[:H].max():.1f}] Nm")
        return U

    def warmstart_from_crocoddyl(self, us_croc):
        """
        Warmstart from Crocoddyl OCP solution torques.
        us_croc includes contact compensation via J^T*lambda.

        us_croc: (T, nu) or list of (nu,) arrays.
        Returns U: (H, nu) clipped.
        """
        if isinstance(us_croc, list):
            us_croc = np.array(us_croc)
        H = min(self.H, len(us_croc))
        u_min, u_max = np.array(self._u_min), np.array(self._u_max)

        U = np.zeros((self.H, self.nu))
        nu = min(us_croc.shape[1], self.nu)
        for t in range(H):
            U[t] = np.clip(us_croc[t, :nu], u_min, u_max)
        for t in range(H, self.H):
            U[t] = U[H - 1]

        print(f"[MPPI] warmstart_from_crocoddyl: H={H}, "
              f"tau range=[{U[:H].min():.1f}, {U[:H].max():.1f}] Nm")
        return U

    def _running_cost_jax(self, dx, u, u_prev, h, w_run_h, target_force, target_rail_vel):
        """
        Returns (scalar_cost, phi_vec) for one rollout step.
        w_run_h: (nr_run,) weight array for this timestep, ordered by KEYS_RUN.
        All features use 0.5*||r||^2 convention to match Crocoddyl/Pinocchio.
        """
        dt = self.dt
        arm = self._arm_dofadr_jax
        site_id = self._contact_site_id
        mx = self.mx

        phi_tau = 0.5 * jnp.sum(u ** 2)
        phi_jv  = 0.5 * jnp.sum(dx.qvel[arm] ** 2)
        phi_ja  = 0.5 * jnp.sum(dx.qacc[arm] ** 2)
        def bias_fn(q):
            dx_q = dx.replace(qpos=q)
            dx_q = mjx.forward(mx, dx_q)
            return dx_q.qfrc_bias[arm]
        nq = dx.qpos.shape[0]
        nv = dx.qvel.shape[0]
        tangent = jnp.zeros(nq)
        tangent = tangent.at[:nv].set(dx.qvel)
        _, dtau_dq_v = jax.jvp(bias_fn, (dx.qpos,), (tangent,))
        phi_jtc = 0.5 * jnp.sum(dtau_dq_v ** 2)
        v_act   = dx.qvel[arm]
        phi_eng = 0.5 * jnp.sum((v_act * u) ** 2)
        phi_geo  = 0.5 * (dx.qvel @ dx._impl.qM @ dx.qvel)
        efc_addr     = dx._impl.contact.efc_address[0]
        contact_dist = dx._impl.contact.dist[0]
        f_contact    = jnp.where(contact_dist < 0,
                                 jnp.abs(dx._impl.efc_force[efc_addr]), 0.0)
        phi_pf = 0.5 * (f_contact - target_force) ** 2
        phi_surf = 0.5 * jnp.where(contact_dist > 0, contact_dist ** 2, 0.0)
        if self._rail_len > 1e-9:
            def site_pos_fn(q):
                dx_q = dx.replace(qpos=q)
                dx_q = mjx.forward(mx, dx_q)
                return dx_q.site_xpos[site_id]
            _, v_site = jax.jvp(site_pos_fn, (dx.qpos,), (tangent,))
            v_proj = jnp.dot(v_site, self._rail_dir)
            phi_prog_vel = (v_proj - target_rail_vel) ** 2
        else:
            phi_prog_vel = jnp.array(0.0)

        if self._rail_len > 1e-9:
            site_pos = dx.site_xpos[site_id]
            diff = site_pos - self._p_start_jax
            lateral = diff - jnp.dot(diff, self._rail_dir) * self._rail_dir
            phi_rail_lat = jnp.dot(lateral, lateral)
        else:
            phi_rail_lat = jnp.array(0.0)

        if self._rock_body_id >= 0 and self._rock_quat_ref_jax is not None:
            q_cur = dx.xquat[self._rock_body_id]
            dot = jnp.dot(q_cur, self._rock_quat_ref_jax)
            phi_rock_ori = 1.0 - dot ** 2
        else:
            phi_rock_ori = jnp.array(0.0)

        if self._rail_len > 1e-9:
            site_pos_prog = dx.site_xpos[site_id]
            traveled = jnp.dot(site_pos_prog - self._p_start_jax, self._rail_dir)
            phi_progress = jnp.maximum(0.0, self._rail_len - traveled)
        else:
            phi_progress = jnp.array(0.0)

        phi_run = jnp.stack([phi_tau, phi_jv, phi_ja, phi_jtc,
                              phi_eng, phi_geo, phi_pf, phi_surf,
                              phi_prog_vel, phi_rail_lat,
                              phi_rock_ori, phi_progress])

        cost = jnp.dot(w_run_h, phi_run) * dt

        q_per_jnt = dx.qpos[self._jnt_qposadr]
        q_violation = jnp.where(self._jnt_limited,
            jnp.maximum(0.0, self._q_min - q_per_jnt) + jnp.maximum(0.0, q_per_jnt - self._q_max),
            0.0)
        cost += 10000.0 * jnp.sum(q_violation ** 2)

        return cost, phi_run * dt

    def _terminal_cost_jax(self, dx, u_last, w_term):
        """
        Returns (scalar_cost, phi_vec) for terminal state.
        w_term: (nr_term,) ordered by KEYS_TERM = ['JV','JA','JTC','Geo'].
        All features use 0.5*||r||^2 convention to match Crocoddyl/Pinocchio.
        """
        arm = self._arm_dofadr_jax
        phi_jv  = 0.5 * jnp.sum(dx.qvel[arm] ** 2)
        phi_ja  = 0.5 * jnp.sum(dx.qacc[arm] ** 2)
        phi_jtc = 0.5 * jnp.sum(u_last ** 2)
        phi_geo = 0.5 * (dx.qvel @ dx._impl.qM @ dx.qvel)

        phi_term = jnp.stack([phi_jv, phi_ja, phi_jtc, phi_geo])
        cost = jnp.dot(w_term, phi_term)
        return cost, phi_term

    def _batch_rollouts_jax(self, qpos, qvel, U, key,
                            w_run_all, w_term, u_prev_0, target_force,
                            target_rail_vel, noise_sigma,
                            n_dial, dial_decay):
        """
        Run K rollouts on GPU with DIAL iterations — all in one JIT call.
        w_run_all: (H, nr_run) per-timestep weight schedule.
        Returns: costs(K), phis(K,nr), U_shifted(H,nu), E_phi(nr),
                 u_apply(nu), all_positions(K,H,3)
        """
        K, H, nu = self.K, self.H, self.nu
        mx = self.mx
        site_id = self._contact_site_id

        dx0 = self._dx_template.replace(qpos=qpos, qvel=qvel,
                                        ctrl=jnp.zeros(mx.nu))
        dx0 = mjx.forward(mx, dx0)

        def rollout_k(eps_k, U_cur):
            def step_fn(carry, h):
                dx, u_prev, cost_acc, phi_acc = carry
                u_nom = U_cur[h]
                u = jnp.clip(u_nom + eps_k[h], self._u_min, self._u_max)
                ctrl = dx.ctrl.at[:nu].set(u)
                dx = dx.replace(ctrl=ctrl)
                if self._has_locked:
                    dx = dx.replace(
                        qpos=dx.qpos.at[self._locked_qpos_ids].set(self._locked_vals),
                        qvel=dx.qvel.at[self._locked_dof_ids].set(0.0),
                    )
                dx = jax.lax.fori_loop(
                    0, self._n_substeps, lambda _, d: mjx.step(mx, d), dx)
                step_cost, step_phi = self._running_cost_jax(
                    dx, u, u_prev, h, w_run_all[h], target_force, target_rail_vel)
                site_pos = dx.site_xpos[site_id]
                return (dx, u, cost_acc + step_cost, phi_acc + step_phi), site_pos

            init = (dx0, u_prev_0, jnp.array(0.0), jnp.zeros(self.nr_run))
            (dx_final, u_last, total_cost, phi_run), traj_positions = jax.lax.scan(
                step_fn, init, jnp.arange(H))
            term_cost, phi_term = self._terminal_cost_jax(dx_final, u_last, w_term)
            phi_full = jnp.concatenate([phi_run, phi_term])
            return total_cost + term_cost, phi_full, traj_positions

        def dial_iter(carry, _):
            U_cur, key_cur, scale = carry
            key_cur, subkey = jax.random.split(key_cur)
            epsilon = jax.random.normal(subkey, (K, H, nu)) * noise_sigma[None, None, :] * scale

            costs, phis, all_pos = jax.vmap(lambda e: rollout_k(e, U_cur))(epsilon)

            finite_mask = jnp.isfinite(costs)
            safe_costs  = jnp.where(finite_mask, costs,
                                    jnp.max(jnp.where(finite_mask, costs, -jnp.inf)))
            beta    = jnp.min(safe_costs)
            weights = jnp.exp(-(safe_costs - beta) / self.lam)
            weights = weights / (jnp.sum(weights) + 1e-10)

            E_phi  = weights @ phis
            U_next = U_cur + jnp.einsum('k,khn->hn', weights, epsilon)
            U_next = jnp.clip(U_next, self._u_min, self._u_max)

            return (U_next, key_cur, scale * dial_decay), (costs, phis, E_phi, all_pos)

        init_carry = (U, key, jnp.array(1.0))
        (U_final, _, _), (all_costs, all_phis, all_E_phi, all_positions) = jax.lax.scan(
            dial_iter, init_carry, jnp.arange(n_dial))

        u_apply   = U_final[0]
        U_shifted = jnp.concatenate([U_final[1:], U_final[-1:]], axis=0)

        return all_costs[-1], all_phis[-1], U_shifted, all_E_phi[-1], u_apply, all_positions[-1]

    def _batch_rollouts_full_jax(self, qpos, qvel, U, key,
                                 w_run_all, w_term, u_prev_0, target_force,
                                 target_rail_vel, noise_sigma,
                                 n_dial, dial_decay):
        """
        Full-horizon MPPI with DIAL iterations — all on GPU.
        w_run_all: (H, nr_run) per-timestep weight schedule.
        Runs n_dial iterations, decaying noise by dial_decay each time.
        Returns (costs, phis, U_new, E_phi).
        """
        K, H, nu = self.K, self.H, self.nu
        mx = self.mx

        dx0 = self._dx_template.replace(qpos=qpos, qvel=qvel,
                                        ctrl=jnp.zeros(mx.nu))
        dx0 = mjx.forward(mx, dx0)

        def rollout_k(eps_k, U_cur):
            def step_fn(carry, h):
                dx, u_prev, cost_acc, phi_acc = carry
                u_nom = U_cur[h]
                u = jnp.clip(u_nom + eps_k[h], self._u_min, self._u_max)
                ctrl = dx.ctrl.at[:nu].set(u)
                dx = dx.replace(ctrl=ctrl)
                if self._has_locked:
                    dx = dx.replace(
                        qpos=dx.qpos.at[self._locked_qpos_ids].set(self._locked_vals),
                        qvel=dx.qvel.at[self._locked_dof_ids].set(0.0),
                    )
                dx = mjx.step(mx, dx)
                step_cost, step_phi = self._running_cost_jax(
                    dx, u, u_prev, h, w_run_all[h], target_force, target_rail_vel)
                return (dx, u, cost_acc + step_cost, phi_acc + step_phi), None

            init = (dx0, u_prev_0, jnp.array(0.0), jnp.zeros(self.nr_run))
            (dx_final, u_last, total_cost, phi_run), _ = jax.lax.scan(
                step_fn, init, jnp.arange(H))
            term_cost, phi_term = self._terminal_cost_jax(dx_final, u_last, w_term)
            phi_full = jnp.concatenate([phi_run, phi_term])
            return total_cost + term_cost, phi_full

        def dial_iter(carry, _):
            U_cur, key_cur, scale = carry
            key_cur, subkey = jax.random.split(key_cur)

            epsilon = jax.random.normal(subkey, (K, H, nu)) * noise_sigma[None, None, :] * scale

            costs, phis = jax.vmap(lambda e: rollout_k(e, U_cur))(epsilon)

            finite_mask = jnp.isfinite(costs)
            safe_costs  = jnp.where(finite_mask, costs,
                                    jnp.max(jnp.where(finite_mask, costs, -jnp.inf)))
            beta    = jnp.min(safe_costs)
            weights = jnp.exp(-(safe_costs - beta) / self.lam)
            weights = weights / (jnp.sum(weights) + 1e-10)

            U_next = U_cur + jnp.einsum('k,khn->hn', weights, epsilon)
            U_next = jnp.clip(U_next, self._u_min, self._u_max)

            E_phi = weights @ phis
            return (U_next, key_cur, scale * dial_decay), (costs, phis, E_phi)

        init_carry = (U, key, jnp.array(1.0))
        (U_final, _, _), (all_costs, all_phis, all_E_phi) = jax.lax.scan(
            dial_iter, init_carry, jnp.arange(n_dial))

        return all_costs[-1], all_phis[-1], U_final, all_E_phi[-1]

    def set_stick_traj(self, stick_traj, qposadr):
        """
        Register the pre-computed stick trajectory so that _rollout updates
        the stick body position at each internal timestep.

        stick_traj : (T+1, 3) array of stick center world positions
        qposadr    : int, MuJoCo qpos address of the stick freejoint
        """
        self._stick_traj_ctrl    = np.array(stick_traj)
        self._stick_qposadr_ctrl = int(qposadr)

    def update(self, state_qpos, state_qvel):
        """
        Full-horizon MPPI update with DIAL iterations — single GPU call.
        Returns E_phi (nr,) weighted feature expectation.
        """
        self._rng_key, subkey = jax.random.split(self._rng_key)

        costs, phis, U_new, E_phi = self._jit_rollouts_full(
            jnp.array(state_qpos, dtype=jnp.float32),
            jnp.array(state_qvel, dtype=jnp.float32),
            jnp.array(self.U, dtype=jnp.float32),
            subkey,
            self._w_run_schedule(), self._w_term_final(),
            jnp.array(self.u_prev, dtype=jnp.float32),
            jnp.float32(self.target_force),
            jnp.float32(self.target_rail_vel),
            self._noise_sigma_jax,
            int(self.dial_iters),
            float(self.dial_noise_decay),
        )

        self.U              = np.array(U_new)
        self._last_E_phi    = np.array(E_phi)
        self._last_phi_keys = self.KEYS_RUN + self.KEYS_TERM
        self.last_costs     = np.array(costs)
        return self._last_E_phi

    def step(self, state_qpos, state_qvel):
        """
        MPPI step with DIAL iterations — single GPU call.
        Returns u_apply (np.ndarray, nu).
        """
        self._rng_key, subkey = jax.random.split(self._rng_key)

        costs, phis, U_new, E_phi, u_apply, all_positions = self._jit_rollouts(
            jnp.array(state_qpos, dtype=jnp.float32),
            jnp.array(state_qvel, dtype=jnp.float32),
            jnp.array(self.U, dtype=jnp.float32),
            subkey,
            self._w_run_schedule(), self._w_term_final(),
            jnp.array(self.u_prev, dtype=jnp.float32),
            jnp.float32(self.target_force),
            jnp.float32(self.target_rail_vel),
            self._noise_sigma_jax,
            int(self.dial_iters),
            float(self.dial_noise_decay),
        )

        self.U          = np.array(U_new)
        self.u_prev     = np.array(u_apply)
        self._last_E_phi    = np.array(E_phi)
        self._last_phi_keys = self.KEYS_RUN + self.KEYS_TERM
        self.last_costs     = np.array(costs)

        positions_np = np.array(all_positions)
        costs_np = np.array(costs)
        best_idx = int(np.argmin(costs_np))
        self.best_traj = positions_np[best_idx]
        n_show = min(32, len(positions_np))
        indices = np.linspace(0, len(positions_np) - 1, n_show, dtype=int)
        self.all_trajs = positions_np[indices]

        return np.array(u_apply)

    def _set_state(self, mj_data, qpos, qvel):
        if len(qpos) == self._mj_model.nq:
            mj_data.qpos[:] = qpos
            mj_data.qvel[:] = qvel if len(qvel) == self._mj_model.nv else np.zeros(self._mj_model.nv)
            return
        mj_data.qpos[:] = 0.0
        mj_data.qvel[:] = 0.0
        n = min(len(qpos), len(self.arm_qposadr))
        mj_data.qpos[self.arm_qposadr[:n]] = qpos[:n]
        m = min(len(qvel), len(self.arm_dofadr))
        mj_data.qvel[self.arm_dofadr[:m]] = qvel[:m]

    def _get_contact_position(self, mj_data):
        if self._contact_site_id >= 0:
            return mj_data.site_xpos[self._contact_site_id].copy()
        raise RuntimeError("rock_contact_point site not found in model")

    def _get_contact_force(self, mj_data):
        """
        Compute normal force at rock-stick contact from efc_force.
        Mirrors the JAX version: contact[0], dist < 0, abs(efc_force[efc_address]).
        Returns scalar normal force in Newtons.
        """
        total = 0.0
        for i in range(mj_data.ncon):
            c = mj_data.contact[i]
            if self._rock_geom_id not in (c.geom1, c.geom2):
                continue
            if c.dist < 0:
                total += abs(float(mj_data.efc_force[c.efc_address]))
        return total

    def _rock_stick_gap(self, mj_data):
        """Signed gap between rock surface and stick surface (negative = penetrating)."""
        if self._rock_geom_id < 0 or self.stick_body_id is None:
            return 0.0
        rock_center  = mj_data.geom_xpos[self._rock_geom_id].copy()
        stick_center = mj_data.xpos[self.stick_body_id].copy()
        stick_axis   = mj_data.xmat[self.stick_body_id].reshape(3, 3)[:, 2]
        t = np.clip(np.dot(rock_center - stick_center, stick_axis),
                    -self._stick_half_len, self._stick_half_len)
        nearest = stick_center + t * stick_axis
        dist = np.linalg.norm(rock_center - nearest)
        return dist - (self._rock_radius + self._stick_radius)

    def _raw_run_features(self, d_sim, u, u_prev, t, w_run=None):
        """Raw (unweighted) feature values at a single rollout step (CPU-side, for logging)."""
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
            M_full = np.zeros((self._mj_model.nv, self._mj_model.nv))
            mujoco.mj_fullM(self._mj_model, M_full, d_sim.qM)
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
        if 'rock_ori' in w and self._rock_body_id >= 0 and self._rock_quat_ref is not None:
            q_cur = d_sim.xquat[self._rock_body_id]
            dot = float(np.dot(q_cur, self._rock_quat_ref))
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
        if 'progress_vel' in w and self.p_start_world is not None and self.p_end_world is not None:
            rail_vec = self.p_end_world - self.p_start_world
            rail_len = np.linalg.norm(rail_vec)
            if rail_len > 1e-9:
                rail_unit = rail_vec / rail_len
                J = np.zeros((3, self._mj_model.nv))
                mujoco.mj_jacSite(self._mj_model, d_sim, J, None, self._contact_site_id)
                v_rail = float(np.dot(J @ d_sim.qvel, rail_unit))
                raw['progress_vel'] = (v_rail - self.target_rail_vel) ** 2
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

    def _clip_u(self, u):
        return np.clip(u, np.array(self._u_min), np.array(self._u_max))