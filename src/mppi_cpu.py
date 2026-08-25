"""
mppi_cpu.py
===========
CPU-only KinematicMPPI using MuJoCo (no MJX/JAX).
Deterministic: same weights + same seed → identical trajectory.
Uses threading to parallelize sample rollouts (MuJoCo releases the GIL).

Drop-in replacement for KinematicMPPI for determinism testing and IRL.
"""

import numpy as np
import mujoco
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

KEYS_RUN = ['Tau', 'JV', 'JA', 'JTC',
            'Eng_thoracic', 'Eng_clavicle', 'Eng_shoulder', 'Eng_elbow', 'Eng_wrist',
            'Geo',
            'press_force', 'surface', 'progress_vel', 'rail_lat',
            'rock_ori', 'approach', 'traveled',
            'Tau_thoracic', 'Tau_clavicle', 'Tau_shoulder', 'Tau_elbow', 'Tau_wrist',
            'press_capacity',
            'brace',
            'JV_thoracic', 'JV_clavicle', 'JV_shoulder', 'JV_elbow', 'JV_wrist',
            'ElbowTrack']

ENG_GROUPS = {
    "thoracic": [0],
    "clavicle": [1],
    "shoulder": [2, 3, 4],
    "elbow":    [5, 6],
    "wrist":    [7, 8],
}

import os as _os
ELBOW_LOCAL = np.array(ENG_GROUPS['elbow'], dtype=int)
ELBOW_TRACK_REF = np.deg2rad([float(_x) for _x in
                              _os.environ.get('ELBOW_TRACK_REF_DEG', '49.7,47.0').split(',')])

class KinematicMPPI_CPU:
    """
    CPU-only MPPI controller. Deterministic with same seed.
    Uses ThreadPoolExecutor for parallel sample rollouts.
    """

    KEYS_RUN = KEYS_RUN

    def __init__(self, human_mppi, *, horizon=40, num_samples=256,
                 alpha=1.0, beta=0.1, lambda_=0.5,
                 n_dial=3, dial_decay=0.5,
                 target_force=40.0, target_rail_vel=0.5,
                 n_workers=None):
        model = human_mppi.mj_model
        self._mj_model = model
        self.H = horizon
        self.K = num_samples
        self.lam = lambda_
        self.dt = model.opt.timestep
        self.target_force = target_force
        self.target_force_profile = None
        self.target_rail_vel = target_rail_vel
        self.n_dial = n_dial
        self.dial_decay = dial_decay

        self.nu = model.nu
        self.nq = model.nq
        self.nv = model.nv
        self.nr_run = len(KEYS_RUN)

        ctrl = human_mppi.controller
        self.arm_dofadr = np.array(ctrl.arm_dofadr)

        u_min = model.actuator_ctrlrange[:, 0].copy()
        u_max = model.actuator_ctrlrange[:, 1].copy()
        if np.all(u_min == 0) and np.all(u_max == 0):
            u_min = np.full(self.nu, -500.0)
            u_max = np.full(self.nu, 500.0)
        self.u_min = u_min
        self.u_max = u_max

        lj = human_mppi.locked_joint_constraints
        self.locked_qpos_ids = np.array([c[0] for c in lj]) if lj else np.array([], dtype=int)
        self.locked_dof_ids = np.array([c[1] for c in lj]) if lj else np.array([], dtype=int)
        self.locked_vals = np.array([c[2] for c in lj]) if lj else np.array([])

        self.has_stick = (not human_mppi.stick_static
                          and human_mppi.stick_traj is not None
                          and human_mppi._stick_qposadr is not None)
        self.stick_traj = human_mppi.stick_traj if self.has_stick else None
        self.stick_qposadr = int(human_mppi._stick_qposadr) if self.has_stick else 0

        self.stick_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stick")
        self._stick_body_id = self.stick_body_id
        self.rock_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rock_sphere")
        self._rock_geom_id = self.rock_geom_id
        self.rock_body_id = model.geom_bodyid[self.rock_geom_id] if self.rock_geom_id >= 0 else -1
        self._rock_body_id = self.rock_body_id
        self.stick_half_len = 0.4
        self._stick_half_len = self.stick_half_len
        self.stick_radius = 0.02
        self._stick_radius = self.stick_radius
        self.contact_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rock_contact_point")
        self._contact_site_id = self.contact_site_id

        p_start = human_mppi.p_start_world
        p_end = human_mppi.p_end_world
        rail_vec = np.array(p_end) - np.array(p_start)
        self.rail_len = float(np.linalg.norm(rail_vec))
        self.rail_dir = rail_vec / max(self.rail_len, 1e-9)
        self.p_start = np.array(p_start)

        self.rock_quat_ref = None
        rock_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "rock")
        if rock_body_id >= 0:
            d_ref = mujoco.MjData(model)
            from final_models.human_mppi import ACTIVE_ARM_JOINTS as _AAJ
            for i, name in enumerate(_AAJ):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                if jid >= 0 and i < len(human_mppi.q0):
                    d_ref.qpos[model.jnt_qposadr[jid]] = human_mppi.q0[i]
            mujoco.mj_forward(model, d_ref)
            self.rock_quat_ref = d_ref.xquat[rock_body_id].copy()
            self._rock_quat_ref = self.rock_quat_ref

        d_tmp = mujoco.MjData(model)
        d_tmp.qpos[:] = human_mppi.mj_data.qpos.copy()
        d_tmp.qvel[:] = 0.0
        mujoco.mj_forward(model, d_tmp)
        self._tau_grav_x0 = np.clip(d_tmp.qfrc_bias[self.arm_dofadr],
                                     self.u_min, self.u_max)
        J = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, d_tmp, J, None, self.contact_site_id)
        J_arm = J[:, self.arm_dofadr]
        Sigma = alpha * (J_arm.T @ J_arm) + beta * np.eye(self.nu)
        self.L = np.linalg.cholesky(Sigma)

        self._jnt_qposadr = model.jnt_qposadr.copy()
        self._jnt_limited = model.jnt_limited.astype(bool)
        self._jnt_range = model.jnt_range.copy()

        self.w_run = {k: 0.0 for k in KEYS_RUN}
        self._phi_scale = np.ones(self.nr_run)
        self._w_fixed = np.zeros(self.nr_run)
        self._w_run_windows = None
        self.U = np.zeros((self.H, self.nu))

        self.n_workers = n_workers or min(8, max(1, cpu_count() - 1))
        self.current_t = 0
        
        print(f"[KinematicMPPI_CPU] K={self.K}, H={self.H}, n_dial={self.n_dial}, ")

    def set_weights(self, w_run):
        if w_run is not None:
            self.w_run.update(w_run)

    def set_force_target_profile(self, profile):
        """Per-timestep press_force target. Pass None to fall back to scalar."""
        self.target_force_profile = (
            None if profile is None
            else np.asarray(profile, dtype=np.float64)
        )

    def reset(self):
        self.U = np.zeros((self.H, self.nu))
        self.current_t = 0
        self._rng = np.random.RandomState(42)

    def sample_features(self, human_mppi, K=32, T=None, seed=0):
        """Run K independent stochastic rollouts at the current self.U + noise
        and return per-sample per-timestep features.

        Used for diagnostics (visualising the MPPI-sample distribution that
        drives the IRL gradient — the converged xs_irl is just one realisation,
        K=many gives the picture of E_π[φ]).

        Returns
        -------
        phis : (K, T, nr_run) array of features (skipping JTC, matching
               _compute_phi_fast's indexing).
        """
        T = T or human_mppi.T
        rng = np.random.RandomState(seed)
        model = human_mppi.mj_model

        mujoco.mj_resetData(model, human_mppi.mj_data)
        v0 = getattr(human_mppi, 'v0', np.zeros(model.nv))
        from final_models.human_mppi import ACTIVE_ARM_JOINTS as _AAJ
        for i, name in enumerate(_AAJ):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                human_mppi.mj_data.qpos[model.jnt_qposadr[jid]] = human_mppi.q0[i]
                if i < len(v0):
                    human_mppi.mj_data.qvel[model.jnt_dofadr[jid]] = v0[i]
        for qpos_id, dof_id, lock_val in human_mppi.locked_joint_constraints:
            human_mppi.mj_data.qpos[qpos_id] = lock_val
            human_mppi.mj_data.qvel[dof_id] = 0.0
        if self.has_stick:
            adr = self.stick_qposadr
            human_mppi.mj_data.qpos[adr:adr+3] = self.stick_traj[0]
            human_mppi.mj_data.qvel[adr:adr+6] = 0.0
        mujoco.mj_forward(model, human_mppi.mj_data)
        qpos0 = human_mppi.mj_data.qpos.copy()
        qvel0 = human_mppi.mj_data.qvel.copy()

        phis_all = np.zeros((K, T, len(KEYS_RUN)))
        for k in range(K):
            d = mujoco.MjData(model)
            d.qpos[:] = qpos0
            d.qvel[:] = qvel0
            mujoco.mj_forward(model, d)

            prev_traveled = 0.0
            if self.rail_len > 1e-9 and self.contact_site_id >= 0:
                sp = d.site_xpos[self.contact_site_id]
                prev_traveled = np.dot(sp - self.p_start, self.rail_dir)
            prev_v_arm = d.qvel[self.arm_dofadr].copy()

            z = rng.randn(T, self.nu)
            eps = (self.L @ z.T).T
            U_k = self.U[:T] if self.U.shape[0] >= T else \
                np.vstack([self.U, np.zeros((T - self.U.shape[0], self.nu))])
            U_k = U_k + eps[:T]

            for h in range(T):
                if self.has_stick and self.stick_traj is not None:
                    adr = self.stick_qposadr
                    si = min(h, len(self.stick_traj) - 1)
                    d.qpos[adr:adr+3] = self.stick_traj[si]
                    d.qvel[adr:adr+6] = 0.0

                tau_grav = d.qfrc_bias[self.arm_dofadr]
                u_h = np.clip(tau_grav + U_k[h], self.u_min, self.u_max)
                d.ctrl[:] = u_h
                for qi, di, lv in zip(self.locked_qpos_ids,
                                       self.locked_dof_ids,
                                       self.locked_vals):
                    d.qpos[qi] = lv
                    d.qvel[di] = 0.0
                mujoco.mj_step(model, d)

                tgt = (self.target_force_profile[min(h, len(self.target_force_profile)-1)]
                       if self.target_force_profile is not None else None)
                phi, prev_traveled, prev_v_arm = self._compute_phi_fast(
                    model, d, u_h, prev_traveled, tgt, prev_v_arm=prev_v_arm)
                phis_all[k, h] = phi
        return phis_all

    def _compute_phi_fast(self, model, data, u, prev_traveled=0.0,
                          target_force=None, t_abs=None, prev_v_arm=None):
        """Compute 14 features for one timestep. Skips JTC (expensive).

        prev_traveled: traveled distance at previous timestep (for finite-diff velocity).
        prev_v_arm:    arm-DOF velocity at previous timestep (for kinematic ddq → JA).
                       None on the first step ⇒ phi_ja = 0 (no diff available).
        Returns (phi, current_traveled, current_v_arm) so the caller can chain.
        """
        arm = self.arm_dofadr
        v_arm = data.qvel[arm].copy()

        # Demo (human_mppi.py:_compute_run_feature) MUST mirror these
        phi_tau = float(np.sqrt(np.sum(u**2))) / 50.0
        phi_jv = float(np.sqrt(np.sum(v_arm**2)))
        if prev_v_arm is not None:
            ddq_kin = (v_arm - prev_v_arm) / self.dt
            phi_ja  = float(np.sqrt(np.sum(ddq_kin**2))) / 100.0
        else:
            phi_ja  = 0.0
        phi_jtc = 0.0
        u_arm = u[:len(arm)]
        power = v_arm * u_arm
        phi_eng_groups = {}
        phi_tau_groups = {}
        for gname, gidx in ENG_GROUPS.items():
            valid = [i for i in gidx if i < len(arm)]
            phi_eng_groups[gname] = (
                np.sqrt(np.sum(power[valid] ** 2)) / 20.0 if valid else 0.0
            )
            phi_tau_groups[gname] = (
                np.sqrt(np.sum(u_arm[valid] ** 2)) / 50.0 if valid else 0.0
            )

        M_full = np.zeros((model.nv, model.nv))
        mujoco.mj_fullM(model, M_full, data.qM)
        phi_geo = np.sqrt(abs(data.qvel @ M_full @ data.qvel))

        f_contact = 0.0
        for ci in range(data.ncon):
            c = data.contact[ci]
            pair = {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]}
            if pair == {self.rock_body_id, self.stick_body_id}:
                if c.dist < 0:
                    fb = np.zeros(6)
                    mujoco.mj_contactForce(model, data, ci, fb)
                    f_contact = max(f_contact, abs(fb[0]))
        T_ps = getattr(self, '_T_press_start', 0)
        if t_abs is not None and T_ps > 0 and t_abs < T_ps:
            phi_pf = 0.0
        elif getattr(self, 'press_emergent', False):
            _fmax = float(getattr(self, 'force_max', 80.0))
            phi_pf = abs(f_contact) / max(_fmax, 1.0)
        else:
            tgt = float(target_force if target_force is not None else self.target_force)
            phi_pf = abs(f_contact - tgt) / max(tgt, 1.0)

        # Force capacity |f_n - f_max|/f_max — contact-gated. MUST MATCH the demo
        _fmax_c = float(getattr(self, 'force_max', 80.0))
        if t_abs is not None and T_ps > 0 and t_abs < T_ps:
            phi_cap = 0.0
        else:
            _fc = float(np.clip(f_contact, -2.0 * _fmax_c, 2.0 * _fmax_c))
            phi_cap = (_fc - _fmax_c) ** 2 / max(_fmax_c, 1.0) ** 2

        raw_dist = 0.0
        gap = 0.0
        if self.rock_geom_id >= 0 and self.stick_body_id >= 0:
            rp = data.geom_xpos[self.rock_geom_id]
            sc = data.xpos[self.stick_body_id]
            sa = data.xmat[self.stick_body_id].reshape(3, 3)[:, 2]
            tp = np.clip(np.dot(rp - sc, sa), -self.stick_half_len, self.stick_half_len)
            nearest = sc + tp * sa
            raw_dist = np.linalg.norm(rp - nearest)
            gap = raw_dist - self.stick_radius - 0.02
        phi_surf = max(0.0, gap) / self.stick_radius
        phi_approach = abs(gap) / self.stick_radius

        traveled = 0.0
        if self.rail_len > 1e-9 and self.contact_site_id >= 0:
            sp = data.site_xpos[self.contact_site_id]
            traveled = np.dot(sp - self.p_start, self.rail_dir)
        phi_progress = max(0.0, 1.0 - traveled / self.rail_len) if self.rail_len > 1e-9 else 0.0
        phi_traveled = -(traveled / self.rail_len) if self.rail_len > 1e-9 else 0.0

        rail_vel = (traveled - prev_traveled) / self.dt if self.dt > 0 else 0.0
        if getattr(self, 'progress_vel_sq', False):
            phi_prog_vel = rail_vel ** 2
        elif (_pv_cap := getattr(self, 'progress_vel_cap', None)) is not None:
            phi_prog_vel = -min(rail_vel, float(_pv_cap))
        else:
            phi_prog_vel = -rail_vel

        phi_rail_lat = 0.0
        if self.rail_len > 1e-9 and self.contact_site_id >= 0:
            sp = data.site_xpos[self.contact_site_id]
            diff = sp - self.p_start
            lat = diff - np.dot(diff, self.rail_dir) * self.rail_dir
            phi_rail_lat = np.sqrt(np.dot(lat, lat)) / self.stick_radius

        phi_rock_ori = 0.0
        if self.rock_body_id >= 0 and self.rock_quat_ref is not None:
            dot = np.dot(data.xquat[self.rock_body_id], self.rock_quat_ref)
            phi_rock_ori = 1.0 - dot**2

        # Order must match KEYS_RUN:
        phi = np.array([phi_tau, phi_jv, phi_ja, phi_jtc,
                        phi_eng_groups['thoracic'], phi_eng_groups['clavicle'],
                        phi_eng_groups['shoulder'], phi_eng_groups['elbow'],
                        phi_eng_groups['wrist'],
                        phi_geo, phi_pf, phi_surf,
                        phi_prog_vel, phi_rail_lat,
                        phi_rock_ori,
                        phi_approach, phi_traveled,
                        phi_tau_groups['thoracic'], phi_tau_groups['clavicle'],
                        phi_tau_groups['shoulder'], phi_tau_groups['elbow'],
                        phi_tau_groups['wrist'],
                        phi_cap,
                        0.0])
        return phi, traveled, v_arm

    def _rollout_one(self, k, model, qpos0, qvel0, U_k, stick_sched,
                     w_arr, phi_scale, w_fixed, target_sched=None,
                     t_offset=0):
        """Roll out one sample trajectory. Thread-safe (uses own MjData).

        w_arr: either (nr_run,) for uniform weights, or (H, nr_run) for TV weights.
        """
        d = mujoco.MjData(model)
        d.qpos[:] = qpos0
        d.qvel[:] = qvel0
        mujoco.mj_forward(model, d)

        tv = w_arr.ndim == 2

        prev_traveled = 0.0
        if self.rail_len > 1e-9 and self.contact_site_id >= 0:
            sp = d.site_xpos[self.contact_site_id]
            prev_traveled = np.dot(sp - self.p_start, self.rail_dir)
        prev_v_arm = d.qvel[self.arm_dofadr].copy()

        cost = 0.0
        for h in range(self.H):
            if self.has_stick and stick_sched is not None:
                adr = self.stick_qposadr
                d.qpos[adr:adr+3] = stick_sched[h]
                d.qvel[adr:adr+6] = 0.0

            tau_grav = d.qfrc_bias[self.arm_dofadr]
            u_h = np.clip(tau_grav + U_k[h], self.u_min, self.u_max)
            d.ctrl[:] = u_h

            for qi, di, lv in zip(self.locked_qpos_ids, self.locked_dof_ids, self.locked_vals):
                d.qpos[qi] = lv
                d.qvel[di] = 0.0

            mujoco.mj_step(model, d)

            tgt_h = (target_sched[h] if target_sched is not None else None)
            phi, prev_traveled, prev_v_arm = self._compute_phi_fast(
                model, d, u_h, prev_traveled, tgt_h, t_abs=t_offset + h,
                prev_v_arm=prev_v_arm)
            phi_norm = phi / phi_scale
            w_h = w_arr[h] if tv else w_arr
            cost += (np.dot(w_h, phi_norm) + np.dot(w_fixed, phi)) * self.dt

            q_jnt = d.qpos[self._jnt_qposadr]
            viol = np.where(self._jnt_limited,
                            np.maximum(0, self._jnt_range[:, 0] - q_jnt) +
                            np.maximum(0, q_jnt - self._jnt_range[:, 1]), 0.0)
            cost += 10000.0 * np.sum(viol**2)

        return cost

    def solve(self, human_mppi, T=None, visualize=False):
        """Receding-horizon solve using CPU MuJoCo. Deterministic."""
        import time

        model = human_mppi.mj_model
        data = human_mppi.mj_data
        T = T or human_mppi.T

        self._T_press_start = int(getattr(human_mppi, 'T_press_start', 0) or 0)

        self.reset()

        mujoco.mj_resetData(model, data)
        v0 = getattr(human_mppi, 'v0', np.zeros(model.nv))
        from final_models.human_mppi import ACTIVE_ARM_JOINTS as _AAJ
        for i, name in enumerate(_AAJ):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                data.qpos[model.jnt_qposadr[jid]] = human_mppi.q0[i]
                dof = model.jnt_dofadr[jid]
                if i < len(v0):
                    data.qvel[dof] = v0[i]
        for qpos_id, dof_id, lock_val in human_mppi.locked_joint_constraints:
            data.qpos[qpos_id] = lock_val
            data.qvel[dof_id] = 0.0
        if self.has_stick:
            adr = self.stick_qposadr
            data.qpos[adr:adr+3] = self.stick_traj[0]
            data.qvel[adr:adr+6] = 0.0
        mujoco.mj_forward(model, data)

        w_arr_uniform = np.array([self.w_run.get(k, 0.0) for k in KEYS_RUN])
        phi_scale = self._phi_scale
        w_fixed = self._w_fixed

        use_tv = self._w_run_windows is not None and len(self._w_run_windows) > 1
        if use_tv:
            n_w = len(self._w_run_windows)
            window_size = max(1, T // n_w)
            w_windows = np.array([
                [wd.get(k, 0.0) for k in KEYS_RUN]
                for wd in self._w_run_windows
            ])

        xs, us, force_log = [], [], []
        q, v = human_mppi._extract_pin_state()
        xs.append(np.concatenate([q, v]))

        t_total = 0.0

        for t in range(T):
            t0 = time.perf_counter()

            qpos0 = data.qpos.copy()
            qvel0 = data.qvel.copy()

            if use_tv:
                w_arr = np.zeros((self.H, self.nr_run))
                for h in range(self.H):
                    k_win = min((t + h) // window_size, n_w - 1)
                    w_arr[h] = w_windows[k_win]
            else:
                w_arr = w_arr_uniform

            stick_sched = None
            if self.has_stick:
                indices = np.clip(np.arange(self.H) + t, 0, len(self.stick_traj) - 1)
                stick_sched = self.stick_traj[indices]

            target_sched = None
            if self.target_force_profile is not None:
                idx = np.clip(np.arange(self.H) + t, 0,
                              len(self.target_force_profile) - 1)
                target_sched = self.target_force_profile[idx]

            for dial in range(self.n_dial):
                scale = self.dial_decay ** dial
                z = self._rng.randn(self.K, self.H, self.nu)
                eps = np.einsum('ij,khj->khi', self.L, z) * scale

                costs = np.zeros(self.K)
                for k in range(self.K):
                    costs[k] = self._rollout_one(k, model, qpos0, qvel0,
                                                 self.U + eps[k], stick_sched,
                                                 w_arr, phi_scale, w_fixed,
                                                 target_sched, t_offset=t)

                finite = np.isfinite(costs)
                if not np.any(finite):
                    continue
                costs_safe = np.where(finite, costs, np.max(costs[finite]))
                c_min = np.min(costs_safe)
                weights = np.exp(-(costs_safe - c_min) / self.lam)
                weights /= np.sum(weights) + 1e-10

                self.U += np.einsum('k,khn->hn', weights, eps)
                self.U = np.clip(self.U, self.u_min - 500, self.u_max + 500)

            mujoco.mj_forward(model, data)
            tau_grav = data.qfrc_bias[self.arm_dofadr]
            u = np.clip(tau_grav + self.U[0], self.u_min, self.u_max)
            data.ctrl[:] = u
            us.append(u.copy())

            if self.has_stick:
                sidx = min(t, len(self.stick_traj) - 1)
                adr = self.stick_qposadr
                data.qpos[adr:adr+3] = self.stick_traj[sidx]
                data.qvel[adr:adr+6] = 0.0

            for qi, di, lv in zip(self.locked_qpos_ids, self.locked_dof_ids, self.locked_vals):
                data.qpos[qi] = lv
                data.qvel[di] = 0.0

            mujoco.mj_step(model, data)

            f_step = 0.0
            for ci in range(data.ncon):
                c = data.contact[ci]
                pair = {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]}
                if pair == {self.rock_body_id, self.stick_body_id}:
                    fb = np.zeros(6)
                    mujoco.mj_contactForce(model, data, ci, fb)
                    f_step = max(f_step, abs(fb[0]))
            force_log.append(f_step)

            self.U = np.concatenate([self.U[1:], np.zeros((1, self.nu))], axis=0)

            dt_step = time.perf_counter() - t0
            t_total += dt_step

            if t % 10 == 0:
                p_site = data.site_xpos[self.contact_site_id] if self.contact_site_id >= 0 else np.zeros(3)
                progress_pct = np.dot(p_site - self.p_start, self.rail_dir) / self.rail_len * 100 if self.rail_len > 1e-6 else 0.0
                gap = 0.0
                if self.rock_geom_id >= 0 and self.stick_body_id >= 0:
                    rp = data.geom_xpos[self.rock_geom_id]
                    sc = data.xpos[self.stick_body_id]
                    sa = data.xmat[self.stick_body_id].reshape(3, 3)[:, 2]
                    tp = np.clip(np.dot(rp - sc, sa), -self.stick_half_len, self.stick_half_len)
                    nearest = sc + tp * sa
                    gap = np.linalg.norm(rp - nearest) - self.stick_radius - 0.02
                f_contact = 0.0
                for ci in range(data.ncon):
                    c = data.contact[ci]
                    pair = {model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]}
                    if pair == {self.rock_body_id, self.stick_body_id}:
                        fb = np.zeros(6)
                        mujoco.mj_contactForce(model, data, ci, fb)
                        f_contact = max(f_contact, abs(fb[0]))
                print(f"  [t={t:4d}]  progress={progress_pct:.1f}%  "
                      f"|u|={np.linalg.norm(u):.1f}  "
                      f"gap={gap:.4f}  force={f_contact:.1f}N  "
                      f"step={dt_step:.1f}s")

            q, v = human_mppi._extract_pin_state()
            xs.append(np.concatenate([q, v]))

        print(f"\n[KinematicMPPI_CPU.solve] done — {T} steps, {t_total:.1f}s total "
              f"({t_total/T:.2f}s/step)")
        self._last_xs = np.array(xs)
        self._last_us = np.array(us)
        self._last_forces = np.array(force_log)
        return self._last_xs, self._last_us

    def install_as_solver(self, human_mppi):
        """
        Patch human_mppi so MO_IRL uses this CPU MPPI.
        Same interface as KinematicMPPI.install_as_solver().
        """
        kin = self
        T = human_mppi.T

        class _ProblemShim:
            def __init__(self):
                self.T = T

        class _KinSolverShim:
            def __init__(self):
                self.problem = _ProblemShim()
                self.termination_tolerance = 1e-4
                self.with_callbacks = False
                self.xs = []
                self.us = []

            def solve(self, xs_init=None, us_init=None, maxiter=1000,
                      isFeasible=False, init_reg=None):
                xs, us = kin.solve(human_mppi, T=T, visualize=False)
                self.xs = list(xs)
                self.us = list(us)
                human_mppi._last_xs = xs
                human_mppi._last_us = us
                human_mppi._press_forces = kin._last_forces
                return True

            def reset(self):
                """Delegate to the underlying kin's reset. Required so
                MO_IRL._compute_dw's `solver.reset()` between gradient
                evaluations actually clears self.U → cold-start MPPI
                each IRL iter. See mirror in mppi_mjx_kinematic.py."""
                kin.reset()

        human_mppi.solver = _KinSolverShim()
        human_mppi._kin_mppi = kin

        _orig_update = human_mppi.update_solver_weights
        _orig_update_tv = human_mppi.update_solver_weights_tv

        def _patched_update(w_run_new, w_term_new):
            _orig_update(w_run_new, w_term_new)
            kin_w = {k: w_run_new.get(k, 0.0) for k in kin.KEYS_RUN
                     if k in w_run_new}
            if kin_w:
                kin.set_weights({**kin.w_run, **kin_w})

        def _patched_update_tv(w_run_windows, w_term_windows):
            _orig_update_tv(w_run_windows, w_term_windows)
            keys_run = human_mppi.keys_run
            n_w = len(w_run_windows)
            kin_windows = []
            for k in range(n_w):
                w_full = dict(zip(keys_run, w_run_windows[k]))
                kin_w = {key: w_full.get(key, 0.0) for key in kin.KEYS_RUN}
                kin_windows.append(kin_w)
            kin.set_tv_weights(kin_windows)

        human_mppi.update_solver_weights = _patched_update
        human_mppi.update_solver_weights_tv = _patched_update_tv

        print(f"[KinematicMPPI_CPU] Installed as solver on HumanMPPI "
              f"(features: {kin.KEYS_RUN})")

    def set_tv_weights(self, w_run_windows):
        """Store time-varying weight dicts (one per window)."""
        self._w_run_windows = list(w_run_windows) if w_run_windows else None
