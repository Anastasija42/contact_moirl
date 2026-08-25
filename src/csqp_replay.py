"""Replay a recovered cost.

Rolls the recovered weights out on each demonstration's own model and writes the
resulting trajectory next to the weights, so a fit can be compared against the
recording without re-running the IRL.
"""
import os

import numpy as np
import pinocchio as pin

def replay(args, models, pairs, keys, demos_xs, demos_us, nr):
    """Roll out `args.replay_npz` and save rollout_recovered.npz beside it."""
    dd = np.load(args.replay_npz, allow_pickle=True)
    w_full = np.asarray(dd["w_hat_full"], float)
    B_window = np.asarray(dd["B_window"], float)
    K = int(dd["K"])
    theta = w_full.reshape(K, nr)
    W_win = B_window @ theta
    for _kn, _sc in [('press_capacity', args.press_cap_scale),
                     ('press_force', args.press_force_scale)]:
        if abs(_sc - 1.0) > 1e-9 and _kn in keys:
            W_win[:, keys.index(_kn)] *= float(_sc)
            if _kn == 'press_capacity':
                print(f"[probe] scaled {_kn} x{_sc}  (push F* toward Fmax={args.force_max})")
            else:
                print(f"[probe] scaled {_kn} x{_sc}")
    cold = bool(args.replay_cold)
    roll_xs, roll_us, demo_gaps, roll_gaps, roll_forces = [], [], [], [], []
    roll_force_profiles, demo_force_targets = [], []
    demo_railprog, roll_railprog = [], []
    roll_phis = []
    init_xs = []
    def _railprog(states, mdl):
        import pinocchio as pin
        if not hasattr(mdl, "p_start_world") or not hasattr(mdl, "p_end_world"):
            return np.zeros(len(states))
        rail = np.asarray(mdl.p_end_world) - np.asarray(mdl.p_start_world)
        ru = rail / (np.linalg.norm(rail) + 1e-12)
        pd = mdl.pin_model.createData(); out = []
        for x in states:
            q = np.asarray(x)[:mdl.nq]
            pin.framesForwardKinematics(mdl.pin_model, pd, q)
            pin.updateFramePlacements(mdl.pin_model, pd)
            pos = pd.oMf[mdl.contact_frame_id].translation
            out.append(float((pos - np.asarray(mdl.p_start_world)) @ ru))
        return np.asarray(out)
    def _pernode_gap(sv):
        return np.array([float(np.linalg.norm(np.asarray(f))) for f in sv.fs])
    for i, m in enumerate(models):
        m.update_solver_weights_tv(W_win, None)
        dxs = [np.asarray(x).copy() for x in demos_xs[i]]
        dus = [np.asarray(u).copy() for u in demos_us[i]]
        m.solver.solve([x.copy() for x in dxs], [u.copy() for u in dus], 0)
        demo_gaps.append(_pernode_gap(m.solver))
        demo_gap_tot = float(m.solver.gap_norm)
        if getattr(args, 'warmstart_neutral', False):
            # must discover the reach+press from the cost -- no rail-follow, no press seed.
            _x0 = np.asarray(dxs[0], float).copy(); _x0[m.nq:] = 0.0
            xw = [_x0.copy() for _ in range(len(dxs))]
            uw = [np.zeros_like(np.asarray(dus[0], float)) for _ in range(len(dus))]
        elif cold:
            xw, uw = m._ik_warmstart()
        else:
            xw, uw = dxs, dus
        init_xs.append(np.stack([np.asarray(x, float) for x in xw]))
        _rec = (args.record_rollout is not None and i == int(args.record_rollout))
        if _rec:
            _rr, _ry = m.solver.reset_rho, m.solver.reset_y
            m.solver.reset_rho = False; m.solver.reset_y = False
            _xs_it, _kkt, _gapn, _fmean = [], [], [], []
            _cx = [np.asarray(x).copy() for x in xw]
            _cu = [np.asarray(u).copy() for u in uw]
            for _k in range(int(args.sqp_iter)):
                m.solver.solve(_cx, _cu, 1)
                _cx = [np.asarray(x).copy() for x in m.solver.xs]
                _cu = [np.asarray(u).copy() for u in m.solver.us]
                _xs_it.append(np.stack(_cx))
                _kkt.append(float(m.solver.KKT)); _gapn.append(float(m.solver.gap_norm))
                _fmean.append(float(np.linalg.norm(m.get_contact_forces(), axis=1).mean()))
            m.solver.reset_rho, m.solver.reset_y = _rr, _ry
            _hist = os.path.join(args.outdir, f"rollout_history_cyc{pairs[i][2]}.npz")
            os.makedirs(args.outdir, exist_ok=True)
            np.savez(_hist, xs_iters=np.asarray(_xs_it), kkt=np.asarray(_kkt),
                     gap=np.asarray(_gapn), fmean=np.asarray(_fmean),
                     demo_xs=np.stack(dxs), nq=int(m.nq))
            print(f"[record] cyc{pairs[i][2]}: {len(_xs_it)} iterates -> {_hist} "
                  f"(KKT {_kkt[0]:.1e}->{_kkt[-1]:.1e}, gap {_gapn[0]:.1e}->{_gapn[-1]:.1e})")
        elif args.challenger_continuation and 'press_capacity' in keys:
            _pcols = [keys.index(_k) for _k in ('press_capacity', 'press_force') if _k in keys]
            _xs = [np.asarray(x).copy() for x in xw]; _us = [np.asarray(u).copy() for u in uw]
            for _frac in [float(x) for x in str(args.cont_fracs).split(',') if x.strip()]:
                _Ws = np.asarray(W_win, float).copy()
                for _pc in _pcols: _Ws[:, _pc] *= _frac
                m.update_solver_weights_tv(_Ws, None)
                m.solver.solve([np.asarray(x).copy() for x in _xs],
                               [np.asarray(u).copy() for u in _us], args.sqp_iter)
                _xs = [np.asarray(x).copy() for x in m.solver.xs]
                _us = [np.asarray(u).copy() for u in m.solver.us]
        else:
            m.solver.solve([np.asarray(x).copy() for x in xw],
                           [np.asarray(u).copy() for u in uw], args.sqp_iter)
        s = m.solver
        roll_gaps.append(_pernode_gap(s))
        _fprof_node = np.linalg.norm(m.get_contact_forces(), axis=1)
        _froll = float(_fprof_node.mean())
        _tp = getattr(m, 'target_force_profile', None)
        _ftgt = float(np.mean(_tp)) if _tp is not None else 0.0
        roll_forces.append(_froll)
        roll_force_profiles.append(np.asarray(_fprof_node, float))
        demo_force_targets.append(np.asarray(_tp, float) if _tp is not None
                                  else np.zeros(len(_fprof_node)))
        print(f"[replay {'cold' if cold else 'warm'}] {pairs[i]}: "
              f"iter={s.iter} KKT={s.KKT:.2e} "
              f"demo_gap={demo_gap_tot:.2e} roll_gap={s.gap_norm:.2e} "
              f"| F_recovered={_froll:.1f}N  F_target={_ftgt:.1f}N")
        roll_xs.append(np.stack(m.solver.xs.copy()))
        roll_us.append(np.stack(m.solver.us.copy()))
        demo_railprog.append(_railprog(dxs, m))
        roll_railprog.append(_railprog(m.solver.xs, m))
        try:
            _rf = m.get_traj_features(list(m.solver.xs), list(m.solver.us))
            roll_phis.append(np.asarray(_rf[1], float))
        except Exception as _e:  # noqa: BLE001
            roll_phis.append(None); print(f"[replay] get_traj_features failed: {_e}")
    _deg = 180.0 / np.pi
    _nqj = int(getattr(m, "nq", 0))
    _pj = []
    for _rx, _dx in zip(roll_xs, demos_xs):
        _rx = np.asarray(_rx, float); _dx = np.asarray(_dx, float)
        _L = min(len(_rx), len(_dx)); _q = min(_nqj, _rx.shape[1], _dx.shape[1])
        if _L > 0 and _q > 0:
            _pj.append(np.sqrt(((( _rx[:_L, :_q] - _dx[:_L, :_q]) * _deg) ** 2).mean(axis=0)))
    _qrj_test = np.mean(np.stack(_pj), axis=0) if _pj else np.zeros(0)
    try:
        _jn = list(m.pin_model.names)[1:]
    except Exception:
        _jn = []
    print("\nper-joint HELD-OUT RMSE (deg):")
    for _n, _e in zip(_jn, _qrj_test):
        print(f"   {_n:24s}{float(_e):7.2f}")
    out = os.path.join(os.path.dirname(args.replay_npz), "rollout_recovered.npz")
    np.savez(out,
             q_rmse_perjoint_test=_qrj_test, joint_names=np.array(_jn),
             rollout_xs=np.asarray(roll_xs), rollout_us=np.asarray(roll_us),
             demos_xs=np.asarray(demos_xs), demos_us=np.asarray(demos_us),
             demo_gaps=np.asarray(demo_gaps), roll_gaps=np.asarray(roll_gaps),
             roll_forces=np.asarray(roll_forces),
             roll_force_profiles=np.asarray(roll_force_profiles),
             demo_force_targets=np.asarray(demo_force_targets),
             demo_railprog=np.asarray(demo_railprog),
             roll_railprog=np.asarray(roll_railprog),
             init_xs=np.asarray(init_xs),
             roll_phis=np.asarray(roll_phis, dtype=object),
             keys_run=np.asarray(list(getattr(m, 'keys_run', []))),
             keys=np.asarray(keys), pair_cycles=np.asarray([c for _, _, c in pairs]))
    print(f"[replay] saved -> {out}")
