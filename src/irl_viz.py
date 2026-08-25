"""
IRL Visualization utilities.

Usage:
    from irl_viz import plot_weight_evolution, plot_feature_signal, plot_gradient_heatmap, plot_convergence
    plot_weight_evolution(IRL)
    plot_feature_signal(IRL)
    plot_gradient_heatmap(IRL)
    plot_convergence(IRL)

Contact force visualisation
---------------------------
    from irl_viz import plot_contact_forces

    # Get forces from Crocoddyl (after Human.solve()):
    f_croc_vecs = human.get_contact_forces()          # list of (3,) vectors
    f_croc = np.array([f[2] for f in f_croc_vecs])   # z = normal force

    # Get forces from MuJoCo (xs/us from MPPI or deterministic rollout):
    f_mj = model_ocp_mppi._simulate_press_forces(xs_mppi, us_mppi)

    plot_contact_forces(f_croc=f_croc, f_mujoco=f_mj,
                        dt=args['dt'], target_force=args['target_force'])
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def _ws_array(IRL):
    """Return weight history as (n_iter, nr) array from IRL.ws.

    IRL.ws entries can be:
      - np.ndarray  : flat weight vector [w_run | w_term]
      - 3-tuple     : (wv_run, wv_term, wv_combined) from dict_to_vector()
      - 2-tuple     : (w_run_dict, w_term_dict)
    """
    rows = []
    for entry in IRL.ws:
        if isinstance(entry, np.ndarray):
            rows.append(entry.copy())
        elif isinstance(entry, (tuple, list)):
            if len(entry) == 3:
                rows.append(np.array(entry[2]).copy())
            elif len(entry) == 2:
                w_run, w_term = entry
                run_vals  = [w_run[k]  for k in sorted(w_run.keys())]
                term_vals = [w_term[k] for k in sorted(w_term.keys())]
                rows.append(np.array(run_vals + term_vals))
        else:
            rows.append(np.atleast_1d(np.array(entry)).copy())
    return np.array(rows)

def _feature_labels(IRL):
    keys_run  = IRL.keys_run
    keys_term = IRL.keys_term
    n_w       = getattr(IRL, 'n_w', 1)
    labels = []
    for w in range(n_w):
        suffix = '' if n_w == 1 else f'_w{w}'
        for k in keys_run:
            labels.append(f'{k}{suffix}')
    for w in range(n_w):
        suffix = '' if n_w == 1 else f'_w{w}'
        for k in keys_term:
            labels.append(f'term_{k}{suffix}')
    return labels

def plot_weight_evolution(IRL, figsize=None):
    """
    Plot how each weight changes over IRL iterations.
    Run-cost weights on the left panel, terminal-cost weights on the right.
    """
    ws = _ws_array(IRL)
    n_iter = ws.shape[0]
    iters  = np.arange(n_iter)

    keys_run  = IRL.keys_run
    keys_term = IRL.keys_term
    n_w       = getattr(IRL, 'n_w', 1)
    nr_run    = IRL.nr_run
    nr_term   = IRL.nr_term

    nr_run_tv  = nr_run  * n_w
    nr_term_tv = nr_term * n_w

    run_ws  = ws[:, :nr_run_tv]
    term_ws = ws[:, nr_run_tv:]

    fig, axes = plt.subplots(1, 2, figsize=figsize or (14, 5))

    ax = axes[0]
    for i, k in enumerate(keys_run):
        for w in range(n_w):
            idx   = w * nr_run + i
            label = k if n_w == 1 else f'{k}_win{w}'
            ax.plot(iters, run_ws[:, idx], label=label)
    ax.set_title('Run-cost weight evolution')
    ax.set_xlabel('IRL iteration')
    ax.set_ylabel('Weight value')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i, k in enumerate(keys_term):
        for w in range(n_w):
            idx   = w * nr_term + i
            label = f'term_{k}' if n_w == 1 else f'term_{k}_win{w}'
            ax.plot(iters, term_ws[:, idx], label=label)
    ax.set_title('Terminal-cost weight evolution')
    ax.set_xlabel('IRL iteration')
    ax.set_ylabel('Weight value')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle('IRL Weight Evolution', fontsize=12)
    plt.tight_layout()
    return fig

def plot_feature_signal(IRL, figsize=None):
    """
    Plot the phi_diff = phi_demo - E_w[phi] per feature across IRL iterations.
    Stored in IRL.phi_diffs_history as list of (nr_run+nr_term,) arrays.
    """
    if not hasattr(IRL, 'phi_diffs_history') or len(IRL.phi_diffs_history) == 0:
        print('[irl_viz] phi_diffs_history is empty — run IRL.solve() first.')
        return None

    diffs = np.array(IRL.phi_diffs_history)
    n_iter, nr = diffs.shape
    iters = np.arange(1, n_iter + 1)

    keys_run  = IRL.keys_run
    keys_term = IRL.keys_term
    n_w       = getattr(IRL, 'n_w', 1)
    nr_run    = IRL.nr_run

    labels = _feature_labels(IRL)

    fig, axes = plt.subplots(1, 2, figsize=figsize or (14, 5))

    ax = axes[0]
    for i in range(nr_run):
        ax.plot(iters, diffs[:, i], label=labels[i])
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_title('Run-feature differences (phi_demo − E_w[phi])')
    ax.set_xlabel('IRL iteration')
    ax.set_ylabel('phi_diff')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for i in range(nr_run, nr):
        ax.plot(iters, diffs[:, i], label=labels[i])
    ax.axhline(0, color='k', linewidth=0.8, linestyle='--')
    ax.set_title('Terminal-feature differences (phi_demo − E_w[phi])')
    ax.set_xlabel('IRL iteration')
    ax.set_ylabel('phi_diff')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Feature Signal per IRL Iteration', fontsize=12)
    plt.tight_layout()
    return fig

def plot_gradient_heatmap(model, figsize=None):
    """
    Visualise the per-timestep gradient from model._last_gradient_info
    (populated after model.get_irl_gradient() is called).

    model : HumanMPPI instance
    """
    if not hasattr(model, '_last_gradient_info'):
        print('[irl_viz] _last_gradient_info not found — call model.get_irl_gradient() first.')
        return None

    info = model._last_gradient_info
    diff = info['diff_per_t']
    phi_demo = info['phi_demo_per_t']
    E_phi    = info['E_phi_per_t']
    keys     = info['phi_keys']
    T, nr    = diff.shape

    fig, axes = plt.subplots(3, 1, figsize=figsize or (14, 10))

    def _heatmap(ax, data, title, cmap='RdBu'):
        im = ax.imshow(data.T, aspect='auto', cmap=cmap,
                       origin='lower', interpolation='nearest')
        ax.set_yticks(range(nr))
        ax.set_yticklabels(keys, fontsize=7)
        ax.set_xlabel('Timestep')
        ax.set_title(title)
        plt.colorbar(im, ax=ax, fraction=0.02)

    _heatmap(axes[0], phi_demo, 'phi_demo_H  (demo feature window)')
    _heatmap(axes[1], E_phi,    'E_w[phi_H]  (MPPI expected feature)')
    _heatmap(axes[2], diff,     'Gradient signal: phi_demo − E_w[phi]')

    plt.tight_layout()
    return fig

def plot_convergence(IRL, figsize=None):
    """
    Multi-panel convergence plot: cost_diff, opt_div, state_diff, Js norm.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize or (12, 8))

    def _safe_plot(ax, data, title, ylabel):
        if data is None or len(data) == 0:
            ax.set_title(title + '  (no data)')
            return
        ax.plot(data)
        ax.set_title(title)
        ax.set_xlabel('IRL iteration')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)

    cost_diffs  = getattr(IRL, 'cost_diffs',  None)
    opt_div     = getattr(IRL, 'opt_div',      None)
    state_diffs = getattr(IRL, 'state_diffs',  None)
    Js          = getattr(IRL, 'Js',           None)

    _safe_plot(axes[0, 0], cost_diffs,  'Cost Diff (nopt vs opt)', 'cost_diff')
    _safe_plot(axes[0, 1], opt_div,     'Optimality Divergence',   'opt_div')
    _safe_plot(axes[1, 0], state_diffs, 'State Diff',              'state_diff')

    ax = axes[1, 1]
    if Js and len(Js) > 0:
        jac_norms = [float(np.linalg.norm(np.atleast_1d(j))) for j in Js]
        ax.plot(jac_norms)
        ax.set_title('Jacobian norm (gradient magnitude)')
        ax.set_xlabel('IRL iteration')
        ax.set_ylabel('||J||')
        ax.grid(True, alpha=0.3)
    else:
        ax.set_title('Jacobian norm  (no data)')

    fig.suptitle('IRL Convergence', fontsize=12)
    plt.tight_layout()
    return fig

def plot_all(IRL, model=None):
    """Plot all four diagnostics. Pass model=HumanMPPI if gradient heatmap is desired."""
    figs = {}
    figs['weights']     = plot_weight_evolution(IRL)
    figs['features']    = plot_feature_signal(IRL)
    figs['convergence'] = plot_convergence(IRL)
    if model is not None:
        figs['gradient'] = plot_gradient_heatmap(model)
    plt.show()
    return figs

def plot_contact_forces(f_croc=None, f_mujoco=None, f_pd=None,
                        dt=0.01, target_force=None,
                        label_croc='Crocoddyl (OCP)',
                        label_mujoco='MuJoCo (MPPI)',
                        label_pd='MuJoCo (PD warm-start)',
                        figsize=None):
    """
    Plot contact normal force over time from Crocoddyl and/or MuJoCo.

    Parameters
    ----------
    f_croc    : (T,) array or list of (3,) vectors — Crocoddyl contact forces.
                If list of 3-vectors, the z-component is taken as the normal.
                Obtain with:
                    vecs = human.get_contact_forces()
                    f_croc = np.array([f[2] for f in vecs])
    f_mujoco  : (T,) array — MuJoCo scalar contact forces after MPPI.
                Obtain with:
                    f_mujoco = model_ocp_mppi._simulate_press_forces(xs, us)
    f_pd      : (T,) array — MuJoCo forces from the PD warm-start rollout.
                Obtain with:
                    f_pd = model_ocp_mppi._simulate_press_forces(xs_mj, U_pd)
                where xs_mj and U_pd come from set_warmstart_trajectory().
    dt           : timestep in seconds (for x-axis in seconds)
    target_force : scalar — draw a horizontal reference line if provided
    """
    import numpy as np

    fig, ax = plt.subplots(figsize=figsize or (11, 4))

    def _plot(forces, label, ls='-', lw=2):
        f = np.asarray(forces)
        if f.ndim == 2 and f.shape[1] == 3:
            f = f[:, 2]
        t = np.arange(len(f)) * dt
        ax.plot(t, f, label=label, ls=ls, lw=lw)

    if f_croc   is not None: _plot(f_croc,   label_croc,   ls='-',  lw=2.0)
    if f_pd     is not None: _plot(f_pd,     label_pd,     ls='--', lw=1.5)
    if f_mujoco is not None: _plot(f_mujoco, label_mujoco, ls=':',  lw=2.0)

    if target_force is not None:
        ax.axhline(target_force, color='k', ls='-.', lw=1.2,
                   label=f'target {target_force:.0f} N')

    ax.set_xlabel('time (s)')
    ax.set_ylabel('contact normal force (N)')
    ax.set_title('Contact force: Crocoddyl vs MuJoCo')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def _pick_iters_to_show(n_total, max_shown=6):
    """Choose iteration indices to overlay. Always include first IRL iter and
    final iter; sample evenly in between up to max_shown."""
    if n_total <= max_shown:
        return list(range(n_total))
    idx = np.linspace(0, n_total - 1, max_shown).round().astype(int)
    return sorted(set(idx.tolist()))

def _resolve_pin(model, contact_frame=None):
    """Accept either a wrapper (with .pin_model/.pin_data/.contact_frame_id/.nq)
    or a raw pinocchio.Model. Returns (pin_model, pin_data, frame_id, nq).

    `contact_frame` (str name or int id) is required only when `model` is a
    raw Pinocchio Model and we have no other way to locate the contact frame.
    """
    import pinocchio as pin
    if hasattr(model, 'pin_model'):
        pin_model = model.pin_model
        pin_data  = getattr(model, 'pin_data', None) or pin_model.createData()
        nq        = getattr(model, 'nq', pin_model.nq)
        if contact_frame is None:
            frame_id = getattr(model, 'contact_frame_id', None)
        else:
            frame_id = (contact_frame
                        if isinstance(contact_frame, (int, np.integer))
                        else pin_model.getFrameId(contact_frame))
    else:
        pin_model = model
        pin_data  = pin_model.createData()
        nq        = pin_model.nq
        if contact_frame is None:
            for cand in ('rock_contact_point', 'rock_frame',
                         'tool_fixed_joint'):
                try:
                    fid = pin_model.getFrameId(cand)
                    if fid < pin_model.nframes:
                        frame_id = fid
                        break
                except Exception:
                    pass
            else:
                frame_id = pin_model.nframes - 1
        else:
            frame_id = (contact_frame
                        if isinstance(contact_frame, (int, np.integer))
                        else pin_model.getFrameId(contact_frame))
    if frame_id is None:
        raise ValueError("plot_trajectory_morph: no contact frame found. "
                         "Pass contact_frame='<frame_name>' explicitly.")
    return pin_model, pin_data, int(frame_id), int(nq)

def _ee_path(pin_model, pin_data, frame_id, nq, xs):
    """Forward-kinematic frame world translation for each step. (T,3) out."""
    import pinocchio as pin
    xs = np.asarray(xs)
    T  = len(xs)
    out = np.zeros((T, 3))
    for t in range(T):
        q = xs[t, :nq]
        pin.forwardKinematics(pin_model, pin_data, q)
        pin.updateFramePlacements(pin_model, pin_data)
        out[t] = pin_data.oMf[frame_id].translation
    return out

def plot_trajectory_morph(IRL, model, iters_to_show=None, max_shown=6,
                           joints_per_row=4, contact_frame=None,
                           figsize=None, save_path=None):
    """Visualise the optimisation by morphing the trajectory across IRL iters.

    Two panels (side-by-side or stacked):
      - LEFT: 3D end-effector path. Demo as a thick black line; selected
        IRL iterations as a colour-graded path (early → bright, late → dark).
      - RIGHT: per-joint q(t) grid. Same colour code; demo overlaid in black.

    Parameters
    ----------
    IRL          : MO_IRL instance after solve(). Reads IRL.Xs (list of (T,nx)).
                   Convention: Xs[0] = demo; Xs[1] = init/bad; Xs[2:] = iters.
    model        : Either an OCP wrapper exposing
                   `.pin_model / .pin_data / .contact_frame_id / .nq`,
                   OR a raw `pinocchio.Model` (in which case pass
                   `contact_frame=<frame_name_or_id>` if the default
                   auto-detection misses the contact frame).
    iters_to_show: explicit list of indices into IRL.Xs[1:] to overlay. If
                   None, auto-picks `max_shown` indices spanning the IRL iters.
    max_shown    : max number of IRL iters to plot (default 6).
    joints_per_row: subplot columns in the joint grid.
    contact_frame: name (str) or id (int) of the frame to trace in 3D. Only
                   needed when `model` is a raw pinocchio.Model and the
                   auto-detection fallback misses the frame.
    save_path    : if set, save the figure here.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D proj
    _ = Axes3D
    pin_model, pin_data, frame_id, nq = _resolve_pin(model, contact_frame)

    Xs = list(IRL.Xs)
    if len(Xs) < 2:
        print('[irl_viz] plot_trajectory_morph: need at least demo + 1 iter')
        return None
    xs_demo = np.asarray(Xs[0])
    iter_xs = Xs[1:]
    if iters_to_show is None:
        iters_to_show = _pick_iters_to_show(len(iter_xs), max_shown=max_shown)
    selected = [iter_xs[i] for i in iters_to_show]
    cmap = plt.get_cmap('viridis')
    colors = [cmap(t) for t in np.linspace(0.15, 0.95, len(selected))]

    deg = 180.0 / np.pi
    joint_names = [pin_model.names[i + 1]
                   for i in range(min(nq, len(pin_model.names) - 1))]

    n_rows_grid = int(np.ceil(nq / joints_per_row))
    fig = plt.figure(figsize=figsize or (5 + 3 * joints_per_row,
                                           max(4, 2.2 * n_rows_grid)))
    gs = gridspec.GridSpec(n_rows_grid, joints_per_row + 2, figure=fig,
                            width_ratios=[2] * 2 + [1] * joints_per_row)

    ax3d = fig.add_subplot(gs[:, :2], projection='3d')
    demo_ee = _ee_path(pin_model, pin_data, frame_id, nq, xs_demo)
    ax3d.plot(demo_ee[:, 0], demo_ee[:, 1], demo_ee[:, 2],
              color='k', lw=2.5, label='demo')
    ax3d.scatter(*demo_ee[0], color='k', marker='o', s=40)
    ax3d.scatter(*demo_ee[-1], color='k', marker='X', s=60)
    for col, idx, xs in zip(colors, iters_to_show, selected):
        ee = _ee_path(pin_model, pin_data, frame_id, nq, xs)
        ax3d.plot(ee[:, 0], ee[:, 1], ee[:, 2], color=col, lw=1.4,
                  alpha=0.9, label=f'iter {idx}')
    ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('z')
    ax3d.set_title('End-effector path (○ start, ✕ end)')
    ax3d.legend(loc='upper left', fontsize=8)

    for j in range(nq):
        r = j // joints_per_row
        c = 2 + (j % joints_per_row)
        ax = fig.add_subplot(gs[r, c])
        t = np.arange(len(xs_demo))
        ax.plot(t, xs_demo[:, j] * deg, color='k', lw=2.0, label='demo')
        for col, idx, xs in zip(colors, iters_to_show, selected):
            xs = np.asarray(xs)
            t_x = np.arange(len(xs))
            ax.plot(t_x, xs[:, j] * deg, color=col, lw=1.2, alpha=0.85)
        name = joint_names[j] if j < len(joint_names) else f'q[{j}]'
        ax.set_title(name, fontsize=9)
        ax.tick_params(labelsize=8)
        ax.grid(True, alpha=0.3)
        if j == 0:
            ax.legend(fontsize=7)

    fig.suptitle('Trajectory morph: demo (black) vs IRL iters (viridis)',
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path is not None:
        fig.savefig(save_path, dpi=120)
    return fig

def plot_weight_heatmap(IRL, log_scale=True, sort_by_final=True,
                         per_window=True, figsize=None, save_path=None):
    """Heatmap of |w_k| across IRL iterations. Rows = iters, cols = features.

    For time-varying weights (n_w > 1 with windowed mode, or K basis
    coefficients in basis mode), `per_window=True` renders one heatmap row
    of subplots, one per window/basis-index. `per_window=False` aggregates
    via max(|w|) across windows so a single heatmap suffices.

    log_scale     : colour by log10(|w| + eps). Useful when weights span
                    many orders of magnitude (typical for IRL).
    sort_by_final : sort feature columns so the highest-magnitude features
                    at the final iter appear on the left.
    """
    ws = _ws_array(IRL)
    if ws.size == 0:
        print('[irl_viz] plot_weight_heatmap: IRL.ws is empty')
        return None
    n_iter = ws.shape[0]

    n_w     = getattr(IRL, 'n_w', 1)
    K       = int(getattr(IRL, 'K', n_w))
    nblocks = K if K > 1 else n_w
    nr_run  = IRL.nr_run
    nr_term = IRL.nr_term
    keys    = list(IRL.keys_run) + [f'term_{k}' for k in IRL.keys_term]
    nr      = nr_run + nr_term

    expected = nblocks * nr_run + nblocks * nr_term
    if ws.shape[1] != expected:
        nblocks = 1
        expected = nr_run + nr_term

    cube = np.zeros((n_iter, nblocks, nr))
    if nblocks > 1:
        run_part  = ws[:, :nblocks * nr_run].reshape(n_iter, nblocks, nr_run)
        term_part = (ws[:, nblocks * nr_run:].reshape(n_iter, nblocks, nr_term)
                     if nr_term > 0
                     else np.zeros((n_iter, nblocks, 0)))
        cube = np.concatenate([run_part, term_part], axis=2)
    else:
        cube[:, 0, :] = ws[:, :nr]

    if per_window and nblocks > 1:
        panels = [(cube[:, k, :], f'window {k}') for k in range(nblocks)]
    else:
        agg = np.max(np.abs(cube), axis=1)
        title = ('max(|w|) over windows' if nblocks > 1
                 else 'weights (single window)')
        panels = [(agg, title)]

    if sort_by_final:
        final = np.max(np.abs(cube[-1]), axis=0)
        col_order = np.argsort(-final)
    else:
        col_order = np.arange(nr)
    ordered_keys = [keys[i] for i in col_order]

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels,
                              figsize=figsize or (max(7, 1.0 * nr + 1.5),
                                                   1.5 + 0.35 * n_iter
                                                   if n_panels == 1
                                                   else 2 + 0.3 * n_iter),
                              squeeze=False)
    axes = axes[0]

    all_vals = np.concatenate([np.abs(p[0]).ravel() for p in panels])
    eps = max(1e-12, float(np.nanmin(all_vals[all_vals > 0]))) if np.any(all_vals > 0) else 1e-12
    if log_scale:
        vmax = float(np.log10(max(np.nanmax(all_vals), eps)))
        vmin = float(np.log10(eps))
    else:
        vmax = float(np.nanmax(all_vals))
        vmin = 0.0

    for ax, (data, ptitle) in zip(axes, panels):
        m = np.abs(data)[:, col_order]
        if log_scale:
            m = np.log10(m + eps)
        im = ax.imshow(m, aspect='auto', cmap='viridis',
                       origin='lower', interpolation='nearest',
                       vmin=vmin, vmax=vmax)
        ax.set_yticks(range(n_iter))
        ax.set_yticklabels([f'i{i}' for i in range(n_iter)], fontsize=7)
        ax.set_xticks(range(len(ordered_keys)))
        ax.set_xticklabels(ordered_keys, rotation=60, ha='right', fontsize=8)
        ax.set_title(ptitle, fontsize=10)
        ax.set_xlabel('feature'); ax.set_ylabel('IRL iter')

    cbar = fig.colorbar(im, ax=axes.tolist(), fraction=0.03, pad=0.02)
    cbar.set_label('log10(|w| + eps)' if log_scale else '|w|', fontsize=9)

    fig.suptitle('Weight evolution heatmap '
                 f'(iters={n_iter}, features={nr}, windows={nblocks})',
                 fontsize=12)
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig

def plot_weight_evolution_lines(IRL, top_n=10, per_window=False,
                                  log_scale=True, normalize=False,
                                  figsize=None, save_path=None):
    """Per-feature weight trajectory across IRL iters as line plots.

    Top-N features (by max |w| over iters) are highlighted in colour with
    labels; the rest are drawn faintly in grey. Much easier to read than a
    heatmap when there are many features × windows × iters.

    Parameters
    ----------
    top_n        : highlight this many features with colour + legend.
    per_window   : if True, draw one subplot per basis/window (so you can see
                   which window each feature is learning to care about). If
                   False, aggregate via max(|w|) across windows into one plot.
    log_scale    : log y-axis (recommended — IRL weights span many decades).
    normalize    : if True, divide each feature's trajectory by its max so
                   all features fit on [0, 1]. Reveals SHAPE of change at the
                   cost of losing absolute magnitude.
    """
    ws = _ws_array(IRL)
    if ws.size == 0:
        print('[irl_viz] plot_weight_evolution_lines: IRL.ws is empty')
        return None
    n_iter = ws.shape[0]
    iters  = np.arange(n_iter)

    n_w     = getattr(IRL, 'n_w', 1)
    K       = int(getattr(IRL, 'K', n_w))
    nblocks = K if K > 1 else n_w
    nr_run  = IRL.nr_run
    nr_term = IRL.nr_term
    keys    = list(IRL.keys_run) + [f'term_{k}' for k in IRL.keys_term]
    nr      = nr_run + nr_term

    expected = nblocks * nr_run + nblocks * nr_term
    if ws.shape[1] != expected:
        nblocks = 1
        expected = nr_run + nr_term

    cube = np.zeros((n_iter, nblocks, nr))
    if nblocks > 1:
        run_part  = ws[:, :nblocks * nr_run].reshape(n_iter, nblocks, nr_run)
        term_part = (ws[:, nblocks * nr_run:].reshape(n_iter, nblocks, nr_term)
                     if nr_term > 0
                     else np.zeros((n_iter, nblocks, 0)))
        cube = np.concatenate([run_part, term_part], axis=2)
    else:
        cube[:, 0, :] = ws[:, :nr]

    if per_window and nblocks > 1:
        panels = [(np.abs(cube[:, k, :]), f'window {k}') for k in range(nblocks)]
    else:
        panels = [(np.max(np.abs(cube), axis=1),
                   'max(|w|) over windows' if nblocks > 1 else 'weights')]

    pooled = np.max(np.stack([p[0] for p in panels], axis=0), axis=(0, 1))
    top_idx = list(np.argsort(-pooled)[:min(top_n, nr)])
    top_set = set(top_idx)
    cmap_top = plt.get_cmap('tab20')

    n_panels = len(panels)
    fig, axes = plt.subplots(1, n_panels, sharey=True,
                              figsize=figsize or (max(8, 6 * n_panels), 5),
                              squeeze=False)
    axes = axes[0]

    for ax, (data, ptitle) in zip(axes, panels):
        d = np.maximum(np.abs(data), 1e-12)
        if normalize:
            d = d / d.max(axis=0, keepdims=True).clip(min=1e-12)
        for k in range(nr):
            if k in top_set:
                continue
            ax.plot(iters, d[:, k], color='lightgray', lw=0.8, alpha=0.5)
        for rank, k in enumerate(top_idx):
            ax.plot(iters, d[:, k], color=cmap_top(rank % 20),
                    lw=2.0, label=keys[k])
        if log_scale and not normalize:
            ax.set_yscale('log')
        ax.set_xlabel('IRL iteration')
        ax.set_ylabel(('|w| / max(|w|)' if normalize else '|w|')
                      + (' (log)' if log_scale and not normalize else ''))
        ax.set_title(ptitle, fontsize=10)
        ax.grid(True, alpha=0.3, which='both')
    axes[-1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5),
                     fontsize=8, frameon=False, title=f'top {len(top_idx)}')

    fig.suptitle(f'Weight trajectory — {len(top_idx)} most-active features '
                 f'(of {nr})', fontsize=12)
    fig.tight_layout(rect=(0, 0, 0.97, 0.96))
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches='tight')
    return fig

def plot_full_cost_decomposition(IRL_obj, model, keys_subset=None,
                                   max_rows=None, best_idx=None,
                                   figsize=None, save_path=None):
    """Three columns per feature: weight decomposition, feature trace,
    cost contribution. Works for windowed and basis modes.

    Parameters
    ----------
    IRL_obj : MO_IRL with .ws, .phis_set, .keys_run, .q_norm, .K / .n_w
    model   : OCP model exposing .T and .dt
    keys_subset : list[str] of feature names to plot (default: all keys_run)
    max_rows    : cap on number of rows (after keys_subset)
    best_idx    : index into IRL_obj.ws / IRL_obj.q_norm to read final weights
                  from. Defaults to argmin(q_norm).
    """
    keys_run = list(IRL_obj.keys_run)
    nr_r     = len(keys_run)
    T        = int(model.T)
    dt       = float(model.dt)
    t_axis   = np.arange(T) * dt

    is_basis = (getattr(IRL_obj, 'weight_mode', 'windowed') == 'basis')
    n_blocks = int(getattr(IRL_obj, 'K', None)
                   if is_basis else getattr(IRL_obj, 'n_w', 1))
    if n_blocks is None or n_blocks < 1:
        n_blocks = 1

    if best_idx is None:
        best_idx = (int(np.argmin(IRL_obj.q_norm))
                    if len(IRL_obj.q_norm) > 0 else -1)

    w_flat = np.asarray(IRL_obj.ws[best_idx][0])
    w_tv   = w_flat[:n_blocks * nr_r].reshape(n_blocks, nr_r)

    if is_basis and hasattr(IRL_obj, 'B_step') and IRL_obj.B_step is not None:
        B      = np.asarray(IRL_obj.B_step)
        W_step = (B @ w_tv)[:T]
    else:
        ws_size = max(1, T // n_blocks)
        W_step  = np.zeros((T, nr_r))
        for i in range(T):
            W_step[i] = w_tv[min(i // ws_size, n_blocks - 1)]
        B = None

    phi_demo = np.array([np.asarray(p)[:nr_r] for p in IRL_obj.phis_set[0][:T]])
    phi_irl  = np.array([np.asarray(p)[:nr_r] for p in IRL_obj.phis_set[-1][:T]])

    if keys_subset is None:
        keys_subset = keys_run
    keys_subset = [k for k in keys_subset if k in keys_run]
    if max_rows is not None:
        keys_subset = keys_subset[:max_rows]

    n_rows = len(keys_subset)
    if n_rows == 0:
        print('[irl_viz] plot_full_cost_decomposition: no matching keys')
        return None

    fig, axes = plt.subplots(n_rows, 3,
                              figsize=figsize or (14, 1.6 * n_rows + 1),
                              sharex=True, squeeze=False)
    cmap = plt.cm.viridis
    block_colors = [cmap(k / max(n_blocks - 1, 1)) for k in range(n_blocks)]

    for row, feat in enumerate(keys_subset):
        j = keys_run.index(feat)

        ax = axes[row, 0]
        if is_basis and B is not None:
            stack = (B[:T] * w_tv[:, j]).T
            ax.stackplot(t_axis, stack, colors=block_colors, alpha=0.8,
                          labels=[f"k={k}: θ={w_tv[k, j]:.2e}"
                                  for k in range(n_blocks)])
            ax.plot(t_axis, W_step[:, j], 'k-', lw=1.5)
        else:
            for kw in range(n_blocks):
                t0 = (kw * T // n_blocks) * dt
                t1 = (((kw + 1) * T // n_blocks) * dt
                      if kw < n_blocks - 1 else T * dt)
                ax.axvspan(t0, t1, color=block_colors[kw], alpha=0.2,
                            label=f"k={kw}: w={w_tv[kw, j]:.2e}")
            ax.plot(t_axis, W_step[:, j], 'k-', lw=2)
        ax.set_ylabel(f"{feat}\nW(t)", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=6, framealpha=0.5)
        if row == 0:
            ax.set_title("Weight decomposition", fontsize=10)

        ax = axes[row, 1]
        ax.plot(t_axis, phi_demo[:, j], 'C0-', lw=1.4, label='demo')
        ax.plot(t_axis, phi_irl[:, j],  'C3-', lw=1.4, label='IRL')
        ax.set_ylabel("φ(t)", fontsize=8)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title("Feature trace  (demo vs IRL)", fontsize=10)
            ax.legend(fontsize=7)

        ax = axes[row, 2]
        ax.plot(t_axis, W_step[:, j] * phi_demo[:, j], 'C0-', lw=1.4,
                label='demo W·φ')
        ax.plot(t_axis, W_step[:, j] * phi_irl[:, j],  'C3-', lw=1.4,
                label='IRL  W·φ')
        ax.set_ylabel("W·φ", fontsize=8)
        ax.grid(True, alpha=0.3)
        if row == 0:
            ax.set_title("Cost contribution  (W·φ)", fontsize=10)
            ax.legend(fontsize=7)

    for c in range(3):
        axes[-1, c].set_xlabel('time (s)')

    q_norm_str = (f"  q_norm={IRL_obj.q_norm[best_idx]:.3f}"
                   if len(IRL_obj.q_norm) > 0 else "")
    fig.suptitle(f"Cost decomposition (iter {best_idx}){q_norm_str}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    if save_path is not None:
        fig.savefig(save_path, dpi=130, bbox_inches='tight')
    return fig