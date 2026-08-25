"""Command-line interface for the CSQP population MO-IRL driver.

Kept apart from the driver so that `main()` reads as the pipeline it runs rather
than as several hundred lines of option declarations. The published configuration
lives in these defaults: a bare run reproduces it, and only what genuinely varies
between runs needs to be passed on the command line.
"""
import argparse

def build_parser():
    """Return the driver's fully-populated argument parser."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="S2,S3,S1")
    ap.add_argument("--task", default="down_long")
    ap.add_argument("--two_cost_force", action="store_true",
                    help="recover force as a preference via two costs: press_force=F^2 (effort) + "
                         "press_capacity=(F-Fmax)^2 (pull to capacity)")
    ap.add_argument("--force_max", type=float, default=80.0,
                    help="fmax (N) for the press_capacity term (two_cost_force)")
    ap.add_argument("--force_max_map", default=None,
                    help="per-subject Fmax as 'S1:14,S2:28,S3:47'; overrides --force_max for the "
                         "listed subjects so a pooled run keeps each subject's own "
                         "capacity/balance inste")
    ap.add_argument("--cycle", type=int, default=5)
    ap.add_argument("--dominant", default=None,
                    help="biomech feature dominant in w* (else near-uniform 0.01)")
    ap.add_argument("--dominant_val", type=float, default=1.0)
    ap.add_argument("--low_val", type=float, default=0.01)
    ap.add_argument("--n_cycles", type=int, default=5,
                    help="cycles per subject (demos = n_subjects * n_cycles). 3 subjects x 3 "
                         "cycles = 9 demos")
    ap.add_argument("--feature_scale", action=argparse.BooleanOptionalAction, default=True,
                    help="per-feature normalization in MO_IRL (equalizes the gradient share; drops "
                         "Lambda/Beta)")
    ap.add_argument("--q_norm_force_weight", type=float, default=0.0,
                    help="force-aware line search: add q_norm_force_weight * force_RMSE(N) to the "
                         "joint criterion")
    ap.add_argument("--representative", action=argparse.BooleanOptionalAction, default=True,
                    help="pick the --n_cycles most representative cycles per subject")
    ap.add_argument("--rep_offset", type=int, default=0,
                    help="skip the first N representative cycles")
    ap.add_argument("--learn_task_only", nargs="+",
                    default=["progress_vel", "rock_ori", "rail_lat"],
                    help="granular unfreeze: learn only these task features")
    ap.add_argument("--normalize_w", dest="normalize_w", action="store_true", default=True,
                    help="renormalize the weight vector each step (OCP-scale-invariant; prevents "
                         "the weights blowing up on non-realizable recorded demos)")
    ap.set_defaults(normalize_w=False)
    ap.add_argument("--ls_dq_tol", type=float, default=2.0,
                    help="velocity-guard tolerance (accept if dq_norm < tol x prev)")
    ap.add_argument("--ls_accel_tol", type=float, default=3.0,
                    help="acceleration-guard ceiling = tol x demo accel magnitude")
    ap.add_argument("--best_last", action=argparse.BooleanOptionalAction, default=True,
                    help="report the last accepted step (Pareto-front endpoint) instead of "
                         "argmin(q_norm)")
    ap.add_argument("--ls_base", choices=["q_norm", "opt", "cost", "pareto"], default="pareto",
                    help="line-search acceptance: 'q_norm'=joint match; 'opt'=feature divergence")
    ap.add_argument("--progress_vel_target_mode", action=argparse.BooleanOptionalAction, default=True,
                    help="quadratic progress_vel (v_rail - target_rail_vel)^2 instead of the "
                         "default linear reward")
    ap.add_argument("--target_rail_vel", type=float, default=0.0,
                    help="target rail velocity for progress_vel_target_mode (0 = s_dot^2)")
    ap.add_argument("--no_capacity", action="store_true",
                    help="drop only press_capacity ((f-Fmax)^2, the unidentified fraction-of- "
                         "capacity term), keeping press_force = f_n^2 as the force feature")
    ap.add_argument("--press_in_actuation", action="store_true",
                    help="add the normal press to the contact actuation along crocoddyl's exact "
                         "ContactModel1D axis")
    ap.add_argument("--press_normal_dual", action=argparse.BooleanOptionalAction, default=True,
                    help="corrected scheme: the solve does not "
                         "inject the normal -> the contact dual is free")
    ap.add_argument("--press_friction_dual", action=argparse.BooleanOptionalAction, default=True,
                    help="option B: the rollout's tangential friction mu*f_n uses its own lagged "
                         "contact dual")
    ap.add_argument("--demo_force_measured", action="store_true",
                    help="two-cost only: inject the measured press into the demo's "
                         "press_force=(f/Fmax)^2 and press_capacity=((f-Fmax)/Fmax)^2 features")
    ap.add_argument("--ls_cold_start", action=argparse.BooleanOptionalAction, default=True,
                    help="cold-start the line-search baseline and every probe from the rail-IK "
                         "warmstart instead of the expert demo")
    ap.add_argument("--use_recorded", action=argparse.BooleanOptionalAction, default=True,
                    help="use the recorded mocap cycles as demos (real-data mode) instead of "
                         "synthetic OCP-at-w* demos. us via inverse dynamics on each subject's own "
                         "body")
    ap.add_argument("--n_w", type=int, default=1,
                    help="time-varying shared cost: number of weight windows (windowed) or solver "
                         "granularity (basis). 1 = constant")
    ap.add_argument("--mode", choices=["windowed", "basis"], default="basis",
                    help="w(t) parametrization: windowed (piecewise-constant, n_w blocks) or basis "
                         "(Gaussian, K basis funcs)")
    ap.add_argument("--K", type=int, default=12,
                    help="basis mode: number of Gaussian basis functions (default n_w)")
    ap.add_argument("--tv_lo", type=float, default=0.15,
                    help="tv_wstar: early-stroke fraction of the exertion weight")
    ap.add_argument("--avg_force", action=argparse.BooleanOptionalAction, default=True,
                    help="use the per-(subject,task) averaged press force (mean of the force "
                         "slices, -fz) as the press target, instead of the noisy per-cycle "
                         "measured force")
    ap.add_argument("--target_force", type=float, default=15.0,
                    help="constant contact-force target (N) for the press_force residual and the "
                         "friction actuation")
    ap.add_argument("--static_stick", action="store_true",
                    help="use a static stick (fixed rail) instead of the moving stick")
    ap.add_argument("--tau_split", choices=["none", "full", "pruned"], default="full",
                    help="per-segment torque experiment: 'full' = 5 Tau_<group> + 5 Eng + "
                         "press_force (global Tau dropped); 'pruned' = only the identifiable "
                         "proximal Tau")
    ap.add_argument("--max_iter", type=int, default=12)
    ap.add_argument("--line_search_steps", type=int, default=14)
    ap.add_argument("--sqp_iter", type=int, default=50)
    ap.add_argument("--init_press_cap", type=float, default=0.0,
                    help="start the IRL's press_capacity weight high (not the style_init floor) so "
                         "the challenger converges from iter 0")
    ap.add_argument("--init_press_force", type=float, default=-1.0,
                    help="explicit initial press_force weight (default -1 = off)")
    ap.add_argument("--force_strict_slack_frac", type=float, default=0.0,
                    help="fixed (supplied) force as a hard OCP constraint: the contact dual is "
                         "bounded to measured f_n +/- slack_frac*f_n (band center = avg_force "
                         "profile)")
    ap.add_argument("--press_in_effort", action=argparse.BooleanOptionalAction, default=True,
                    help="the recover-and-identify option: count the freely-recovered normal press "
                         "dual as proximal muscle effort")
    ap.add_argument("--warmstart_press", type=float, default=0.0,
                    help="bake a light generic press (N) into the IK/cold warmstart torques so the "
                         "challenger+rollout seed presses")
    ap.add_argument("--replay_npz", default=None)
    ap.add_argument("--demos_from_npz", default=None,
                    help="train the IRL on the projected-feasible demos")
    ap.add_argument("--drop_cycles", default="",
                    help="comma-separated cycle ids to exclude from the training pairs "
                         "(e.g. cycles that fail to project to feasibility)")
    ap.add_argument("--replay_cold", action="store_true",
                    help="replay: cold IK warmstart (else warm-start from demo traj)")
    ap.add_argument("--press_cap_scale", type=float, default=1.0,
                    help="replay: multiply recovered press_capacity weight (>1 pushes F* toward "
                         "Fmax; probes whether the cost can drive more press)")
    ap.add_argument("--press_force_scale", type=float, default=1.0,
                    help="replay: multiply recovered press_force weight (<1 lets force rise)")
    ap.add_argument("--record_rollout", type=int, default=None,
                    help="replay: record the full CSQP iterate history (xs,KKT,step per iteration) "
                         "for this cycle index (0-4), to pinpoint where the cold rollout diverges")
    ap.add_argument("--effort_limits", action="store_true",
                    help="hard box-constrain commanded torque u to per-joint URDF effortLimit")
    ap.add_argument("--effort_limit_scale", type=float, default=1.0,
                    help="uniform scale on the effort box (>1 relaxes, <1 tightens)")
    ap.add_argument("--project_demo", action="store_true")
    ap.add_argument("--term_pos_slack", type=float, default=0.02,
                    help="hard terminal tool-position constraint half-width (m)")
    ap.add_argument("--project_iter", type=int, default=300)
    ap.add_argument("--project_wq", type=float, default=1e4, help="q-tracking weight")
    ap.add_argument("--project_wv", type=float, default=1e2, help="v-tracking weight")
    ap.add_argument("--project_wf", type=float, default=1e1,
                    help="contact-force tracking toward measured profile: pins the freed normal- "
                         "force dual so projected torque is unique/physical")
    ap.add_argument("--project_wq2", type=float, default=1e2,
                    help="stage-2 q-tracking weight (< project_wq): lets q drift a little so gap "
                         "can close while force is pulled to measured")
    ap.add_argument("--contact_normal_track", action="store_true",
                    help="contactModel1D xref[t] rides the recorded normal penetration")
    ap.add_argument("--project_kinematics", action="store_true",
                    help="minimal-disturbance IK: shift penetrating demo frames onto the surface")
    ap.add_argument("--demo_euler_consistent", action="store_true",
                    help="bake demo v,a as forward differences (match the Euler shooting "
                         "integrator) instead of np.gradient (central)")
    ap.add_argument("--demo_smooth_q", type=int, default=0,
                    help="savitzky-Golay window (odd, 0=off) applied to the resampled demo "
                         "positions before differentiating")
    ap.add_argument("--demo_edge_hold", type=int, default=0,
                    help="clamp the first and last N (projected) demo control vectors to the value "
                         "at node N / -(N+1) (0=off)")
    ap.add_argument("--contact_consistent_accel", action=argparse.BooleanOptionalAction, default=True,
                    help="path A: project the recorded q̈ through the contact constraint")
    ap.add_argument("--force_mean", type=float, default=None,
                    help="rescale the measured avg force profile to this mean (N), keeping its "
                         "shape")
    ap.add_argument("--demo_press_warmstart", type=float, default=0.0,
                    help="synthetic demo: seed the solve with a lightly-pressing (N Newton) "
                         "contact trajectory")
    ap.add_argument("--cont_fracs", type=str, default="0.3,0.6,1.0",
                    help="press_capacity continuation stages for --challenger_continuation")
    ap.add_argument("--tau_norm_accept", action=argparse.BooleanOptionalAction, default=True,
                    help="add torque-trajectory error (tau_norm, challenger vs demo torques) as a "
                         "third pareto acceptance axis (with q_norm, opt_div)")
    ap.add_argument("--pareto_qnorm_cap", type=float, default=float('inf'),
                    help="q_norm-favoring cap for --ls_base pareto: accept a Pareto step only if "
                         "q_norm also stays within this fraction of the baseline")
    ap.add_argument("--challenger_press_warmstart", action="store_true",
                    help="warm-start the IRL challenger and line-search probes from the demo's "
                         "pressing torques")
    ap.add_argument("--ramp_fracs", type=str, default="0.02,0.08,0.2,0.4,0.65,1.0",
                    help="press_capacity continuation stages (fractions of the full weight), "
                         "comma-separated")
    ap.add_argument("--demo_force_continuation", action="store_true",
                    help="synthetic demo: ramp press_capacity up in warm-started stages so the "
                         "demo climbs to high press instead of a cold solve stalling ~20N")
    ap.add_argument("--check_press_grad", action="store_true",
                    help="check the contact-force (press) gradient: analytical df_du (what CSQP "
                         "uses) vs finite-difference dlambda/du (physics)")
    ap.add_argument("--project_bake_ureg", type=float, default=0.0,
                    help="path B mop-up: stage-2 u-reg weight toward the")
    ap.add_argument("--eps_abs", type=float, default=None,
                    help="inner-QP absolute tolerance (None -> model default 1e-10)")
    ap.add_argument("--max_qp_iters", type=int, default=None)
    ap.add_argument("--termination_tolerance", type=float, default=None)
    ap.add_argument("--pool_scope", choices=["averaged", "same_demo", "all_demos"],
                    default="averaged",
                    help="population pooling (paper Eq. 4). 'averaged'")
    ap.add_argument("--pool_last_k", type=int, default=None,
                    help="cap the accumulating per-demo rollout pool to the last K rollouts "
                         "(Sarmad pool_last_k)")
    ap.add_argument("--style_init", type=float, default=1e-6,
                    help="starting weight on every biomech (non-task) feature")
    ap.add_argument("--reg_lambda", type=float, default=1e-5,
                    help="l1 weight in the elastic-net regularizer on w (default 1e-5, ~off)")
    ap.add_argument("--reg_beta", type=float, default=1e-3,
                    help="l2 weight in the elastic-net regularizer on w (default 1e-8, ~off)")
    ap.add_argument("--lock_thorax", action=argparse.BooleanOptionalAction, default=True,
                    help="hold the thoracic (trunk) DOF at q0 and freeze its effort terms "
                         "(Tau_thoracic/Eng_thoracic) out of the IRL")
    ap.add_argument("--lock_thorax_tol", type=float, default=1e-3,
                    help="position half-window (rad) for the thoracic hold at q0")
    ap.add_argument("--outdir", default="toy_irl_study/csqp_population")
    ap.add_argument("--force_scale", default=None,
                    help="JSON mapping 'subject/task' (or 'subject') -> multiplicative scale on "
                         "the averaged press force")
    ap.add_argument("--hard_rail", action=argparse.BooleanOptionalAction, default=True,
                    help="enforce the contact point on the rail line as a hard constraint during "
                         "contact (±hard_rail_tol lateral, sliding axis free)")
    ap.add_argument("--hard_rail_tol", type=float, default=0.012,
                    help="lateral half-width (m) for --hard_rail (default 1mm)")
    ap.add_argument("--windowed_rail", action=argparse.BooleanOptionalAction, default=True,
                    help="estimate the rail from the contact-window segment only (not the full "
                         "stroke) so long cycles don't overshoot it")
    ap.add_argument("--rail_from_contact", action=argparse.BooleanOptionalAction, default=True,
                    help="two-pass rail: re-fit the rail from the rock_contact_point")
    ap.add_argument("--contact_normal_slack", type=float, default=0.0,
                    help="compliant normal contact in [0,1): scale down the contact Baumgarte "
                         "gains so the contact gives a bit under load")
    ap.add_argument("--contact_mask_thresh", type=float, default=0.15,
                    help="force threshold for --contact_mask_from_force, as a fraction of the "
                         "profile peak (default 0.15 = 15%% of peak press)")
    ap.add_argument("--contact_windows", default=None,
                    help="JSON mapping 'subject/task' -> [start, end] contact window (fractions of "
                         "the stroke in [0,1])")
    ap.add_argument("--rock_ori_seed", type=float, default=None,
                    help="override the rock_ori (tool-orientation hold) seed weight; default 1e-6 "
                         "(effectively off)")
    ap.set_defaults(
        balanced_press_init=False,
        basis_pernode=False,
        challenger_continuation=False,
        contact_aware_demo=False,
        contact_mask_from_force=False,
        demo_force_at_target=False,
        flatten_force=False,
        force_ref_fill=False,
        force_track_max=False,
        learn_task=False,
        ls_use_accel=False,
        ls_use_dq=False,
        measured_force=False,
        no_press_feature=False,
        no_press_force=False,
        q_norm_meanjoint=False,
        reuse_demo=False,
        select_best_feature=False,
        slsqp_inner=False,
        tv_wstar=False,
        verify_demo_force=False,
        warmstart_gate_contact=False,
        warmstart_neutral=False,
    )

    return ap
