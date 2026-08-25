"""Initial postures only: each body holding its rock above the shared stick, pointing at it.

No OCP solve -- this is the starting pose the forward rollout would begin from, so the
grip can be judged before spending an hour solving seven trajectories.

The rock is aimed at the stick's contact NORMAL, which is one shared direction only under
--share_stick_frame. Position and orientation are solved TOGETHER (weighted damped least
squares) rather than putting orientation in the null space of position: the null-space form
can only orient using redundancy the body has spare at that pose, and the short-armed
hominins have none, so they came out 48-78 deg off while the apes hit 0.
"""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, '/home/ana/tool_handling/friction_lib/build')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "morphologies_study"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import pinocchio as pin
import crocoddyl  # noqa: F401  (must precede friction_lib)
import run_species_forward as rsf
rsf.URDF_DIR = rsf.GEN_URDF_DIR
from run_csqp_identifiability import load_geometry

ORDER = ["human", "homo_neanderthal", "homo_naledi", "australopithecus_sediba",
         "australopithecus_prometheus", "chimp", "bonobo"]


def rock_axes(h):
    """The rock's principal axes, expressed in the CONTACT frame.

    Returns (longest, middle, shortest) as unit column vectors. The rock is a real scanned
    cobble (5.5 x 7.4 x 4.3 cm), so 'which way is it facing' is a property of the mesh, not
    of the contact frame -- which is built per body and carries no shape information.
    """
    vm, m = h.visual_model, h.pin_model
    d, gd = m.createData(), vm.createData()
    q = np.asarray(h.q0, float)
    pin.forwardKinematics(m, d, q)
    pin.updateGeometryPlacements(m, d, vm, gd)
    pin.framesForwardKinematics(m, d, q); pin.updateFramePlacements(m, d)
    for i, g in enumerate(vm.geometryObjects):
        if "rock" not in g.name:
            continue
        o = g.geometry
        if not hasattr(o, "vertices"):
            continue
        V = np.asarray(o.vertices(), float)
        C = V - V.mean(axis=0)
        _, _, Vt = np.linalg.svd(C, full_matrices=False)      # rows = principal axes
        R_world = np.asarray(gd.oMg[i].rotation)              # mesh -> world
        R_c = np.asarray(d.oMf[h.contact_frame_id].rotation)  # contact -> world
        # principal axes in the contact frame, longest first
        return [R_c.T @ (R_world @ Vt[k]) for k in range(3)]
    return None


def across_dir(n_out, ax):
    """The tangential direction ACROSS the stick: perpendicular to both the outward normal
    and the stick's own axis. This is the scraping edge's direction."""
    t = np.cross(np.asarray(n_out, float), np.asarray(ax, float))
    return t / (np.linalg.norm(t) + 1e-12)


def demo_contact_rotation(h, q_traj, frame_frac=0.15):
    """The rock attitude the DEMONSTRATION actually uses, re-expressed for this body.

    Imposing a synthetic alignment looks wrong because the real grip is not aligned with
    anything: over the human stroke the cobble's long axis holds ~40 deg from across the
    stick, ~75 deg from along it and ~55 deg from the normal -- an oblique, edge-on hold,
    and near-constant (40 -> 40 deg end to end), which is also why rock_ori is the heaviest
    recovered weight. So take the demonstrated rotation IN THE STICK'S FRAME and give every
    body that same tool-to-workpiece relationship. This transfers the grip, not the human's
    joint angles.
    """
    m, d = h.pin_model, h.pin_model.createData()
    q = np.asarray(q_traj, float)
    k = int(round(frame_frac * (len(q) - 1)))
    pin.framesForwardKinematics(m, d, q[k]); pin.updateFramePlacements(m, d)
    Rc = np.asarray(d.oMf[h.contact_frame_id].rotation)
    # Return where the ROCK'S OWN principal axes point in the world, not the contact frame's
    # rotation. The contact frame is constructed per body, so handing every taxon the same
    # contact rotation still left the physical cobble lying differently (naledi 54 deg vs
    # 38 for the rest). The mesh axes are the thing that has to match.
    return np.column_stack([Rc @ a for a in rock_axes(h)])      # world <- principal


def desired_contact_rotation(h, n_out, ax):
    """World rotation the CONTACT frame should hold: rock's long side laid ACROSS the stick.

    The shortest principal axis is driven into the stick (anti-parallel to the outward
    normal) and the longest is laid ACROSS it -- perpendicular to the stick's axis, in the
    tangent plane -- so the cobble's long edge crosses the work like a scraper blade rather
    than lying down the length of it. Aiming only the contact frame's pointing axis leaves
    spin about it free, which is exactly the freedom that decides this.
    """
    axes = rock_axes(h)
    if axes is None:
        return None
    P = np.column_stack(axes)                    # contact <- principal
    d3 = -np.asarray(n_out, float)               # shortest axis presses into the stick
    d1 = across_dir(n_out, ax)                   # longest lies ACROSS it (the blade edge)
    d1 = d1 - np.dot(d1, d3) * d3
    d1 = d1 / (np.linalg.norm(d1) + 1e-12)
    # keep the sign that is nearer the rock's current lie, so it is not flipped end-for-end
    m_, d_ = h.pin_model, h.pin_model.createData()
    pin.framesForwardKinematics(m_, d_, np.asarray(h.q0, float)); pin.updateFramePlacements(m_, d_)
    R_c = np.asarray(d_.oMf[h.contact_frame_id].rotation)
    if float((R_c @ axes[0]) @ d1) < 0:
        d1 = -d1
    d2 = np.cross(d3, d1)
    D = np.column_stack([d1, d2, d3])            # world <- principal
    return D @ P.T                               # world <- contact


def solve_start(h, hover_m, ori_w, iters=600, full_ori=True, slide=False, demo_R=None,
                anchor=True, lift=0.0, comfort=0.0, no_trunk=True, normal_up=False):
    """(q, position residual m, degrees off perpendicular) for the hover pose."""
    m, d = h.pin_model, h.pin_model.createData()
    fid = h.contact_frame_id
    qmin = np.asarray(h.pin_model.lowerPositionLimit, float)
    qmax = np.asarray(h.pin_model.upperPositionLimit, float)
    n_out = np.asarray(h.R_surface, float)[:, 2]          # outward from the stick
    if normal_up:
        # Contact on the TOP of the shaft, so the stick sits under the hand. The inherited
        # normal points wherever the recorded contact happened to be around the girth, which
        # is what leaves the hand alongside the stick rather than over it.
        _ax0 = h.p_end_world - h.p_start_world
        _ax0 = _ax0 / (np.linalg.norm(_ax0) + 1e-12)
        _up = np.array([0.0, 0.0, 1.0]) - float(np.dot([0.0, 0.0, 1.0], _ax0)) * _ax0
        if np.linalg.norm(_up) > 1e-6:
            n_out = _up / np.linalg.norm(_up)
    ax = h.p_end_world - h.p_start_world
    ax = ax / np.linalg.norm(ax)

    q_mid = 0.5 * (qmin + qmax)          # middle of each joint's own range
    q = np.asarray(h.q0, float).copy()
    pin.framesForwardKinematics(m, d, q); pin.updateFramePlacements(m, d)
    p_tool = d.oMf[fid].translation.copy()
    # hover on the shoulder side, level with where the tool already is along the stick
    sh = (d.oMi[m.getJointId('right_shoulder_Y')].translation.copy()
          if m.existJointName('right_shoulder_Y') else p_tool)
    out = sh - h.p_start_world
    out = out - np.dot(out, ax) * ax
    out = out / (np.linalg.norm(out) + 1e-12)
    # LIFT the hover toward vertical. Pointing it purely at the shoulder keeps the approach
    # inside each body's workspace, but its elevation is set by where the shoulder happens
    # to be: +43 deg on the down-stroke, only +16-21 deg on the up, which leaves the tool
    # barely 2 cm above the stick and mostly off to one side. Blending in world-up (taken
    # perpendicular to the stick) lifts it without swinging the arm over the workpiece, the
    # way a pure top-of-stick direction did.
    if lift > 0:
        up_perp = np.array([0.0, 0.0, 1.0]) - float(np.dot([0.0, 0.0, 1.0], ax)) * ax
        n = np.linalg.norm(up_perp)
        if n > 1e-6:
            out = out + lift * (up_perp / n)
            out = out / (np.linalg.norm(out) + 1e-12)
    # WHERE ALONG THE RAIL the stance begins. Projecting the tool's incidental position
    # scattered the seven starts from -49% to +19% of the rail on the up-stroke, so no two
    # bodies began the same task. Anchor every body at the rail start instead.
    on_axis = h.p_start_world if anchor else (
        h.p_start_world + float(np.dot(p_tool - h.p_start_world, ax)) * ax)
    # The 2 cm stand-off must be taken along the CONTACT NORMAL, which is where the
    # cylinder is offset, not along the shoulder-side hover direction. Using `out` here put
    # the tool 32 mm from the drawn axis instead of 20 -- a 12 mm float that no stick offset
    # could close, because the two directions simply differ.
    # `on_axis` already lies ON the surface: the rail is the demonstrated tool-tip path and
    # the cylinder axis sits one radius inboard of it. Adding a further radius here put the
    # tool at exactly 2R from the axis. The stand-off is therefore zero, and `hover` is the
    # only clearance.
    target = on_axis + hover_m * out

    if demo_R is not None:
        # map THIS body's rock axes onto the demonstrated world directions
        P = np.column_stack(rock_axes(h))          # contact <- principal
        R_des = np.asarray(demo_R, float) @ P.T    # world <- contact
    elif full_ori:
        R_des = desired_contact_rotation(h, n_out, ax)
    else:
        R_des = None

    for _ in range(iters):
        pin.framesForwardKinematics(m, d, q); pin.updateFramePlacements(m, d)
        Rc = np.asarray(d.oMf[fid].rotation)
        e_p = d.oMf[fid].translation - target
        if slide:
            # Freeing WHERE ALONG THE STICK the tool hovers looks like a free DOF -- it was
            # pinned to wherever the retarget left the tool -- but unclamped it simply walks
            # off the rail (-211% to +279% of the stick length) and the attitude collapses
            # with it (apes 25-48 deg off). Left here, off by default, for a clamped retry.
            e_p = e_p - float(np.dot(e_p, ax)) * ax
        if R_des is not None:
            # full 3-DOF: log(Rc^T R_des) in the tool's local frame
            w_err = -pin.log3(Rc.T @ R_des)
        else:
            cx = np.cross(Rc[:, 2], n_out)
            dt = float(np.clip(np.dot(Rc[:, 2], n_out), -1.0, 1.0))
            ln = np.linalg.norm(cx)
            w_err = -(Rc.T @ (cx / ln * np.arctan2(ln, dt))) if ln > 1e-9 else np.zeros(3)
        Jp = pin.computeFrameJacobian(m, d, q, fid, pin.LOCAL_WORLD_ALIGNED)[:3, :]
        Jw = pin.computeFrameJacobian(m, d, q, fid, pin.LOCAL)[3:, :]
        # HOLD THE TRUNK. Left free it leans in to meet the stick -- measured 11.5 deg for
        # the human and 29.3 for the chimp -- which the transfer does not allow (the thorax
        # is static there), so the posture would not be reachable as posed. Zero the
        # thoracic columns and let the stick position take up the slack instead.
        if no_trunk:
            for _n in ('middle_thoracic_X', 'middle_thoracic_Y', 'middle_thoracic_Z'):
                if m.existJointName(_n):
                    _iv = int(m.joints[m.getJointId(_n)].idx_v)
                    Jp[:, _iv] = 0.0; Jw[:, _iv] = 0.0
        J = np.vstack([Jp, ori_w * Jw])
        e = np.concatenate([e_p, ori_w * w_err])
        dq = -J.T @ np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), e)
        # COMFORT in the null space: the rock's position and attitude use six DOFs; the
        # rest of the arm is free, and left alone it settles wherever the iteration drifts.
        # Bias the remainder toward the middle of each joint's own range so the body starts
        # from a posture it has margin in, rather than one that happens to satisfy the tool.
        if comfort > 0.0:
            Jp_ = np.linalg.solve(J @ J.T + 1e-6 * np.eye(6), J)
            N = np.eye(m.nv) - J.T @ Jp_
            dq = dq + comfort * (N @ pin.difference(m, q, q_mid))
        if np.linalg.norm(dq) < 1e-9:
            break
        q = np.clip(pin.integrate(m, q, dq), qmin, qmax)

    pin.framesForwardKinematics(m, d, q); pin.updateFramePlacements(m, d)
    Rc = np.asarray(d.oMf[fid].rotation)
    off = np.degrees(np.arccos(np.clip(float(Rc[:, 2] @ n_out), -1.0, 1.0)))
    # where it ended up along the rail, as a fraction of the stick
    s_along = float((d.oMf[fid].translation - h.p_start_world) @ ax)
    L = float(np.linalg.norm(h.p_end_world - h.p_start_world))
    # how the cobble actually lies: long axis vs the stick, short axis vs the normal
    axes = rock_axes(h)
    lon = np.degrees(np.arccos(np.clip(
        abs(float((Rc @ axes[0]) @ across_dir(n_out, ax))), -1.0, 1.0)))
    sho = np.degrees(np.arccos(np.clip(abs(float((Rc @ axes[2]) @ n_out)), -1.0, 1.0)))
    res = d.oMf[fid].translation - target
    if slide:
        res = res - float(np.dot(res, ax)) * ax          # standoff error only
    return q, float(np.linalg.norm(res)), off, lon, sho, 100.0 * s_along / (L + 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--geom_subject", default="S2")
    ap.add_argument("--task", default="down_long")
    ap.add_argument("--cycle", type=int, default=20)
    ap.add_argument("--rail_length", type=float, default=0.20)
    ap.add_argument("--stick_radius", type=float, default=0.02)
    ap.add_argument("--hover", type=float, default=0.05)
    ap.add_argument("--normal_up", action="store_true",
                    help="contact the TOP of the shaft so the stick sits under the "
                         "hand, instead of wherever around the girth the recording "
                         "happened to touch.")
    ap.add_argument("--grip_tilt", type=float, default=0.0,
                    help="tilt the target grip about the stick axis, in degrees. Positive lays the cobble flatter. 0 = the demonstrated hold.")
    ap.add_argument("--stick_offset", default="",
                    help="move the stick before solving, as along,out,across in cm "
                         "(out = away from the stick along the contact normal).")
    ap.add_argument("--comfort", type=float, default=0.0,
                    help="null-space bias toward mid-range joints, so the free DOFs start with margin instead of wherever the IK drifts. Try 0.05-0.2.")
    ap.add_argument("--lift", type=float, default=0.0,
                    help="blend world-up into the shoulder-side hover direction, so the "
                         "tool clears the stick rather than sitting beside it. 0 = pure "
                         "shoulder direction (up-stroke elevation only ~20 deg).")
    ap.add_argument("--ori_w", type=float, default=0.35,
                    help="orientation weight vs position in the 6D solve")
    ap.add_argument("--synthetic-grip", action="store_true",
                    help="impose an axis-aligned grip (long edge across the stick, face flat) "
                         "instead of the demonstrated oblique one. Looks wrong: the real hold "
                         "is not aligned with any stick direction.")
    ap.add_argument("--grip_frame", type=float, default=0.15,
                    help="fraction into the demo stroke to take the grip from")
    ap.add_argument("--axis-only", action="store_true",
                    help="constrain only the pointing axis (leaves the rock free to spin "
                         "about it, so the long side lands wherever)")
    ap.add_argument("--out", default="analysis/start_postures.json")
    a = ap.parse_args()

    w_run, _ = rsf.load_wstar(a.weights)
    q0f, qtraj, dt = load_geometry(a.geom_subject, a.task, a.cycle)
    ps, pe, rad = rsf.reference_stick(a.geom_subject, a.task, a.cycle, w_run)
    rsf.STICK_RADIAL = np.asarray(rad, float)
    rsf.STICK_RADIUS = float(a.stick_radius)
    axis = (pe - ps) / np.linalg.norm(pe - ps)
    if a.stick_offset:
        _c = [float(v) / 100.0 for v in a.stick_offset.split(",")]
        _n = np.asarray(rad, float); _n = -( _n - float(np.dot(_n, axis)) * axis)
        _n = _n / (np.linalg.norm(_n) + 1e-12)
        _a2 = np.cross(_n, axis)
        _off = _c[0] * axis + _c[1] * _n + _c[2] * _a2
        ps = ps + _off; pe = pe + _off
        print(f"stick moved by {np.round(_off*100,1)} cm (along/out/across = {a.stick_offset})")
    stick = (ps, ps + a.rail_length * axis)

    def _tilt(R_world, axis, deg):
        """Rotate a target attitude about the stick's axis. Positive lays the cobble
        flatter (more horizontal); the demonstrated hold is the zero point."""
        if abs(deg) < 1e-9:
            return R_world
        th = np.deg2rad(deg)
        k = np.asarray(axis, float); k = k / (np.linalg.norm(k) + 1e-12)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        Rz = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
        return Rz @ R_world

    demo_R = None
    if not a.synthetic_grip:
        _h = rsf.build_species("human", a.geom_subject, a.task, q0f, qtraj, dt, w_run,
                               stick=stick)
        demo_R = demo_contact_rotation(_h, qtraj, a.grip_frame)
        if abs(a.grip_tilt) > 1e-9:
            demo_R = _tilt(demo_R, axis, a.grip_tilt)
            print(f'grip tilted {a.grip_tilt:+.0f} deg about the stick axis')
        print("grip target: the DEMONSTRATED rock attitude, in the stick's frame")

    out = {}
    print(f"{'taxon':28s}{'long axis across':>20s}{'flat on stick':>15s}{'standoff':>11s}{'along':>10s}")
    for sp in ORDER:
        h = rsf.build_species(sp, a.geom_subject, a.task, q0f, qtraj, dt, w_run, stick=stick)
        q, res, off, lon, sho, along = solve_start(h, a.hover, a.ori_w,
                                                   full_ori=not a.axis_only,
                                                   demo_R=demo_R, lift=a.lift,
                                                   comfort=a.comfort,
                                                   normal_up=a.normal_up)
        out[sp] = dict(q=[float(v) for v in q], off_deg=off, res_mm=res * 1000.0,
                       long_deg=lon, short_deg=sho, along_pct=along)
        print(f"{sp:28s}{lon:17.0f} deg{sho:12.0f} deg{res*1000:8.1f} mm{along:9.0f}%")
    json.dump(dict(stick=[[float(v) for v in stick[0]], [float(v) for v in stick[1]]],
                   stick_radius=a.stick_radius, postures=out),
              open(a.out, "w"), indent=1)
    print(f"saved -> {a.out}")


if __name__ == "__main__":
    main()
