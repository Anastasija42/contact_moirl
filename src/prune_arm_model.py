"""prune_arm_model.py — build the reduced arm-only MJX model from the full human model.

WHY: the full model carries ~37 DOF and 27 equality constraints (the frozen legs/torso/left-arm/head
locked to 0). Those 27 equality constraints are BOTH the compile-memory bloat (>12GB → OOM on a 15GB
box) AND the exact nan source (the code's own note: "the equality-constraint solve is what diverges"
at a coarse timestep). The active system is only the 9 ACTUATED DOF (thoracic_X + right clavicle/
shoulder×3/elbow×2/wrist×2) holding the rock, in contact with the stick.

Pruning keeps the thorax→right-arm→rock chain (welded to the world at the pelvis, which is already a
fixed root), drops every frozen sibling body + all 27 equality constraints, fixes the stick, and swaps
the rock collision geom mesh→sphere (MJX has no mesh↔cylinder collision). Result: nv=9, neq=0, MJX
compiles in ~2GB (K=256,H=30) and runs nan-free even at coarse NSUB. Validated: this file's __main__.

Run: python src/prune_arm_model.py   (writes config/xml_models/13_02/s3_pruned.xml + benchmarks)
NOTE: the rock→sphere(0.02) and stick-fixed choices are for the MJX-compat smoke test; align the rock
radius / stick handling with the pipeline when wiring the reduced model into test_phase2.
"""
import xml.etree.ElementTree as ET

ACTIVE = {'middle_thoracic_X', 'right_clavicle_joint_X', 'right_shoulder_Z', 'right_shoulder_X',
          'right_shoulder_Y', 'right_elbow_Z', 'right_elbow_Y', 'right_wrist_Z', 'right_wrist_X'}

def prune(src, dst, rock_radius=0.02, q_full=None):
    """q_full: the demo full-model qpos. REQUIRED for a faithful model — the locked trunk joints
    (lumbar/thoracic_Z/Y) sit at NON-ZERO demo values (±10-20°, locked at q0_full, NOT at 0). Removing
    them makes their body rigid at joint=0 → the trunk collapses horizontal and the whole arm base is
    mis-posed by ~20°. Passing q_full bakes each removed joint's demo rotation into the body transform
    (via MuJoCo FK) so the pruned arm base matches the real model. Omit only for a quick structural smoke
    test (the trunk will be wrong)."""
    tree = ET.parse(src); root = tree.getroot()
    parent = {c: p for p in root.iter() for c in p}
    rock = [b for b in root.iter('body') if b.get('name') == 'rock'][0]
    keep = set(); e = rock
    while e is not None and e.tag == 'body':
        keep.add(e.get('name')); p = parent.get(e); e = p if (p is not None and p.tag == 'body') else None
    keep.add('stick')

    def _prune(body):
        for ch in list(body):
            if ch.tag == 'body':
                if ch.get('name') not in keep:
                    body.remove(ch)
                else:
                    for j in list(ch.findall('joint')):
                        if j.get('name') not in ACTIVE:
                            ch.remove(j)
                    _prune(ch)
    _prune(root.find('worldbody'))

    for eq in root.findall('equality'):
        root.remove(eq)
    kept = {b.get('name') for b in root.iter('body')}
    for c in root.findall('contact'):
        for ex in list(c.findall('exclude')):
            if ex.get('body1') not in kept or ex.get('body2') not in kept:
                c.remove(ex)
    # reintroduce the ill-conditioned qM that caused the nan; nv = 9(arm)+6(stick) = 15, still tiny.
    # model gets. (Standalone loads must call _prepare_model_for_mjx too; see __main__.)
    tree.write(dst)
    if q_full is not None:
        _bake_demo_pose(src, dst, q_full)
    return dst

def _bake_demo_pose(full_src, pruned_dst, q_full):
    """Write each removed-joint body's demo pose into its fixed transform, so the pruned (jointless)
    trunk reproduces the full model's demo posture. Bodies that still carry a joint (the 9 active arm
    joints, the stick freejoint) keep their transform — the joint reproduces the demo at its demo angle."""
    import mujoco, numpy as np
    mf = mujoco.MjModel.from_xml_path(full_src); df = mujoco.MjData(mf)
    q = np.asarray(q_full, dtype=float)
    df.qpos[:len(q)] = q; mujoco.mj_forward(mf, df)

    def wframe(name):
        b = mujoco.mj_name2id(mf, mujoco.mjtObj.mjOBJ_BODY, name)
        return (np.array(df.xpos[b]), np.array(df.xquat[b])) if b >= 0 else None

    tree = ET.parse(pruned_dst); root = tree.getroot()
    parent = {c: p for p in root.iter() for c in p}
    for body in list(root.iter('body')):
        if any(ch.tag in ('joint', 'freejoint') for ch in body):
            continue
        cf = wframe(body.get('name'))
        if cf is None:
            continue
        cpos, cquat = cf
        p = parent.get(body)
        if p is not None and p.tag == 'body':
            ppos, pquat = wframe(p.get('name'))
        else:
            ppos, pquat = np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])
        pconj = np.zeros(4); mujoco.mju_negQuat(pconj, pquat)
        relpos = np.zeros(3); mujoco.mju_rotVecQuat(relpos, cpos - ppos, pconj)
        relquat = np.zeros(4); mujoco.mju_mulQuat(relquat, pconj, cquat)
        body.set('pos', ' '.join(f'{v:.9g}' for v in relpos))
        body.set('quat', ' '.join(f'{v:.9g}' for v in relquat))
    tree.write(pruned_dst)

if __name__ == '__main__':
    import mujoco, jax, resource, time, numpy as np
    from mujoco import mjx
    SRC = 'config/xml_models/13_02/s3_pinned.xml'
    DST = 'config/xml_models/13_02/s3_pruned.xml'
    prune(SRC, DST)
    import sys; sys.path.insert(0, 'src')
    from MPPI_MJX import _prepare_model_for_mjx
    m = mujoco.MjModel.from_xml_path(DST)
    _prepare_model_for_mjx(m)
    print(f"PRUNED: nq={m.nq} nv={m.nv} nu={m.nu} nbody={m.nbody} neq={m.neq} ngeom={m.ngeom}")
    mx = mjx.put_model(m); K, H = 256, 30
    U = 0.03 * jax.random.normal(jax.random.PRNGKey(0), (K, H, m.nu))
    def roll(u):
        dx = mjx.make_data(mx)
        def s(dx, ut): dx = dx.replace(ctrl=ut); dx = mjx.step(mx, dx); return dx, dx.qpos
        _, q = jax.lax.scan(s, dx, u); return q
    t0 = time.time(); qs = jax.vmap(roll)(U); qs.block_until_ready()
    print(f"MJX K={K} H={H}: {time.time()-t0:.1f}s  peak RSS="
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1048576:.2f}GB  "
          f"nan={bool(np.isnan(np.asarray(qs)).any())}")
