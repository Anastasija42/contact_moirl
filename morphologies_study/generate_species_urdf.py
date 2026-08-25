"""
generate_species_urdf.py
========================
Generate species upper-limb URDFs from the human base + the sourced parameter
table (human_model/urdf/morphs/species_modeling_params.md)

This generator sets, per species, BILATERALLY and from one PARAMS table:
  * segment lengths   -> joint origins  (humerus = elbow_Z.y, forearm = wrist_Z.y,
                         shoulder/clavicle lateral offset = shoulder_Z.z)
  * humeral torsion   -> elbow_Z origin rpy PITCH (about y, the humeral long axis)
  * glenoid tilt      -> shoulder_Z origin rpy PITCH (cranial = +, modelling knob)
  * segment masses    -> link inertial mass; inertia tensor scaled by mass ratio
  * visual mesh scale -> anisotropic: along-limb (sy) from bone-length ratio,
                         girth (sx,sz) from sqrt(mass_ratio / length_ratio);
                         torso/head/legs/joint-blobs by overall body_scale
                         (= body_mass^(1/3) vs human), torso width by torso_w.

Visual scaling is cosmetic (does not touch kinematics/dynamics). Kinematic edits
(lengths, torsion) and mass/inertia DO reach the solver via keep_species_limits /
keep_urdf_inertias in run_species_forward.py.

Run (unified_env, repo root):
    python morphologies_study/generate_species_urdf.py
"""
import math
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASE = REPO / "human_model" / "urdf" / "human.urdf"
OUT = REPO / "human_model" / "urdf" / "generated"

# Reference lengths (mm) the per-species ratios are taken AGAINST. These MUST be
HUM_REF = dict(humerus=276.0, radius=287.0, clavicle=150.0, hand=190.0,
               torsion_deg=165.0, body_mass=65.0)

PARAMS = {
    "human": dict(humerus=311, radius=230, clavicle=150, hand=190,
                  torsion_deg=165, glenoid_tilt_deg=0,
                  m_upper=1.80, m_fore=1.28, m_hand=0.45,
                  body_mass=65, torso_w=1.00,
                  effort_scale=1.00, effort_fore=1.00, vel_scale=1.00, rom={}),
    "homo_neanderthal": dict(humerus=304, radius=224, clavicle=168, hand=190,
                  torsion_deg=137, glenoid_tilt_deg=0,
                  m_upper=2.10, m_fore=1.40, m_hand=0.50,
                  body_mass=77, torso_w=1.19,
                  effort_scale=1.25, effort_fore=1.40, vel_scale=0.80, rom={}),
    "homo_naledi": dict(humerus=256, radius=190, clavicle=135, hand=180,
                  torsion_deg=91, glenoid_tilt_deg=12,
                  m_upper=1.00, m_fore=0.61, m_hand=0.23,
                  body_mass=37, torso_w=0.91,
                  effort_scale=0.80, effort_fore=1.00, vel_scale=1.00, rom={}),
    "australopithecus_sediba": dict(humerus=269, radius=226, clavicle=107.5, hand=175,
                  torsion_deg=117, glenoid_tilt_deg=12,
                  m_upper=1.00, m_fore=0.70, m_hand=0.25,
                  body_mass=29, torso_w=0.95,
                  effort_scale=0.67, effort_fore=1.00, vel_scale=1.00, rom={}),
    "australopithecus_prometheus": dict(humerus=290, radius=245, clavicle=142.9, hand=185,
                  torsion_deg=120, glenoid_tilt_deg=12,
                  m_upper=0.80, m_fore=0.60, m_hand=0.20,
                  body_mass=30, torso_w=1.00,
                  effort_scale=0.68, effort_fore=1.00, vel_scale=1.00, rom={}),
    "chimp": dict(humerus=302, radius=270, clavicle=123, hand=253,
                  torsion_deg=153, glenoid_tilt_deg=10,
                  m_upper=2.10, m_fore=1.50, m_hand=0.70,
                  body_mass=50, torso_w=1.10,
                  effort_scale=1.35, effort_fore=1.00, vel_scale=1.15,
                  rom={"wrist_Z": (-60, 20), "wrist_X": (-15, 15), "elbow_Y": (-20, 80)}),
    "bonobo": dict(humerus=283, radius=256, clavicle=103, hand=234,
                  torsion_deg=150, glenoid_tilt_deg=10,
                  m_upper=1.39, m_fore=1.03, m_hand=0.60,
                  body_mass=41, torso_w=1.05,
                  effort_scale=1.15, effort_fore=1.00, vel_scale=0.90,
                  rom={"wrist_Z": (-60, 20), "wrist_X": (-15, 15), "elbow_Y": (-20, 80)}),
}

ARM_JOINTS = ["clavicle_joint_X", "shoulder_Z", "shoulder_X", "shoulder_Y",
              "elbow_Z", "elbow_Y", "wrist_Z", "wrist_X"]
FOREARM_JOINTS = ["elbow_Z", "elbow_Y", "wrist_Z", "wrist_X"]

MESH_SHAPE = {
    "human":                       dict(fore_bend_deg=0,  upper_twist_deg=0),
    "homo_neanderthal":            dict(fore_bend_deg=0,  upper_twist_deg=10),
    "homo_naledi":                 dict(fore_bend_deg=5,  upper_twist_deg=40),
    "australopithecus_sediba":     dict(fore_bend_deg=15, upper_twist_deg=25),
    "australopithecus_prometheus": dict(fore_bend_deg=20, upper_twist_deg=22),
    "chimp":                       dict(fore_bend_deg=20, upper_twist_deg=8),
    "bonobo":                      dict(fore_bend_deg=18, upper_twist_deg=8),
}
MESHDIR_GEN = REPO / "human_model" / "meshes" / "generated"

TORSION_SIGN = +1.0

def _f(s):
    return [float(x) for x in s.split()]

def _s(v):
    return " ".join(f"{x:.6g}" for x in v)

def set_origin(joint, xyz=None, rpy=None):
    o = joint.find("origin")
    if xyz is not None:
        o.set("xyz", _s(xyz))
    if rpy is not None:
        o.set("rpy", _s(rpy))

def scale_mass_inertia(link, mass_ratio):
    inertial = link.find("inertial")
    if inertial is None:
        return
    m = inertial.find("mass")
    base = float(m.get("value"))
    m.set("value", f"{base * mass_ratio:.6g}")
    I = inertial.find("inertia")
    if I is not None:
        for k in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz"):
            I.set(k, f"{float(I.get(k)) * mass_ratio:.6g}")

def scale_visual_meshes(link, sy=1.0, girth=1.0, iso=None):
    """Multiply mesh scale: sy along the limb (y), girth on x,z; or iso for an
    isotropic factor (joint blobs / torso / head)."""
    for vis in link.findall("visual"):
        mesh = vis.find("geometry/mesh")
        if mesh is None:
            continue
        sc = _f(mesh.get("scale"))
        if iso is not None:
            sc = [sc[0] * iso[0], sc[1] * iso[1], sc[2] * iso[2]] if isinstance(iso, (list, tuple)) \
                 else [c * iso for c in sc]
        else:
            sc = [sc[0] * girth, sc[1] * sy, sc[2] * girth]
        mesh.set("scale", _s(sc))

def apply_limits(root, p):
    """Scale effort + velocity on both arms, and apply right-side ROM overrides.
    Effort/velocity multiply the human <limit> values; ROM replaces lower/upper
    (degrees -> rad) on the right side only (left stays human; it is locked)."""
    for side in ("left", "right"):
        for js in ARM_JOINTS:
            j = jname(root, f"{side}_{js}")
            if j is None:
                continue
            lim = j.find("limit")
            if lim is None:
                continue
            eff = float(lim.get("effort")) * p["effort_scale"]
            if js in FOREARM_JOINTS:
                eff *= p["effort_fore"]
            lim.set("effort", f"{eff:.6g}")
            lim.set("velocity", f"{float(lim.get('velocity')) * p['vel_scale']:.6g}")
    for js, (lo_deg, hi_deg) in p.get("rom", {}).items():
        j = jname(root, f"right_{js}")
        if j is None:
            continue
        lim = j.find("limit")
        lim.set("lower", f"{math.radians(lo_deg):.6g}")
        lim.set("upper", f"{math.radians(hi_deg):.6g}")

def jname(root, name):
    for j in root.findall("joint"):
        if j.get("name") == name:
            return j
    return None

def lname(root, name):
    for l in root.findall("link"):
        if l.get("name") == name:
            return l
    return None

def generate(species, p):
    tree = ET.parse(BASE)
    root = tree.getroot()
    root.set("name", f"{species}_upperlimb")

    r_hum = p["humerus"] / HUM_REF["humerus"]
    r_rad = p["radius"] / HUM_REF["radius"]
    r_cla = p["clavicle"] / HUM_REF["clavicle"]
    r_hand = p["hand"] / HUM_REF["hand"]
    body_s = (p["body_mass"] / HUM_REF["body_mass"]) ** (1.0 / 3.0)
    torsion_off = TORSION_SIGN * math.radians(p["torsion_deg"] - HUM_REF["torsion_deg"])
    glen = math.radians(p["glenoid_tilt_deg"])

    g_upper = math.sqrt(max(p["m_upper"] / 1.80, 1e-6) / max(r_hum, 1e-6))
    g_fore = math.sqrt(max(p["m_fore"] / 1.28, 1e-6) / max(r_rad, 1e-6))
    g_hand = math.sqrt(max(p["m_hand"] / 0.45, 1e-6) / max(r_hand, 1e-6))

    for side in ("left", "right"):
        elbow = jname(root, f"{side}_elbow_Z")
        o = _f(elbow.find("origin").get("xyz"))
        o[1] *= r_hum
        set_origin(elbow, xyz=o, rpy=[0.0, torsion_off, 0.0])

        wrist = jname(root, f"{side}_wrist_Z")
        o = _f(wrist.find("origin").get("xyz"))
        o[1] *= r_rad
        set_origin(wrist, xyz=o)

        sh = jname(root, f"{side}_shoulder_Z")
        o = _f(sh.find("origin").get("xyz"))
        o[2] *= r_cla
        o[1] *= body_s
        set_origin(sh, xyz=o, rpy=[0.0, glen, 0.0])

        scale_mass_inertia(lname(root, f"{side}_upperarm"), p["m_upper"] / 1.80)
        scale_mass_inertia(lname(root, f"{side}_lowerarm"), p["m_fore"] / 1.28)
        scale_mass_inertia(lname(root, f"{side}_hand"), p["m_hand"] / 0.45)
        hi = lname(root, f"{side}_hand").find("inertial/origin")
        ho = _f(hi.get("xyz")); ho[1] *= r_hand; hi.set("xyz", _s(ho))

        for l, sy, g in ((f"{side}_upperarm", r_hum, g_upper),
                         (f"{side}_lowerarm", r_rad, g_fore),
                         (f"{side}_hand", r_hand, g_hand)):
            link = lname(root, l)
            for vis in link.findall("visual"):
                mesh = vis.find("geometry/mesh")
                if mesh is None:
                    continue
                fn = mesh.get("filename").lower()
                if "upperarm" in fn or "lowerarm" in fn or "hand_mesh" in fn:
                    seg = "upperarm" if "upperarm" in fn else (
                          "lowerarm" if "lowerarm" in fn else "hand")
                    cand = MESHDIR_GEN / f"{species}_{seg}.STL"
                    if cand.exists():
                        mesh.set("filename", str(cand))
                    scale_visual_meshes_one(mesh, sy=sy, girth=g)
                else:
                    scale_visual_meshes_one(mesh, iso=body_s)

    AXIAL_Y = ["middle_lumbar_Z", "middle_thoracic_Z", "middle_cervical_Z",
               "left_clavicle_joint_X", "right_clavicle_joint_X",
               "left_hip_Z", "right_hip_Z", "left_knee_Z", "right_knee_Z",
               "left_ankle_Z", "right_ankle_Z"]
    for jn in AXIAL_Y:
        j = jname(root, jn)
        if j is None:
            continue
        o = _f(j.find("origin").get("xyz"))
        o[1] *= body_s
        set_origin(j, xyz=o)

    for lk in root.findall("link"):
        nm = lk.get("name")
        for vis in lk.findall("visual"):
            mesh = vis.find("geometry/mesh")
            if mesh is None:
                continue
            fn = mesh.get("filename").lower()
            if any(arm in nm for arm in ("upperarm", "lowerarm", "hand")):
                continue
            if "torso" in fn or "abdomen" in fn:
                scale_visual_meshes_one(mesh, iso=[body_s * p["torso_w"], body_s, body_s * p["torso_w"]])
            elif any(k in fn for k in ("head", "neck", "pelvis", "leg", "thight", "knee", "foot")):
                scale_visual_meshes_one(mesh, iso=body_s)
            else:
                continue
            vo = vis.find("origin")
            if vo is not None:
                vo.set("xyz", _s([c * body_s for c in _f(vo.get("xyz"))]))

    apply_limits(root, p)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"{species}.urdf"
    tree.write(out, encoding="unicode", xml_declaration=False)
    return out, dict(r_hum=r_hum, r_rad=r_rad, r_cla=r_cla, body_s=body_s,
                     torsion_off_deg=math.degrees(torsion_off))

def scale_visual_meshes_one(mesh, sy=1.0, girth=1.0, iso=None):
    sc = _f(mesh.get("scale"))
    if iso is not None:
        if isinstance(iso, (list, tuple)):
            sc = [sc[0] * iso[0], sc[1] * iso[1], sc[2] * iso[2]]
        else:
            sc = [c * iso for c in sc]
    else:
        sc = [sc[0] * girth, sc[1] * sy, sc[2] * girth]
    mesh.set("scale", _s(sc))

def verify(path):
    """Load in pinocchio, FK at neutral, report shoulder->elbow / elbow->wrist
    lengths and brachial index (forearm/humerus)."""
    try:
        import numpy as np
        import pinocchio as pin
    except Exception as e:
        return f"(skip verify: {e})"
    m = pin.buildModelFromUrdf(str(path))
    d = m.createData()
    q = pin.neutral(m)
    pin.forwardKinematics(m, d, q)
    pin.updateFramePlacements(m, d)
    def jpos(n):
        return d.oMi[m.getJointId(n)].translation
    hum = np.linalg.norm(jpos("right_elbow_Z") - jpos("right_shoulder_Z"))
    fore = np.linalg.norm(jpos("right_wrist_Z") - jpos("right_elbow_Z"))
    bi = 100.0 * fore / hum
    wz = m.upperPositionLimit[m.idx_qs[m.getJointId("right_wrist_Z")]]
    return (f"humerus={hum*1000:5.1f}mm  forearm={fore*1000:5.1f}mm  model_BI={bi:5.1f}"
            f"  wristZ_upper={math.degrees(wz):4.0f}deg")

if __name__ == "__main__":
    print(f"base: {BASE}\nout:  {OUT}\n")
    for sp, p in PARAMS.items():
        out, info = generate(sp, p)
        meas_bi = 100.0 * p["radius"] / p["humerus"]
        print(f"{sp:30s} -> {out.name}")
        print(f"    ratios hum={info['r_hum']:.3f} rad={info['r_rad']:.3f} "
              f"clav={info['r_cla']:.3f} body={info['body_s']:.3f} "
              f"torsion_off={info['torsion_off_deg']:+.0f}deg  "
              f"effort x{p['effort_scale']:.2f}(fore x{p['effort_fore']:.2f})  vel x{p['vel_scale']:.2f}")
        print(f"    {verify(out)}   (measured BI={meas_bi:.1f})")
