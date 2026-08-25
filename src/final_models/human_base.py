"""HumanBase — shared logic for all human-with-tool model classes."""
from subject_lookup import canonical_subject, disk_name
from repo_paths import REPO
from MPPI import MPPIController
import mujoco
from mujoco import viewer
import xml.etree.ElementTree as ET
import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import meshcat.geometry as g
import meshcat.transformations as tf
from pinocchio.robot_wrapper import RobotWrapper
from utils_slice_trajectories import remove_legs_from_visual_model
import os
import math
from cost_features import ResidualModelJointAcceleration
from utils_model_residuals import *
import mujoco.viewer
import mim_solvers
import imageio
import hppfcl

class HumanBase:
    """Shared base for MPPI and Crocoddyl solver families.

    Holds: model loading, rail init, limits, stick & rock setup, contact frame,
    viz helpers, solver-weight updaters, trajectory-feature getters.
    Solver-specific setup and solve live in subclasses' _post_init and solve.
    """

    def __init__(self, args):
        self.args = args
        self.date = args.get('date')
        self.mu = args['mu']
        self.target_force = args['target_force']
        self.subject_id = canonical_subject(args.get('subject_id', 'S3'))
        self.urdf_path = args.get('urdf_path', None)
        self.keep_urdf_inertias = args.get('keep_urdf_inertias', False)
        self.keep_species_limits = args.get('keep_species_limits', False)
        self.absolute_stick = args.get('absolute_stick', False)

        self.pin_model, self.visual_model, self.collision_model, \
        self.mj_model, self.mj_data, self.q0 = self.load_robot_and_tool()

        temp_data = self.pin_model.createData()
        self.tool_frame_id = self.pin_model.getFrameId("rock_frame")
        
        pin.framesForwardKinematics(self.pin_model, temp_data, self.q0)
        pin.updateFramePlacements(self.pin_model, temp_data)
        
        self.p_start_world = temp_data.oMf[self.tool_frame_id].translation.copy()
        
        q_traj_arg = self.args.get('q_traj', None)
        if 'p_start' in self.args and 'p_end' in self.args:
            self.p_start_world = np.array(self.args['p_start'])
            self.p_end_world   = np.array(self.args['p_end'])

        else:
            self.p_start_world, self.p_end_world = self.estimate_rail_from_traj(q_traj_arg, temp_data)

        self.visual_model = remove_legs_from_visual_model(self.visual_model)
        self.nq = self.pin_model.nq
        self.nv = self.pin_model.nv
        self.nu = self.nv
        self.nx = self.nq + self.nv

        self._setup_limits()

        _arm = float(self.args.get('armature', 0.0))
        if _arm > 0.0:
            self.pin_model.armature = np.full(self.nv, _arm, dtype=float)

        pin.framesForwardKinematics(self.pin_model, temp_data, self.q0)
        pin.updateFramePlacements(self.pin_model, temp_data)

        self._contact_anchor_idx = 0
        _cw = self.args.get('contact_window', None)
        if _cw is not None and q_traj_arg is not None:
            _qt = np.asarray(q_traj_arg, dtype=float); _Tf = len(_qt) - 1
            _s = float(_cw[0])
            _step = int(round(_s * _Tf)) if 0.0 <= _s <= 1.0 else int(round(_s))
            self._contact_anchor_idx = int(np.clip(_step, 0, _Tf))
        self.q_anchor = (np.asarray(q_traj_arg, dtype=float)[self._contact_anchor_idx]
                         if q_traj_arg is not None else self.q0)

        self.add_stick_and_finalize()
        # do NOT elongate it onto the rail. For CSQP the contact is defined on the
        if self.args.get('elongate_rock', False):
            self.elongate_rock_to_contact()

        self.contact_frame_id = self.pin_model.getFrameId(self.args['contact'])

        pin.framesForwardKinematics(self.pin_model, self.pin_data, self.q0)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        
        if self.absolute_stick and 'p_start' in self.args and 'p_end' in self.args:
            # Keep the shared world stick; do NOT slide it onto this model's hand.
            self.p_start_world = np.array(self.args['p_start'], dtype=float)
            self.p_end_world = np.array(self.args['p_end'], dtype=float)

        else:
            rail_vec = self.p_end_world - self.p_start_world
            _tmp = self.pin_model.createData()
            pin.framesForwardKinematics(self.pin_model, _tmp, self.q_anchor)
            pin.updateFramePlacements(self.pin_model, _tmp)
            self.p_start_world = _tmp.oMf[self.contact_frame_id].translation.copy()
            self.p_end_world = self.p_start_world + rail_vec

        if (self.args.get('rail_from_contact', False) and q_traj_arg is not None
                and not self.absolute_stick):
            _pc_s, _pc_e = self.estimate_rail_from_traj(
                q_traj_arg, self.pin_model.createData(),
                frame_id=self.contact_frame_id)
            _rail_vec_c = _pc_e - _pc_s
            _tmp2 = self.pin_model.createData()
            pin.framesForwardKinematics(self.pin_model, _tmp2, self.q_anchor)
            pin.updateFramePlacements(self.pin_model, _tmp2)
            self.p_start_world = _tmp2.oMf[self.contact_frame_id].translation.copy()
            self.p_end_world = self.p_start_world + _rail_vec_c

        # forcing v0=0 makes the OCP jump rest->moving at node 0 and blows up
        _q_traj = args.get('q_traj', None)
        if _q_traj is not None and len(np.asarray(_q_traj, dtype=float)) >= 2:
            _qt = np.asarray(_q_traj, dtype=float)
            _v0 = (_qt[1] - _qt[0]) / float(args.get('dt', 0.01))
            self.v0 = (_v0[:self.nv] if len(_v0) >= self.nv
                       else np.concatenate([_v0, np.zeros(self.nv - len(_v0))]))
        else:
            self.v0 = np.zeros(self.nv)
        self.x0 = np.concatenate((self.q0, self.v0))

        self.dt = args.get('dt', 0.01)
        self.T  = args.get('T', 500)
        self.stick_static = args.get('stick_static', True)

        self._post_init(args)

    def _post_init(self, args):
        """Solver-specific __init__ tail. Override in subclass."""
        pass

    def _ensure_full_viz(self, open_browser: bool = True):
        """Lazily build a Meshcat visualizer for the full-body model.

        Attaches the rock to the right_wrist_X joint the same way
        load_robot_and_tool does it on the reduced model (fixed joint at
        translation [0.035, -0.12, 0] relative to the wrist joint, rock
        URDF geometry added on top).
        """
        if getattr(self, "_full_viz", None) is not None:
            return self._full_viz
        import os as _os
        import pinocchio as _pin
        from utils_slice_trajectories import remove_legs_from_visual_model

        remove_legs_from_visual_model(self.full_visual_model)

        self._full_viz = _pin.visualize.MeshcatVisualizer(
            self.full_pin_model, self.full_collision_model, self.full_visual_model,
        )
        self._full_viz.initViewer(open=open_browser)
        self._full_viz.loadViewerModel()
        self._full_pin_data = self.full_pin_model.createData()
        return self._full_viz

    def _set_stick_transform(self, p_center: np.ndarray):
        """Override the per-frame meshcat transform of the pinocchio stick
        geometry. viz.display(q) sets this to the static placement each
        frame, so call this AFTER display(q) to animate the stick."""
        if getattr(self, "_full_viz", None) is None:
            return
        R = getattr(self, "_full_stick_R", None)
        if R is None:
            return
        T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p_center
        self._full_viz.viewer["pinocchio/visuals/stick_visual_full"].set_transform(T)

    def _attach_rock_to_full_model(self):
        """Attach the rock as a fixed joint on full_pin_model (idempotent).
        Uses the REDUCED model's rock_fixed_joint placement if available, so
        the elongate_rock_to_contact shift carries over to the full viewer.
        Also adds rock_frame and a red contact-point sphere at the same
        offset used by the reduced model's rock_contact_point frame."""
        import os as _os
        import hppfcl as _hppfcl
        if self.full_pin_model.existJointName("rock_fixed_joint_full"):
            return
        if not self.full_pin_model.existJointName("right_wrist_X"):
            return
        _here = _os.path.dirname(_os.path.abspath(__file__))
        _root = _here
        while _root and not _os.path.isdir(_os.path.join(_root, "human_model")):
            parent = _os.path.dirname(_root)
            if parent == _root:
                _root = _here
                break
            _root = parent
        tool_urdf = _os.path.join(_root, "human_model/tool/rock.urdf")
        if not _os.path.isfile(tool_urdf):
            return
        rock_model  = pin.buildModelFromUrdf(tool_urdf)
        rock_visual = pin.buildGeomFromUrdf(rock_model, tool_urdf, pin.GeometryType.VISUAL, package_dirs=_os.path.join(_root, "human_model"))
        wrist_jid = self.full_pin_model.getJointId("right_wrist_X")

        placement_rock = pin.SE3.Identity()
        placement_rock.translation = np.array([0.05, -0.12, 0.0])

        rock_jid = self.full_pin_model.addJoint(
            wrist_jid, pin.JointModel(), placement_rock, "rock_fixed_joint_full"
        )
        self.full_pin_model.appendBodyToJoint(
            rock_jid,
            rock_model.inertias[0] if len(rock_model.inertias) > 0 else pin.Inertia.Zero(),
            pin.SE3.Identity(),
        )
        self.full_pin_model.addFrame(pin.Frame(
            "rock_frame", rock_jid, 0,
            pin.SE3.Identity(), pin.FrameType.BODY,
        ))
        for geom in rock_visual.geometryObjects:
            g_obj = geom.copy()
            g_obj.parentJoint = rock_jid
            g_obj.placement   = geom.placement
            self.full_visual_model.addGeometryObject(g_obj)

        contact_name = self.args.get('contact', 'rock_contact_point')
        contact_offset_local = None
        if self.pin_model.existFrame(contact_name):
            f = self.pin_model.frames[self.pin_model.getFrameId(contact_name)]

            contact_offset_local = f.placement.copy()
        if contact_offset_local is not None and \
                not self.full_visual_model.existGeometryName("contact_point_visual"):
            sphere = _hppfcl.Sphere(0.008)
            sphere_geom = pin.GeometryObject(
                "contact_point_visual", rock_jid, sphere, contact_offset_local
            )
            sphere_geom.meshColor = np.array([1.0, 0.0, 0.0, 1.0])
            self.full_visual_model.addGeometryObject(sphere_geom)
            self._full_contact_offset_local = contact_offset_local

    def _setup_full_stick(self, q_traj_full):
        """Add a stick GeometryObject to full_visual_model whose placement
        is computed against the FULL-body trajectory.

        The __init__ stick in self.visual_model uses a rail estimated with
        the reduced model (torso locked at q0_full). For the full-body
        viewer — where the torso animates — we need a rail computed from
        the actual full-body contact-frame positions so the stick lines up
        with where the arm reaches in world.

        Must be called before _ensure_full_viz (loadViewerModel picks up
        geometries present at that moment).
        """
        import hppfcl as _hppfcl
        if getattr(self, "_full_stick_done", False):
            return

        self._attach_rock_to_full_model()

        q_traj = self._pad_for_rock(np.asarray(q_traj_full, dtype=np.float64))

        if self.full_pin_model.existFrame(self.args.get('contact', 'rock_contact_point')):
            contact_name = self.args['contact']
        elif self.full_pin_model.existFrame('rock_contact_point'):
            contact_name = 'rock_contact_point'
        else:
            contact_name = None
        rock_name = 'rock_frame' if self.full_pin_model.existFrame('rock_frame') else None

        data = self.full_pin_model.createData()
        contacts = []
        for q in q_traj:
            pin.framesForwardKinematics(self.full_pin_model, data, q)
            pin.updateFramePlacements(self.full_pin_model, data)
            if contact_name is not None:
                fid = self.full_pin_model.getFrameId(contact_name)
            elif rock_name is not None:
                fid = self.full_pin_model.getFrameId(rock_name)
            else:
                return  
            contacts.append(data.oMf[fid].translation.copy())
        contacts = np.array(contacts)

        rail_dir_red = self.p_end_world - self.p_start_world
        rail_len_red = float(np.linalg.norm(rail_dir_red))
        if rail_len_red < 1e-6:
            from sklearn.decomposition import PCA
            rail_axis = PCA(n_components=3).fit(contacts).components_[0]
        else:
            rail_axis = rail_dir_red / rail_len_red

        try:
            pin.framesForwardKinematics(self.full_pin_model, data, q_traj[0])
            pin.updateFramePlacements(self.full_pin_model, data)
            elbow_jid = self.full_pin_model.getJointId("left_elbow_Y")
            wrist_jid = self.full_pin_model.getJointId("left_wrist_X")
            p_elbow = data.oMi[elbow_jid].translation.copy()
            p_wrist = data.oMi[wrist_jid].translation.copy()
            forearm = p_wrist - p_elbow
            n = np.linalg.norm(forearm)
            if n > 1e-6:
                forearm_unit = forearm / n
                cos_ang = abs(float(np.dot(rail_axis, forearm_unit)))
                ang_deg = float(np.degrees(np.arccos(min(1.0, cos_ang))))

                if self.args.get('forearm_swap', False) and cos_ang < 0.866:

                    rail_axis = forearm_unit
                    if np.dot(rail_axis, rail_dir_red) < 0:
                        rail_axis = -rail_axis
                elif cos_ang < 0.866:
                    pass
        except Exception as e:
            pass

        t_stick_final = self.p_stick.copy()
        rail_vec_red = self.p_end_world - self.p_start_world
        axis_len = float(np.linalg.norm(rail_vec_red))
        if axis_len < 1e-6:
            return
        z_axis = rail_vec_red / axis_len
        x_axis = np.array([1.0, 0.0, 0.0]) if abs(z_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        y_axis = np.cross(z_axis, x_axis); y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        R_stick = np.column_stack([x_axis, y_axis, z_axis])

        tool_length = self.args.get('tool_length', None)
        if tool_length is not None:
            stick_half_len = float(tool_length) / 2.0
        else:
            stick_half_len = axis_len

        self.p_stick_full    = t_stick_final
        self._full_stick_R   = R_stick
        self._full_stick_len = stick_half_len

        stick_shape = _hppfcl.Cylinder(0.02, stick_half_len)
        stick_vis = pin.GeometryObject(
            "stick_visual_full", 0, stick_shape, pin.SE3(R_stick, t_stick_final),
        )
        stick_vis.meshColor = np.array([0.55, 0.40, 0.25, 1.0])
        if not self.full_visual_model.existGeometryName("stick_visual_full"):
            self.full_visual_model.addGeometryObject(stick_vis)

        p_c0 = contacts[0]
        deltas = contacts - p_c0
        deltas_perp = deltas - (deltas @ z_axis)[:, None] * z_axis
        self.stick_traj_full = t_stick_final + deltas_perp

        self._elongate_rock_on_full_model(q_traj[0], z_axis)

        self._full_stick_done = True

    def _elongate_rock_on_full_model(self, q_full0, z_axis):
        """Shift rock_fixed_joint_full's local placement so the rock's
        contact point reaches p_stick_full + radius*normal at q_full0.

        Mirrors elongate_rock_to_contact but uses the full_pin_model's
        wrist rotation (which depends on the IK-derived torso)."""
        if not self.full_pin_model.existJointName("rock_fixed_joint_full"):
            return
        rock_jid_full = self.full_pin_model.getJointId("rock_fixed_joint_full")
        hand_jid_full = self.full_pin_model.parents[rock_jid_full]

        data = self.full_pin_model.createData()
        pin.framesForwardKinematics(self.full_pin_model, data, q_full0)
        pin.updateFramePlacements(self.full_pin_model, data)

        offset_local = getattr(self, "_full_contact_offset_local", None)
        rock_pose = data.oMi[rock_jid_full]
        if offset_local is not None:
            contact_pos = (rock_pose * offset_local).translation
        else:
            contact_pos = rock_pose.translation

        radial = contact_pos - self.p_stick_full
        radial -= np.dot(radial, z_axis) * z_axis
        rn = np.linalg.norm(radial)
        if rn < 1e-6:
            return
        normal = radial / rn
        stick_radius = 0.02
        target = self.p_stick_full + stick_radius * normal

        gap_world = target - contact_pos
        gap_world -= np.dot(gap_world, z_axis) * z_axis

        R_hand = data.oMi[hand_jid_full].rotation
        gap_hand = R_hand.T @ gap_world

        self.full_pin_model.jointPlacements[rock_jid_full].translation += gap_hand

    def _update_rock_from_q(self, q_full: np.ndarray):
        """Deprecated — the rock is now a fixed joint on the full model, so
        viz.display(q) handles its placement automatically. Kept as a no-op
        so existing call sites don't break."""
        return

    def _pad_for_rock(self, q: np.ndarray) -> np.ndarray:
        """If the full model has the rock fixed-joint attached (nq+1), pad q
        with a trailing zero so pin.forwardKinematics accepts it."""
        q = np.asarray(q, dtype=np.float64)
        nq = self.full_pin_model.nq
        if q.ndim == 1 and q.shape[0] == nq - 1:
            return np.concatenate([q, np.zeros(1)])
        if q.ndim == 2 and q.shape[1] == nq - 1:
            return np.hstack([q, np.zeros((q.shape[0], 1))])
        return q

    def display_full(self, q_full):
        """Display a single full-body configuration (stick stays at p_stick)."""
        viz = self._ensure_full_viz()
        q = self._pad_for_rock(q_full)
        viz.display(q)

    def animate_full(self, q_traj_full, dt=None, loop=False, stick_traj=None):
        """Animate a full-model trajectory.

        q_traj_full : (T, nq_full)
        stick_traj  : (T, 3) of stick center positions per frame. Defaults to
                      self.stick_traj when the model was built with
                      stick_static=False; otherwise the stick stays put.
        """
        import time as _time

        self._setup_full_stick(q_traj_full)
        viz = self._ensure_full_viz()
        dt  = dt or float(self.dt)
        q_traj = self._pad_for_rock(q_traj_full)

        traj = stick_traj
        if traj is None:
            traj = getattr(self, "stick_traj_full", None)

        def stick_center_at(t_idx: int):
            if traj is None:
                return getattr(self, "p_stick_full", self.p_stick)
            return np.asarray(traj[min(t_idx, len(traj) - 1)], dtype=np.float64)

        try:
            while True:
                for t, q in enumerate(q_traj):
                    viz.display(q)
                    self._set_stick_transform(stick_center_at(t))
                    self._update_rock_from_q(q)
                    _time.sleep(dt)
                if not loop:
                    break
                _time.sleep(0.5)
        except KeyboardInterrupt:
            pass

    def build_full_q_from_cycle(self, arm_q_traj, t_arm, ik_csv_path,
                                arm_joint_names=None):
        """Turn an arm-only cycle (7 joints × T frames) into a full-body q
        trajectory by pulling non-arm joints from the IK CSV at the matching
        timestamps.

        arm_q_traj      : (T, 7) arm joint values from the cycle npz
        t_arm           : (T,)   seconds; timestamps of each arm frame
        ik_csv_path     : str or Path to the subject's ik_joint_angles.csv
        arm_joint_names : list of 7 URDF joint names (defaults to the
                          canonical right-arm set)
        """
        import pandas as _pd
        if arm_joint_names is None:
            arm_joint_names = [
                "middle_thoracic_X",
                "right_clavicle_joint_X",
                "right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y",
                "right_elbow_Z", "right_elbow_Y",
                "right_wrist_Z", "right_wrist_X",
            ]
        df    = _pd.read_csv(str(ik_csv_path))
        q_all = df.drop(columns=["Relative_Time[s]", "Capture_Start_Time"]) \
                  .to_numpy().astype(np.float64)
        t_all = df["Relative_Time[s]"].to_numpy().astype(np.float64)

        idx_q = [self.full_pin_model.joints[self.full_pin_model.getJointId(n)].idx_q
                 for n in arm_joint_names]

        arm_q_traj = np.asarray(arm_q_traj, dtype=np.float64)
        t_arm      = np.asarray(t_arm,      dtype=np.float64)
        out = np.empty((len(arm_q_traj), q_all.shape[1]), dtype=np.float64)
        for k, (arm, t) in enumerate(zip(arm_q_traj, t_arm)):
            j = int(np.abs(t_all - t).argmin())
            q = q_all[j].copy()
            for i, val in zip(idx_q, arm):
                q[i] = val
            out[k] = q
        return out

    def load_robot_and_tool(self):
        """Same as model_mocap.py"""
        _here = os.path.dirname(os.path.abspath(__file__))
        script_directory = _here
        while script_directory and not os.path.isdir(
            os.path.join(script_directory, "config", "scaled_registered_models")
        ):
            parent = os.path.dirname(script_directory)
            if parent == script_directory:
                script_directory = _here
                break
            script_directory = parent
        if self.urdf_path is not None:
            urdf_path = self.urdf_path
        else:
            urdf_path = os.path.join(
                script_directory,
                f"config/scaled_registered_models/{self.date}/{disk_name(self.subject_id)}.urdf"
            )
        model_path = os.path.join(script_directory, "human_model/")
        tool_urdf_path = os.path.join(script_directory, "human_model/tool/rock.urdf")
        xml_name = os.path.join(script_directory, "human_model/xml/human_pinned.xml")

        robot = RobotWrapper.BuildFromURDF(urdf_path, model_path, pin.JointModelFreeFlyer())
        model = robot.model
        visual_model = robot.visual_model
        collision_model = robot.collision_model

        good_urdf_path = str(REPO / "human_model/urdf/human.urdf")
        good_robot = RobotWrapper.BuildFromURDF(good_urdf_path, model_path, pin.JointModelFreeFlyer())
        good_model = good_robot.model

        for i in range(1, len(model.names)):
            joint_name = model.names[i]
            if not self.keep_urdf_inertias and good_model.existJointName(joint_name):
                good_id = good_model.getJointId(joint_name)
                model.inertias[i] = good_model.inertias[good_id].copy()

            I = model.inertias[i]
            Im = np.asarray(I.inertia)
            is_valid = bool(
                np.isfinite(I.mass) and I.mass >= 1e-4
                and np.all(np.isfinite(Im))
                and np.min(np.linalg.eigvalsh(Im)) > -1e-9
            )

            if not is_valid or model.inertias[i].mass < 1e-4:
                safe_mass = 1e-3
                safe_com = np.zeros(3)
                safe_inertia_matrix = np.eye(3) * 1e-5
                model.inertias[i] = pin.Inertia(safe_mass, safe_com, safe_inertia_matrix)

        active_joints_names = [
            "universe",
            "middle_thoracic_X",
            "right_clavicle_joint_X",
            "right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y",
            "right_elbow_Z", "right_elbow_Y", "right_wrist_Z", "right_wrist_X"
        ]

        active_joint_ids = []
        for jn in active_joints_names:
            if model.existJointName(jn):
                active_joint_ids.append(model.getJointId(jn))

        joints_to_lock = [
            model.getJointId(jn)
            for jn in model.names
            if model.existJointName(jn) and model.getJointId(jn) not in active_joint_ids
        ]

        q = pin.neutral(model)
        q[2] = 0.65
        rotation_matrix = pin.utils.rpyToMatrix(math.pi/2, 0, 0)
        quaternion = pin.Quaternion(rotation_matrix)
        q[3:7] = [quaternion.x, quaternion.y, quaternion.z, quaternion.w]

        q0_full = self.args.get('q0_full', None)
        if q0_full is not None and len(q0_full) == len(pin.neutral(model)):
            q = np.array(q0_full, dtype=np.float64)
        else:
            q0_reduced = self.args.get('q0', None)
            if q0_reduced is not None and len(q0_reduced) == len(active_joint_ids):
                for i, jid in enumerate(active_joint_ids[1:], start=0):
                    joint = model.joints[jid]
                    q[joint.idx_q] = q0_reduced[i]
            else:
                q[7], q[10], q[37], q[40] = [math.radians(90)] * 4
                q[17] = math.radians(15)
                q[19] = math.radians(-10)
                q[20] = math.radians(5)
                q[22] = math.radians(80)
                q[21] = math.radians(40)

        if len(collision_model.geometryObjects) == 0:
            for visual_obj in visual_model.geometryObjects:
                col_obj = pin.GeometryObject(
                    visual_obj.name + "_col", visual_obj.parentFrame,
                    visual_obj.parentJoint, visual_obj.geometry, visual_obj.placement
                )
                col_obj.meshColor = np.array([1.0, 0.0, 0.0, 0.5])
                collision_model.addGeometryObject(col_obj)

        self.q0_full = q.copy()
        self.full_model_for_mapping = model.copy()
        self.full_pin_model       = model.copy()
        self.full_visual_model    = visual_model.copy()
        self.full_collision_model = collision_model.copy()

        reduced_model, reduced_geom_models = pin.buildReducedModel(
            model,
            list_of_geom_models=[visual_model, collision_model],
            list_of_joints_to_lock=joints_to_lock,
            reference_configuration=self.q0_full
        )
        reduced_visual, reduced_collision = reduced_geom_models

        combined_model = reduced_model.copy()
        combined_visual = reduced_visual.copy()
        combined_collision = reduced_collision.copy()

        rock_model = pin.buildModelFromUrdf(tool_urdf_path)
        rock_visual = pin.buildGeomFromUrdf(rock_model, tool_urdf_path, pin.GeometryType.VISUAL, package_dirs=model_path)
        rock_collision = pin.buildGeomFromUrdf(rock_model, tool_urdf_path, pin.GeometryType.COLLISION, package_dirs=model_path)

        parent_joint_id = 9
        placement_rock = pin.SE3.Identity()
        placement_rock.translation = np.array([0.05, -0.12, 0.0])

        fixed_joint_id = combined_model.addJoint(
            parent_joint_id, pin.JointModel(), placement_rock, "rock_fixed_joint"
        )
        combined_model.appendBodyToJoint(
            fixed_joint_id,
            rock_model.inertias[0] if len(rock_model.inertias) > 0 else pin.Inertia.Zero(),
            pin.SE3.Identity()
        )
        combined_model.addFrame(pin.Frame(
            "rock_frame", fixed_joint_id, combined_model.getFrameId("rock_fixed_joint"),
            pin.SE3.Identity(), pin.FrameType.BODY
        ))

        for geom in rock_visual.geometryObjects:
            g_obj = geom.copy()
            g_obj.parentJoint = fixed_joint_id
            g_obj.placement = geom.placement
            combined_visual.addGeometryObject(g_obj)

        for geom in rock_collision.geometryObjects:
            g_obj = geom.copy()
            g_obj.parentJoint = fixed_joint_id
            g_obj.placement = geom.placement
            combined_collision.addGeometryObject(g_obj)

        try:
            mj_model = mujoco.MjModel.from_xml_path(xml_name)
            mj_data = mujoco.MjData(mj_model)
        except Exception as e:
            mj_model, mj_data = None, None

        q0_reduced = self.args.get('q0', None)
        return combined_model, combined_visual, combined_collision, None, None, q0_reduced

    def add_stick_and_finalize(self):
        """Same as model_mocap.py - adds stick visualization and contact frame"""
        import hppfcl
        
        t_stick_mid = (self.p_start_world + self.p_end_world) / 2.0
        axis_vector = self.p_end_world - self.p_start_world
        axis_len = np.linalg.norm(axis_vector)
        z_axis = axis_vector / axis_len
        
        x_axis = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)

        temp_data_stick = self.pin_model.createData()
        pin.framesForwardKinematics(self.pin_model, temp_data_stick, self.q0)
        pin.updateFramePlacements(self.pin_model, temp_data_stick)
        stone_frame_id_tmp = self.pin_model.getFrameId("rock_frame")
        stone_pos_tmp = temp_data_stick.oMf[stone_frame_id_tmp].translation

        radial = self.args.get('stick_radial', None)
        if radial is not None:
            radial = np.asarray(radial, dtype=float).copy()
        else:
            radial = t_stick_mid - stone_pos_tmp
        radial -= np.dot(radial, z_axis) * z_axis
        norm_r = np.linalg.norm(radial)
        if norm_r > 1e-6:
            radial /= norm_r
        else:
            radial = np.array([0.0, -1.0, 0.0])
        self.stick_radial = radial.copy()

        stick_radius = float(self.args.get('stick_radius', 0.02))
        offset_dist = float(self.args.get('stick_offset', stick_radius))
        t_stick = t_stick_mid + radial * offset_dist

        R_stick = np.column_stack([x_axis, y_axis, z_axis])

        tool_length = self.args.get('tool_length', None)
        if tool_length is not None:
            stick_half_len = float(tool_length) / 2.0
        else:
            stick_half_len = axis_len
        t_stick_final = t_stick
        stick_shape = hppfcl.Cylinder(stick_radius, stick_half_len)
        stick_placement = pin.SE3(R_stick, t_stick_final)

        stick_vis = pin.GeometryObject("stick_visual", 0, stick_shape, stick_placement)
        stick_vis.meshColor = np.array([0.55, 0.40, 0.25, 1.0])
        self.visual_model.addGeometryObject(stick_vis)

        stick_col = pin.GeometryObject("stick_collision", 0, stick_shape, stick_placement)
        stick_col.meshColor = np.array([0.55, 0.40, 0.25, 1.0])
        self.collision_model.addGeometryObject(stick_col)

        self.collision_model.createData()
        def get_geom_id(name):
            return self.collision_model.getGeometryId(name) if self.collision_model.existGeometryName(name) else None

        stone_id = get_geom_id('rock_link_0')
        stick_id = get_geom_id('stick_collision')
        hand_id  = get_geom_id('right_hand_0_col')
        wrist_id = get_geom_id('right_lowerarm_1_col')

        pairs = [(stone_id, stick_id), (stick_id, hand_id), (stick_id, wrist_id)]
        for id1, id2 in pairs:
            if id1 is not None and id2 is not None:
                self.collision_model.addCollisionPair(pin.CollisionPair(id1, id2))

        temp_data = self.pin_model.createData()
        pin.framesForwardKinematics(self.pin_model, temp_data, self.q0)
        pin.updateFramePlacements(self.pin_model, temp_data)

        stone_frame_id = self.pin_model.getFrameId("rock_frame")
        stone_pose = temp_data.oMf[stone_frame_id]
        stone_pos_world = stone_pose.translation

        stone_in_stick_frame = stick_placement.inverse().act(stone_pos_world)
        vec_to_stick_local = np.array([-stone_in_stick_frame[0], -stone_in_stick_frame[1], 0])
        norm_vec = np.linalg.norm(vec_to_stick_local)
        
        if norm_vec < 1e-6:
            vec_to_stick_local = np.array([0, -1, 0])
        else:
            vec_to_stick_local = vec_to_stick_local / norm_vec

        vec_to_stick_world = stick_placement.rotation @ vec_to_stick_local
        vec_offset_stone_local = (stone_pose.rotation.T) @ vec_to_stick_world

        offset_dist = 0.010
        contact_point_translation = vec_offset_stone_local * offset_dist

        contact_frame_name = "rock_contact_point"
        self.args['contact'] = contact_frame_name

        if not self.pin_model.existFrame(contact_frame_name):
            parent_frame = self.pin_model.frames[stone_frame_id]
            parent_joint_id = parent_frame.parentJoint
            vec_along_stick = (self.p_end_world - self.p_start_world)
            vec_along_stick = vec_along_stick / np.linalg.norm(vec_along_stick)
            
            vec_diff = self.p_start_world - t_stick
            vec_radial = vec_diff - np.dot(vec_diff, vec_along_stick) * vec_along_stick
            vec_radial = vec_radial / np.linalg.norm(vec_radial)
            
            R_surface = np.eye(3)
            R_surface[:, 0] = -vec_along_stick
            R_surface[:, 2] = vec_radial
            R_surface[:, 1] = np.cross(vec_radial, -vec_along_stick)
            
            stone_R_world = temp_data.oMf[stone_frame_id].rotation
            R_contact_local = stone_R_world.T @ R_surface
                
            placement_offset = pin.SE3(R_contact_local, contact_point_translation)
            final_placement = parent_frame.placement * placement_offset
            
            new_frame = pin.Frame(
                contact_frame_name, parent_joint_id, stone_frame_id,
                final_placement, pin.FrameType.OP_FRAME
            )
            self.pin_model.addFrame(new_frame)
        
        self.p_stick = t_stick_final.copy()
        self.stick_to_rock_normal = vec_radial.copy()
        self.pin_data = self.pin_model.createData()

    def elongate_rock_to_contact(self):
        """
        Instead of snapping the arm to the stick, shift the rock's placement
        in the hand frame so that the contact point reaches p_start_world at q0.
        """
        data = self.pin_model.createData()
        pin.framesForwardKinematics(self.pin_model, data, self.q_anchor)
        pin.updateFramePlacements(self.pin_model, data)

        contact_frame_id = self.pin_model.getFrameId(self.args['contact'])
        contact_pos = data.oMf[contact_frame_id].translation.copy()

        stick_radius = 0.02
        target = self.p_stick + stick_radius * self.stick_to_rock_normal

        gap_world_full = target - contact_pos
        stick_axis = self.p_end_world - self.p_start_world
        stick_axis = stick_axis / np.linalg.norm(stick_axis)
        gap_world = gap_world_full - np.dot(gap_world_full, stick_axis) * stick_axis

        rock_joint_id = self.pin_model.getJointId("rock_fixed_joint")
        hand_joint_id = self.pin_model.parents[rock_joint_id]
        R_hand = data.oMi[hand_joint_id].rotation
        gap_hand = R_hand.T @ gap_world

        self.pin_model.jointPlacements[rock_joint_id].translation += gap_hand

        self.pin_data = self.pin_model.createData()
        pin.framesForwardKinematics(self.pin_model, self.pin_data, self.q0)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

    def estimate_rail_from_traj(self, q_traj, temp_data, frame_id=None):
        """PCA-fit the rail line from the tool-frame trajectory.

        frame_id selects which frame's path to fit. Default None = the
        bootstrap rock_frame (rock CENTER), the only frame available before the
        contact frame is built. Pass self.contact_frame_id (rail_from_contact)
        to re-fit through the actual rock_contact_point path — the rock center
        and the tool tip differ by the 10 mm contact offset, and that offset
        ROTATES with the rock, so the two paths are not parallel; fitting the
        center pulls the rail off where the scrape physically happens.

        Default: over the WHOLE stroke. With args['windowed_rail']=True and a
        contact_window, fit over ONLY the contact-window segment — a long cycle's
        approach swing otherwise overshoots the rail (PCA gave 191-224mm rails vs
        ~60mm actual contact paths, so the demo lands 11-18mm off-rail and the OCP
        contorts/over-presses to reach it). Only affects sub-window subjects
        (S1/S2); a full [0,1] window (S3) is unchanged."""
        from sklearn.decomposition import PCA

        q_traj = np.asarray(q_traj, dtype=float)
        _cw = self.args.get('contact_window', None)
        if self.args.get('windowed_rail', False) and _cw is not None:
            _Tf = len(q_traj) - 1
            _to = (lambda v: int(round(v * _Tf)) if 0.0 <= float(v) <= 1.0
                   else int(round(float(v))))
            i0 = int(np.clip(_to(_cw[0]), 0, _Tf - 1))
            i1 = int(np.clip(_to(_cw[1]), i0 + 1, _Tf + 1))
            q_used = q_traj[i0:i1 + 1]
        else:
            q_used = q_traj

        contact_positions = []
        frame_id_for_est = self.tool_frame_id if frame_id is None else frame_id

        for q in q_used:
            pin.forwardKinematics(self.pin_model, temp_data, q)
            pin.updateFramePlacements(self.pin_model, temp_data)
            p = temp_data.oMf[frame_id_for_est].translation.copy()
            contact_positions.append(p)

        contact_positions = np.array(contact_positions)

        pca = PCA(n_components=3)
        pca.fit(contact_positions)
        rail_axis = pca.components_[0]
        rail_center = pca.mean_

        projections = (contact_positions - rail_center) @ rail_axis
        p_a = rail_center + projections.min() * rail_axis
        p_b = rail_center + projections.max() * rail_axis

        first_pos = contact_positions[0]
        if np.linalg.norm(first_pos - p_a) <= np.linalg.norm(first_pos - p_b):
            p_start, p_end = p_a, p_b
        else:
            p_start, p_end = p_b, p_a

        return p_start, p_end

    def _setup_limits(self):
        """Same as model_mocap.py"""
        combined_model = self.pin_model
        human_model_path = str(REPO / "human_model/urdf/human.urdf")
        good_model = pin.buildModelFromUrdf(human_model_path)

        for joint_name in (() if self.keep_species_limits else good_model.names):
            if combined_model.existJointName(joint_name):
                good_id = good_model.getJointId(joint_name)
                main_id = combined_model.getJointId(joint_name)

                joint_good = good_model.joints[good_id]
                joint_main = combined_model.joints[main_id]

                if joint_good.nq > 0:
                    idx_q_good = joint_good.idx_q
                    idx_q_main = joint_main.idx_q
                    combined_model.lowerPositionLimit[idx_q_main : idx_q_main + joint_main.nq] = \
                        good_model.lowerPositionLimit[idx_q_good : idx_q_good + joint_good.nq]
                    
                    combined_model.upperPositionLimit[idx_q_main : idx_q_main + joint_main.nq] = \
                        good_model.upperPositionLimit[idx_q_good : idx_q_good + joint_good.nq]

                if joint_good.nv > 0:
                    idx_v_good = joint_good.idx_v
                    idx_v_main = joint_main.idx_v
                    
                    v_limit = good_model.velocityLimit[idx_v_good : idx_v_good + joint_good.nv]
                    v_limit = np.where(v_limit <= 0, 10.0, v_limit)
                    
                    combined_model.velocityLimit[idx_v_main : idx_v_main + joint_main.nv] = v_limit

        q_min = combined_model.lowerPositionLimit.copy()
        q_max = combined_model.upperPositionLimit.copy()
        vel = combined_model.velocityLimit.copy()
        dq_lb = -np.abs(vel)
        dq_ub = np.abs(vel)

        _wh = float(self.args.get('wrist_hold_deg', 0.0))
        if _wh > 0.0:
            _q0 = np.asarray(self.args.get('q0', None), float) if self.args.get('q0', None) is not None else None
            _b = np.deg2rad(_wh)
            _held = []
            for _n in ('right_wrist_Z', 'right_wrist_X', 'right_wrist_Y'):
                if not combined_model.existJointName(_n):
                    continue
                _iq = int(combined_model.joints[combined_model.getJointId(_n)].idx_q)
                if _q0 is None or _iq >= len(_q0):
                    continue
                _c = float(_q0[_iq])
                q_min[_iq] = max(q_min[_iq], _c - _b)
                q_max[_iq] = min(q_max[_iq], _c + _b)
                _held.append(_n)
            if _held:
                print(f"[HumanCrocoddyl] wrist held within +-{_wh:.0f} deg of its start: {_held}")

        self.x_lb = np.concatenate([q_min, dq_lb])
        self.x_ub = np.concatenate([q_max, dq_ub])

    def search_valid_start_on_rail(self):
        """Same as model_mocap.py"""
        alphas = np.linspace(0.0, 0.5, 50)
        data = self.pin_model.createData()
        q_current = self.q0.copy()

        q_min = self.pin_model.lowerPositionLimit
        q_max = self.pin_model.upperPositionLimit

        has_ff = (self.pin_model.joints[1].shortname() == "JointModelFreeFlyer")
        start_idx = 7 if has_ff else 0

        contact_name = self.args.get('contact', 'rock_contact_point')
        if self.pin_model.existFrame(contact_name):
            frame_id = self.pin_model.getFrameId(contact_name)
        else:
            frame_id = self.tool_frame_id

        original_start = self.p_start_world.copy()
        rail_vector = self.p_end_world - original_start
        valid_found = False
        target_pos = original_start.copy()

        STICK_RADIUS = 0.02
        ROCK_RADIUS  = 0.02
        rail_dir = rail_vector / np.linalg.norm(rail_vector)
        pin.framesForwardKinematics(self.pin_model, data, q_current)
        pin.updateFramePlacements(self.pin_model, data)
        initial_rock_pos = data.oMf[frame_id].translation.copy()
        t_proj = np.dot(initial_rock_pos - original_start, rail_dir)
        nearest_on_axis = original_start + t_proj * rail_dir
        rock_to_axis = initial_rock_pos - nearest_on_axis
        normal_dist = np.linalg.norm(rock_to_axis)
        contact_normal = rock_to_axis / normal_dist if normal_dist > 1e-6 else np.array([0.0, 0.0, 1.0])
        surface_offset = contact_normal * (STICK_RADIUS + ROCK_RADIUS)

        for alpha in alphas:
            target_pos = original_start + alpha * rail_vector + surface_offset
            q_test = q_current.copy()

            for _ in range(50):
                pin.framesForwardKinematics(self.pin_model, data, q_test)
                pin.updateFramePlacements(self.pin_model, data)

                curr_pos = data.oMf[frame_id].translation
                err = curr_pos - target_pos

                if np.linalg.norm(err) < 1e-4:
                    break

                J = pin.computeFrameJacobian(
                    self.pin_model, data, q_test,
                    frame_id, pin.LOCAL_WORLD_ALIGNED
                )
                J_trans = J[:3, :]

                v = -J_trans.T @ np.linalg.inv(
                    J_trans @ J_trans.T + 1e-6 * np.eye(3)
                ) @ err

                q_next = pin.integrate(self.pin_model, q_test, v)
                q_next[start_idx:] = np.clip(
                    q_next[start_idx:], q_min[start_idx:], q_max[start_idx:]
                )
                q_test = q_next

            pin.framesForwardKinematics(self.pin_model, data, q_test)
            pin.updateFramePlacements(self.pin_model, data)
            final_err = np.linalg.norm(
                data.oMf[frame_id].translation - target_pos
            )

            if final_err < 1e-3:
                self.p_start_world = (target_pos - surface_offset).copy()
                self.q0 = q_test.copy()
                valid_found = True
                break

            q_current = q_test.copy()

        if not valid_found:
            self.q0 = q_test.copy()
            self.p_start_world = target_pos.copy()
    
        for jnt_name in self.pin_model.names:
            if self.full_model_for_mapping.existJointName(jnt_name) and \
               self.pin_model.existJointName(jnt_name):
                
                idx_q_reduced = self.pin_model.joints[
                    self.pin_model.getJointId(jnt_name)].idx_q
                idx_q_full = self.full_model_for_mapping.joints[
                    self.full_model_for_mapping.getJointId(jnt_name)].idx_q
                
                nq = self.pin_model.joints[
                    self.pin_model.getJointId(jnt_name)].nq
                
                self.q0_full[idx_q_full : idx_q_full + nq] = \
                    self.q0[idx_q_reduced : idx_q_reduced + nq]

    def init_viz(self):
        """Initialize Pinocchio Meshcat visualizer (same as model_mocap)"""
        self.viz = MeshcatVisualizer(self.pin_model, self.collision_model, self.visual_model)
        self.viz.initViewer(open=True)
        self.viz.loadViewerModel()
        self.viz.display(self.q0)
        print("Meshcat visualizer initialized at http://127.0.0.1:7000/static/")

    def visualize_contact(self, q=None):
        """Same as model_mocap.py - visualize contact point, rail, and surface frame"""
        if self.viz is None:
            self.init_viz()

        if q is None:
            q = self.q0

        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        
        viewer = self.viz.viewer
        
        viewer['rail_line'].set_object(g.Line(
            g.PointsGeometry(np.vstack([self.p_start_world, self.p_end_world]).T),
            g.LineBasicMaterial(color=0x00ff00, linewidth=5)
        ))

        viewer['rail_start'].set_object(g.Sphere(0.015), g.MeshLambertMaterial(color=0x00ff00))
        viewer['rail_start'].set_transform(tf.translation_matrix(self.p_start_world))

        viewer['rail_end'].set_object(g.Sphere(0.015), g.MeshLambertMaterial(color=0x00aa00))
        viewer['rail_end'].set_transform(tf.translation_matrix(self.p_end_world))
        
        stick_length = 0.8
        stick_radius = 0.02
        
        t_stick_mid = (self.p_start_world + self.p_end_world) / 2.0
        axis_vector = self.p_end_world - self.p_start_world
        axis_len = np.linalg.norm(axis_vector)
        z_axis = axis_vector / axis_len
        
        x_axis = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        
        t_stick = self.p_stick
        
        R_stick_meshcat = np.column_stack([x_axis, z_axis, y_axis])
        
        T_stick = np.eye(4)
        T_stick[:3, :3] = R_stick_meshcat
        T_stick[:3, 3] = t_stick
        
        viewer['stick_cylinder'].set_object(
            g.Cylinder(stick_length, stick_radius),
            g.MeshLambertMaterial(color=0x8B4513, opacity=0.9) 
        )
        viewer['stick_cylinder'].set_transform(T_stick)
        
        self.viz.display(q)

    def update_solver_weights(self, w_run_new, w_term_new):
        """
        IRL interface: update MPPI cost weights.
        Compatible with Human.update_solver_weights().
        """
        merged_w_run = {**self.w_run, **w_run_new} 
        self.controller.set_weights(w_run=merged_w_run, w_term=w_term_new)
        self.w_run = merged_w_run
        self.w_term = w_term_new

    def update_solver_weights_tv(self, w_run_windows, w_term_windows):
        """
        Store per-window weights for use during solve().
        w_run_windows : np.ndarray (n_w, nr_run)
        w_term_windows: np.ndarray (n_w, nr_term)
        """
        self._w_run_windows = w_run_windows
        self._w_term_windows = w_term_windows

    def get_control(self, qs, qds, qdds):
        """
        IRL interface: compute torques via RNEA (same as Human.get_control).
        Returns RNEA torques (= what Pinocchio inverse dynamics gives).
        """
        us = []
        for q, qd, qdd in zip(qs, qds, qdds):
            pin.rnea(self.pin_model, self.pin_data, q, qd, qdd)
            us.append(self.pin_data.tau.copy())
        return np.stack(us)

    def get_traj_features(self, xs, us):
        nr_run  = len(self.keys_run)
        nr_term = len(self.keys_term)
        nr      = nr_run + nr_term

        Phi         = np.zeros(nr)
        Phi_int     = np.zeros(nr)
        Phis        = []
        Phis_Cum    = []
        Phis_Cum_Int= []

        T = min(len(us), len(xs) - 1)

        if 'press_force' in self.keys_run:
            is_demo = (hasattr(self, '_demo_forces') and self._demo_forces is not None
                       and hasattr(self, '_demo_xs0')
                       and len(self._demo_forces) >= T
                       and np.allclose(xs[0], self._demo_xs0, atol=1e-6)
                       and (not hasattr(self, '_demo_xsT')
                            or self._demo_xsT is None
                            or np.allclose(xs[-1], self._demo_xsT, atol=1e-6)))
            has_injected = (hasattr(self, '_press_forces') and self._press_forces is not None
                            and len(self._press_forces) >= T)

            if is_demo:
                self._press_forces = self._demo_forces[:T]
            elif has_injected:
                self._press_forces = self._press_forces[:T]
            else:
                self._press_forces = self._simulate_press_forces(xs, us)
        else:
            self._press_forces = None

        x_term = xs[-1]
        q_term = x_term[:self.nq]
        v_term = x_term[self.nq:]
        pin.forwardKinematics(self.pin_model, self.pin_data, q_term, v_term)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        for j, key in enumerate(self.keys_term):
            cost_val = self._compute_term_feature(key, x_term)
            Phi[nr_run + j]     += cost_val
            Phi_int[nr_run + j] += cost_val

        Phis_Cum.append(Phi.copy())
        Phis_Cum_Int.append(Phi_int.copy())
        Phis.append(Phi.copy())

        if 'progress_vel' in self.keys_run:
            rail_vec = self.p_end_world - self.p_start_world
            rail_len = np.linalg.norm(rail_vec)
            rail_unit = rail_vec / max(rail_len, 1e-9)
            self._traveled_cache = np.zeros(T + 1)
            for i in range(T + 1):
                x = xs[i]
                q, v = x[:self.nq], x[self.nq:]
                self._sync_mujoco(q, v, np.zeros(self.mj_model.nu), t=i)
                site_id = getattr(self.controller, 'contact_site_id', None) or self.controller._contact_site_id
                sp = self.mj_data.site_xpos[site_id]
                self._traveled_cache[i] = np.dot(sp - self.p_start_world, rail_unit)

        # causal EMA must be applied both sides or it's a phantom asymmetry.
        _vel_ema = float(getattr(self, 'vel_feat_ema', 1.0))
        xs_arr = np.asarray(xs)
        v_arr  = xs_arr[:, self.nq:self.nq + self.nv]
        if _vel_ema < 1.0 and len(v_arr) > 1:
            self._v_smooth = np.zeros_like(v_arr)
            self._v_smooth[0] = v_arr[0]
            for _k in range(1, len(v_arr)):
                self._v_smooth[_k] = _vel_ema * v_arr[_k] + (1.0 - _vel_ema) * self._v_smooth[_k - 1]
        else:
            self._v_smooth = None
        if 'JA' in self.keys_run or 'JA' in self.keys_term:
            v_for_ja = self._v_smooth if self._v_smooth is not None else v_arr
            self._ddq_cache = np.zeros_like(v_for_ja)
            if v_for_ja.shape[0] >= 2:
                self._ddq_cache[:-1] = np.diff(v_for_ja, axis=0) / self.dt
                self._ddq_cache[-1]  = self._ddq_cache[-2]
        else:
            self._ddq_cache = None

        # Post-hoc torque low-pass for Tau/Eng ONLY — MUST match the rollout's
        _tau_ema = float(getattr(self, 'tau_feat_ema', 1.0))
        if _tau_ema < 1.0 and us is not None and len(us) > 0:
            us_ema = np.zeros_like(us)
            us_ema[0] = _tau_ema * us[0]
            for _k in range(1, len(us)):
                us_ema[_k] = _tau_ema * us[_k] + (1.0 - _tau_ema) * us_ema[_k - 1]
        else:
            us_ema = us

        _jtc_rate = bool(getattr(self, 'jtc_rate', False))
        if _jtc_rate and us is not None and len(us) > 1:
            jtc_arr = np.zeros(len(us))
            jtc_arr[1:] = np.linalg.norm(np.diff(us, axis=0), axis=1) / 50.0
        else:
            jtc_arr = None

        for i in range(T - 1, -1, -1):
            Phi_temp     = np.zeros(nr)
            Phi_temp_int = np.zeros(nr)

            x = xs[i]
            u = us[i]
            u_feat = us_ema[i] if _tau_ema < 1.0 else u

            q_i = x[:self.nq]
            v_i = self._v_smooth[i] if getattr(self, '_v_smooth', None) is not None else x[self.nq:]
            self._sync_mujoco(q_i, v_i, np.zeros(self.mj_model.nu), t=i)

            for j, key in enumerate(self.keys_run):
                if key == 'JTC' and jtc_arr is not None:
                    cost_val = jtc_arr[min(i, len(jtc_arr) - 1)]
                else:
                    _u = u_feat if (key.startswith('Tau') or key.startswith('Eng')) else u
                    cost_val = self._compute_run_feature(key, x, _u, t=i)
                Phi_temp[j]     += cost_val
                Phi_temp_int[j] += cost_val * self.dt

            Phis.append(Phi_temp)
            Phi     += Phi_temp
            Phi_int += Phi_temp_int
            Phis_Cum.append(Phi.copy())
            Phis_Cum_Int.append(Phi_int.copy())

        return Phi, Phis[::-1], Phis_Cum[::-1], Phis_Cum_Int[::-1]

    def get_new_traj_features(self):

        if not hasattr(self, '_last_xs') or not hasattr(self, '_last_us'):
            raise RuntimeError("Call solve() before get_new_traj_features()")
        return self.get_traj_features(self._last_xs, self._last_us)
