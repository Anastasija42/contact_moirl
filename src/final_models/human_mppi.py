"""HumanMPPI — MPPI-based solver for human-with-tool."""
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
from final_models.human_base import HumanBase

try:
    from MPPI_MJX import MPPIMJXController
    _MJX_AVAILABLE = True
    print("[INFO] MPPIMJXController is available for GPU-accelerated MPPI")
except Exception as _mjx_err:
    _MJX_AVAILABLE = False
    _MJX_ERR = _mjx_err
    print(f"[WARN] MPPIMJXController not available, GPU-accelerated MPPI will be disabled: {_MJX_ERR}")

ACTIVE_ARM_JOINTS = [
    "middle_thoracic_X",
    "right_clavicle_joint_X",
    "right_shoulder_Z", "right_shoulder_X", "right_shoulder_Y",
    "right_elbow_Z", "right_elbow_Y",
    "right_wrist_Z", "right_wrist_X",
]

class HumanMPPI(HumanBase):
    """MPPI + MuJoCo solver."""

    def _post_init(self, args):
        """MPPI-specific __init__ tail."""

        self.dt = args.get('dt', 0.01)
        self.T = args.get('T', 500)
        self.stick_static = args.get('stick_static', True)
        self.stick_traj = None
        self._stick_qposadr = None

        arm_q_traj = args.get('q_traj', None)
        if arm_q_traj is not None and len(arm_q_traj) > 0:
            arm_names = list(ACTIVE_ARM_JOINTS)
            arm_idx = [self.full_pin_model.joints[self.full_pin_model.getJointId(n)].idx_q
                       for n in arm_names if self.full_pin_model.existJointName(n)]
            arm_q_arr = np.asarray(arm_q_traj, dtype=np.float64)
            full_q_traj = np.tile(self.q0_full, (len(arm_q_arr), 1))
            ncols = min(arm_q_arr.shape[1], len(arm_idx))
            for k in range(len(arm_q_arr)):
                for i in range(ncols):
                    full_q_traj[k, arm_idx[i]] = arm_q_arr[k, i]
            try:
                self._setup_full_stick(full_q_traj)
            except Exception as e:
                print(f"[HumanMPPI] _setup_full_stick failed ({e}); rock pos in XML will use reduced model")

        xml_path = f"config/xml_models/{self.date}/{self.subject_id}.xml"
        pinned_path = self.create_pinned_model(xml_path)

        temp_data = self.pin_model.createData()
        pin.framesForwardKinematics(self.pin_model, temp_data, self.q0)
        pin.updateFramePlacements(self.pin_model, temp_data)

        p_contact = temp_data.oMf[self.contact_frame_id].translation
        rock_frame_id = self.pin_model.getFrameId("rock_frame")
        rock_pose = temp_data.oMf[rock_frame_id]
        p_rock    = rock_pose.translation
        R_rock    = rock_pose.rotation

        offset_world = p_contact - p_rock

        offset_local = R_rock.T @ offset_world           

        tree = ET.parse(pinned_path)
        root = tree.getroot()
        rock_body = root.find(".//body[@name='rock']")
        if rock_body is not None:
            for s in rock_body.findall("site[@name='rock_contact_point']"):
                rock_body.remove(s)
            site = ET.SubElement(rock_body, "site")
            site.set("name", "rock_contact_point")
            site.set("pos", f"{offset_local[0]:.6f} {offset_local[1]:.6f} {offset_local[2]:.6f}")
            site.set("size", "0.005")
            site.set("rgba", "1 0 0 1")
            if self.args.get('primitive_rock', False):
                _Rrock = float(self.args.get('rock_radius', 0.03))
                _pen = float(self.args.get('rock_penetration', 0.0))
                _non = np.linalg.norm(offset_local) + 1e-12
                _cen = offset_local - (_Rrock - _pen) * (offset_local / _non)
                for g in rock_body.findall("geom[@name='rock_sphere']"):
                    g.set("type", "sphere"); g.set("size", f"{_Rrock:.6f}")
                    g.set("pos", f"{_cen[0]:.6f} {_cen[1]:.6f} {_cen[2]:.6f}")
                print(f"[INFO] primitive_rock: sphere R={_Rrock*1000:.0f}mm centered at {np.round(_cen,4)} (surface grazes the site)")
            tree.write(pinned_path)
            print(f"[INFO] Patched rock_contact_point site (local offset): {np.round(offset_local, 4)}")
        else:
            print("[WARN] rock body not found in XML — contact site NOT added")

        # freejoint) → nv=9, neq=0. Removes the ill-conditioned qM (armature-1e10) that nan'd the MJX
        if __import__('os').environ.get('USE_PRUNED'):
            import sys as _sys
            if 'analysis' not in _sys.path:
                _sys.path.insert(0, 'analysis')
            from prune_arm_model import prune as _prune_arm
            _pruned_path = pinned_path.replace('_pinned', '_pruned')
            # rail → the task-space law goes integration-unstable at the coarse MPPI_DT → float32 nan
            # (the local-only nan; mango's full model is faithful so it runs clean). So build the FULL
            _mF = mujoco.MjModel.from_xml_path(pinned_path); _dF = mujoco.MjData(_mF)
            for _i in range(_mF.njnt):
                _jn = mujoco.mj_id2name(_mF, mujoco.mjtObj.mjOBJ_JOINT, _i)
                if _jn == "root" or _jn is None or not self.full_model_for_mapping.existJointName(_jn):
                    continue
                _pid = self.full_model_for_mapping.getJointId(_jn)
                if self.full_model_for_mapping.joints[_pid].nq != 1:
                    continue
                if self.pin_model.existJointName(_jn):
                    _val = self.q0[self.pin_model.joints[self.pin_model.getJointId(_jn)].idx_q]
                else:
                    _val = self.q0_full[self.full_model_for_mapping.joints[_pid].idx_q]
                _dF.qpos[_mF.jnt_qposadr[_i]] = _val
            _prune_arm(pinned_path, _pruned_path, q_full=_dF.qpos.copy())
            pinned_path = _pruned_path
            print(f"[USE_PRUNED] reduced 9-DOF arm model (demo trunk BAKED) → {pinned_path}")

        self.mj_model = mujoco.MjModel.from_xml_path(pinned_path)
        # MPPI never properly penalized contact-break. With elliptic
        self.mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.mj_data = mujoco.MjData(self.mj_model)

        
        self.setup_lighting()

        for i in range(self.mj_model.njnt):
            jnt_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)

            if jnt_name == "root" or jnt_name is None:
                continue

            if self.full_model_for_mapping.existJointName(jnt_name):
                pin_id = self.full_model_for_mapping.getJointId(jnt_name)
                pin_q_idx = self.full_model_for_mapping.joints[pin_id].idx_q
                pin_nq = self.full_model_for_mapping.joints[pin_id].nq

                mj_q_idx = self.mj_model.jnt_qposadr[i]

                if pin_nq == 1:
                    if self.pin_model.existJointName(jnt_name):
                        red_id = self.pin_model.getJointId(jnt_name)
                        red_q_idx = self.pin_model.joints[red_id].idx_q
                        val = self.q0[red_q_idx]
                    else:
                        val = self.q0_full[pin_q_idx]
                    self.mj_data.qpos[mj_q_idx] = val
                    print(f"[DEBUG] Syncing {jnt_name}: {val:.4f}")

        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._fix_initial_penetration()

        if not self.stick_static:
            _jnt_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, "stick_freejoint")
            if _jnt_id >= 0:
                self._stick_qposadr = self.mj_model.jnt_qposadr[_jnt_id]
                print(f"[INFO] Moving stick: freejoint qposadr={self._stick_qposadr}")
            else:
                print("[WARN] stick_freejoint not found — falling back to stick_static=True")
                self.stick_static = True

        site_id_tmp = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "rock_contact_point")
        if site_id_tmp >= 0:
            p_start_mj = self.mj_data.site_xpos[site_id_tmp].copy()
            rail_vec = self.p_end_world - self.p_start_world
            print(f"[INFO] p_start Pinocchio: {np.round(self.p_start_world, 4)}")
            print(f"[INFO] p_start MuJoCo:    {np.round(p_start_mj, 4)}")
            print(f"[INFO] offset:            {np.round(p_start_mj - self.p_start_world, 4)}")
            self.p_start_world = p_start_mj
            self.p_end_world = p_start_mj + rail_vec
        else:
            print("[WARN] rock_contact_point site not found — p_start not re-anchored")

        self.rock_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "rock_contact_point")
        self.force_sensor_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "rock_force_sensor")
        if self.rock_site_id >= 0 and self.force_sensor_id >= 0:
            self.force_sensor_adr = self.mj_model.sensor_adr[self.force_sensor_id]
        else:
            self.force_sensor_adr = -1

        self.lock_inactive_joints()

        self.horizon = args.get('horizon', 50)
        self.num_samples = args.get('num_samples', 128)
        self.noise_sigma = args.get('noise_sigma', 1.5)
        self.lambda_ = args.get('lambda_', 0.5)
        self.use_gpu = args.get('use_gpu', False)

        x0_mppi = np.concatenate([self.mj_data.qpos, self.mj_data.qvel])
        _ctrl_kwargs = dict(
            model=self.mj_model,
            data=self.mj_data,
            pin_model=self.pin_model,
            pin_data=self.pin_data,
            contact_frame_id=self.contact_frame_id,
            horizon=self.horizon,
            num_samples=self.num_samples,
            noise_sigma=self.noise_sigma,
            lambda_=self.lambda_,
            target_force=self.target_force,
            p_start_world=self.p_start_world,
            p_end_world=self.p_end_world,
            mu=self.mu,
            locked_joint_constraints=self.locked_joint_constraints,
            x0=x0_mppi,
        )
        if self.use_gpu and _MJX_AVAILABLE:
            print("[HumanMPPI] Using GPU controller (MPPIMJXController)")
            self.controller = MPPIMJXController(**_ctrl_kwargs)
        else:
            if self.use_gpu and not _MJX_AVAILABLE:
                print(f"[HumanMPPI] WARNING: use_gpu=True but MJX unavailable, falling back to CPU\n  Reason: {_MJX_ERR}")
            self.controller = MPPIController(**_ctrl_kwargs, q_traj=args.get('q_traj', None))

        self.solver = self._make_solver_shim()

        
        self.w_run = args.get('w_run', {
            'Tau': 0.01,
            'JV':  0.01,
        })
        self.w_term = args.get('w_term', {
            'JV':  1e-6,
            'Geo': 1e-6,
            'JTC': 1e-6,
            'JA':  1e-6,
        })
        self.phi_exclude = set(args.get('phi_exclude', []))

        self._last_xs = None
        self._last_us = None
        
        self.controller.set_weights(w_run=self.w_run, w_term=self.w_term)
        
        self.viewer = None
        self.viz = None

        self.mujoco_init_q = self.mj_data.qpos.copy()

        if not self.stick_static:
            _q_traj = args.get('q_traj', None)
            if _q_traj is not None and len(_q_traj) > 0:
                self.stick_traj = self._estimate_stick_trajectory(np.array(_q_traj))
                print(f"[INFO] Stick trajectory computed: {self.stick_traj.shape} frames")
                self._stick_traj_ctrl = self.stick_traj
                self._stick_qposadr_ctrl = self._stick_qposadr
                self.controller.set_stick_traj(self.stick_traj, self._stick_qposadr)
            else:
                print("[WARN] stick_static=False requires q_traj in args — falling back to stick_static=True")
                self.stick_static = True

    def _fix_initial_penetration(self):
        """
        After MuJoCo model is loaded and mj_forward called, check the actual
        rock-stick gap using MuJoCo geometry. If penetrating, compute a world-space
        correction and apply it via one Pinocchio IK pass, then re-sync qpos.
        """
        STICK_RADIUS = 0.02
        
        ROCK_RADIUS  = 0.02
        STICK_HALF_LEN = 0.4
        MARGIN = 0.00

        rock_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "rock")
        stick_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "stick")
        if rock_body_id < 0 or stick_body_id < 0:
            print("[WARN] _fix_initial_penetration: rock or stick body not found")
            return

        rock_geom_id = None
        for g in range(self.mj_model.body_geomadr[rock_body_id],
                       self.mj_model.body_geomadr[rock_body_id] + self.mj_model.body_geomnum[rock_body_id]):
            if self.mj_model.geom_contype[g] > 0:
                rock_geom_id = g
                break
        if rock_geom_id is None:
            print("[WARN] _fix_initial_penetration: rock collision geom not found")
            return

        rock_center  = self.mj_data.geom_xpos[rock_geom_id].copy()
        stick_center = self.mj_data.xpos[stick_body_id].copy()
        stick_axis   = self.mj_data.xmat[stick_body_id].reshape(3, 3)[:, 2]

        t = np.clip(np.dot(rock_center - stick_center, stick_axis),
                    -STICK_HALF_LEN, STICK_HALF_LEN)
        nearest = stick_center + t * stick_axis
        diff = rock_center - nearest
        dist = np.linalg.norm(diff)
        gap  = dist - (STICK_RADIUS + ROCK_RADIUS)

        print(f"[INFO] MuJoCo rock-stick gap: {gap*100:.2f} cm  "
              f"({'OK' if gap >= 0 else 'PENETRATING'})")

        if gap >= -MARGIN:
            return

        normal = diff / dist if dist > 1e-6 else stick_axis * 0
        correction = normal * (-gap + MARGIN)

        contact_name = self.args.get('contact', 'rock_contact_point')
        frame_id = (self.pin_model.getFrameId(contact_name)
                    if self.pin_model.existFrame(contact_name) else self.tool_frame_id)

        ik_data = self.pin_model.createData()
        q_test  = self.q0.copy()
        q_min   = self.pin_model.lowerPositionLimit
        q_max   = self.pin_model.upperPositionLimit
        has_ff  = (self.pin_model.joints[1].shortname() == "JointModelFreeFlyer")
        start_idx = 7 if has_ff else 0

        pin.framesForwardKinematics(self.pin_model, ik_data, q_test)
        pin.updateFramePlacements(self.pin_model, ik_data)
        current_frame_pos = ik_data.oMf[frame_id].translation.copy()
        target_pos = current_frame_pos + correction

        for _ in range(100):
            pin.framesForwardKinematics(self.pin_model, ik_data, q_test)
            pin.updateFramePlacements(self.pin_model, ik_data)
            err = ik_data.oMf[frame_id].translation - target_pos
            if np.linalg.norm(err) < 1e-4:
                break
            J = pin.computeFrameJacobian(self.pin_model, ik_data, q_test,
                                          frame_id, pin.LOCAL_WORLD_ALIGNED)
            J_t = J[:3, :]
            v = -J_t.T @ np.linalg.inv(J_t @ J_t.T + 1e-6 * np.eye(3)) @ err
            q_test = pin.integrate(self.pin_model, q_test, v)
            q_test[start_idx:] = np.clip(q_test[start_idx:], q_min[start_idx:], q_max[start_idx:])

        self.q0 = q_test.copy()

        for jnt_name in self.pin_model.names:
            if (self.full_model_for_mapping.existJointName(jnt_name) and
                    self.pin_model.existJointName(jnt_name)):
                idx_q_reduced = self.pin_model.joints[self.pin_model.getJointId(jnt_name)].idx_q
                idx_q_full    = self.full_model_for_mapping.joints[self.full_model_for_mapping.getJointId(jnt_name)].idx_q
                nq = self.pin_model.joints[self.pin_model.getJointId(jnt_name)].nq
                self.q0_full[idx_q_full: idx_q_full + nq] = q_test[idx_q_reduced: idx_q_reduced + nq]

        for i in range(self.mj_model.njnt):
            jnt_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if jnt_name == "root" or jnt_name is None:
                continue
            if self.full_model_for_mapping.existJointName(jnt_name):
                pin_id    = self.full_model_for_mapping.getJointId(jnt_name)
                pin_q_idx = self.full_model_for_mapping.joints[pin_id].idx_q
                mj_q_idx  = self.mj_model.jnt_qposadr[i]
                if self.full_model_for_mapping.joints[pin_id].nq == 1:
                    self.mj_data.qpos[mj_q_idx] = self.q0_full[pin_q_idx]

        mujoco.mj_forward(self.mj_model, self.mj_data)

        rock_center2  = self.mj_data.geom_xpos[rock_geom_id].copy()
        nearest2 = stick_center + np.clip(
            np.dot(rock_center2 - stick_center, stick_axis),
            -STICK_HALF_LEN, STICK_HALF_LEN) * stick_axis
        gap2 = np.linalg.norm(rock_center2 - nearest2) - (STICK_RADIUS + ROCK_RADIUS)
        print(f"[INFO] After fix: rock-stick gap = {gap2*100:.2f} cm")

    def setup_lighting(self):
        """Configure lighting in MuJoCo model"""
        self.mj_model.vis.headlight.ambient[:] = [0.5, 0.5, 0.5]
        self.mj_model.vis.headlight.diffuse[:] = [0.8, 0.8, 0.8]
        self.mj_model.vis.headlight.specular[:] = [0.3, 0.3, 0.3]
        self.mj_model.vis.quality.shadowsize = 4096
        self.mj_model.vis.map.zfar = 30.0
    """

    def pin_neck_joints(self):
        locked_joint_names = [
            # Neck
            'middle_cervical_Z', 'middle_cervical_X', 'middle_cervical_Y',
            
            # Torso
            'middle_thoracic_Z', 'middle_thoracic_X', 'middle_thoracic_Y',
            'middle_lumbar_Z', 'middle_lumbar_X',
            
            # Head
            'middle_head_Z', 'middle_head_X', 'middle_head_Y',
            
            # Left arm
            'left_shoulder_Z', 'left_shoulder_X', 'left_shoulder_Y',
            'left_elbow_Z', 'left_elbow_Y',
            'left_wrist_Z', 'left_wrist_X',
            
            # Left clavicle
            'left_clavicle_Z', 'left_clavicle_X', 'left_clavicle_Y',
        ]
        
        # Store mappings: (qpos_id, dof_id, initial_value)
        self.locked_joint_constraints = []
        
        for joint_name in locked_joint_names:
            try:
                joint_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if joint_id >= 0:
                    dof_id = self.mj_model.jnt_dofadr[joint_id]
                    qpos_id = self.mj_model.jnt_qposadr[joint_id]
                    
                    # Very high damping to prevent motion
                    self.mj_model.dof_damping[dof_id] = 10000.0
                    self.mj_model.dof_armature[dof_id] = 1.0
                    
                    # Store the INITIAL position from current qpos
                    initial_value = self.mj_data.qpos[qpos_id]
                    self.locked_joint_constraints.append((qpos_id, dof_id, initial_value))
                    
                    print(f"[INFO] Locked '{joint_name}' at initial value: {initial_value:.4f}")
            except:
                pass
        
        print(f"[INFO] Total joints locked: {len(self.locked_joint_constraints)}")
    """

    def lock_inactive_joints(self):
        """Lock joints not in the reduced Pinocchio model using MuJoCo equality constraints.

        The XML already contains <equality><joint ...> elements for every inactive
        hinge.  Here we:
          1. Set each equality constraint's target value (eq_data) to the current
             qpos so the joint is locked at the initialised stance.
          2. Record the (qpos_adr, dof_adr, lock_val) triples in
             locked_joint_constraints so that existing code referencing that list
             still works (but the teleportation loop is no longer needed — MuJoCo's
             constraint solver handles it).
        """
        self.locked_joint_constraints = []

        active_names = set(ACTIVE_ARM_JOINTS)

        eq_joint_map = {}
        for eq_id in range(self.mj_model.neq):
            if self.mj_model.eq_type[eq_id] == mujoco.mjtEq.mjEQ_JOINT:
                jnt_id = self.mj_model.eq_obj1id[eq_id]
                jnt_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jnt_id)
                eq_joint_map[jnt_name] = eq_id

        for i in range(self.mj_model.njnt):
            jnt_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, i)

            if jnt_name in active_names or jnt_name == "rock_fixed_joint":
                continue

            if self.mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                continue

            dof_adr = self.mj_model.jnt_dofadr[i]
            qpos_adr = self.mj_model.jnt_qposadr[i]
            lock_val = self.mj_data.qpos[qpos_adr]

            # float32-hostile (MJX rollout → nan); LOCK_ARM env overrides it for
            _lock_arm = float(__import__('os').environ.get('LOCK_ARM', 1e10))
            self.mj_model.dof_armature[dof_adr] = _lock_arm
            self.mj_model.dof_damping[dof_adr] = 1e5

            eq_id = eq_joint_map.get(jnt_name)
            if eq_id is not None:
                self.mj_model.eq_data[eq_id, 0] = lock_val
                self.mj_model.eq_active0[eq_id] = 1

            self.locked_joint_constraints.append((qpos_adr, dof_adr, lock_val))
            print(f"[INFO] Locking {jnt_name} at {lock_val:.4f} (armature={_lock_arm:.0e})")

    def lock_thorax_joint(self):
        """Hold the thoracic (trunk) DOF at its q0, even though it is an ACTIVE
        arm joint. The trunk is a weakly-identified direction (still in the
        demo) that the sampler otherwise recruits via redundancy -- blowing up
        the trunk RMSE and stealing the shoulder signal (CSQP shows the same).
        We lock it with the same huge-armature/damping trick as the inactive
        joints, WITHOUT removing it from ACTIVE_ARM_JOINTS, so the control and
        feature vectors keep their dimension and no joint indices shift (avoids
        the 7-vs-9 joint-index landmine).

        Must be called on the model AFTER construction but BEFORE the MJX
        controller is built, so the device model compiles with the locked
        armature (changing mj_model after the controller compiles will not
        propagate to the GPU model).
        """
        jname = 'middle_thoracic_X'
        jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            print(f"[WARN] lock_thorax: joint '{jname}' not found -- skipped")
            return
        dof_adr = self.mj_model.jnt_dofadr[jid]
        qpos_adr = self.mj_model.jnt_qposadr[jid]
        lock_val = float(self.mj_data.qpos[qpos_adr])
        self.mj_model.dof_armature[dof_adr] = float(__import__('os').environ.get('LOCK_ARM', 1e10))
        self.mj_model.dof_damping[dof_adr] = 1e5
        for eq_id in range(self.mj_model.neq):
            if (self.mj_model.eq_type[eq_id] == mujoco.mjtEq.mjEQ_JOINT
                    and self.mj_model.eq_obj1id[eq_id] == jid):
                self.mj_model.eq_data[eq_id, 0] = lock_val
                self.mj_model.eq_active0[eq_id] = 1
                break
        mujoco.mj_forward(self.mj_model, self.mj_data)
        red_idx = None
        if self.pin_model.existJointName(jname):
            _pj = self.pin_model.getJointId(jname)
            red_idx = int(self.pin_model.joints[_pj].idx_q)
            self._locked_q_idx = sorted(
                set(getattr(self, '_locked_q_idx', []) or []) | {red_idx})
        print(f"[INFO] lock_thorax: holding '{jname}' at q0={lock_val:.4f} "
              f"(armature=1e10, damping=1e5)  | q_norm excludes reduced idx {red_idx}")

    def create_pinned_model(self, input_file):
        """Remove freejoint from XML to create pinned model with lighting, stick, and rock"""
        output_file = input_file.replace('.xml', '_pinned.xml')
        
        tree = ET.parse(input_file)
        root = tree.getroot()

        option_elem = root.find("option")
        if option_elem is None:
            option_elem = ET.SubElement(root, "option")
        option_elem.set("impratio", "10")
        option_elem.set("noslip_iterations", "10")

        default_elem = root.find("default")
        if default_elem is None:
            default_elem = ET.SubElement(root, "default")
        default_geom = default_elem.find("geom")
        if default_geom is None:
            default_geom = ET.SubElement(default_elem, "geom")
        default_geom.set("solref",  "0.4 1")
        default_geom.set("solimp",  "0.6 0.8 0.05")

        contact_elem = root.find("contact")
        if contact_elem is None:
            contact_elem = ET.SubElement(root, "contact")
        
        for excl in contact_elem.findall("exclude"):
            contact_elem.remove(excl)
        
        excludes = [
            ("middle_pelvis",  "left_upperleg"),
            ("middle_pelvis",  "right_upperleg"),
            ("middle_pelvis",  "middle_abdomen"),
            ("middle_abdomen", "middle_thorax"),
            ("middle_thorax",  "middle_head"),
            ("left_lowerleg",  "left_foot"),
            ("right_lowerleg", "right_foot"),
            ("left_lowerarm",  "left_hand"),
            ("right_lowerarm", "right_hand"),
            ("left_foot",       "right_foot"),
            ("right_hand",      "rock"),
            ("right_hand",      "stick"),
            ("right_lowerarm",  "stick"),
            ("left_lowerarm",   "stick"),
            ("left_hand",       "stick"),
        ]
        
        for b1, b2 in excludes:
            excl = ET.SubElement(contact_elem, "exclude")
            excl.set("body1", b1)
            excl.set("body2", b2)

        rock_stick_pair = ET.SubElement(contact_elem, "pair")
        rock_stick_pair.set("geom1", "rock_sphere")
        rock_stick_pair.set("geom2", "stick_geom")
        rock_stick_pair.set("solref", "0.02 1.5")
        rock_stick_pair.set("solimp", "0.9 0.99 0.001")
        rock_stick_pair.set("friction", f"{self.mu} {self.mu} 0.005 0.0001 0.0001")
        
        visual = root.find("visual")
        if visual is None:
            visual = ET.SubElement(root, "visual")
            headlight = ET.SubElement(visual, "headlight")
            headlight.set("ambient", "0.5 0.5 0.5")
            headlight.set("diffuse", "0.8 0.8 0.8")
            headlight.set("specular", "0.3 0.3 0.3")
            
            quality = ET.SubElement(visual, "quality")
            quality.set("shadowsize", "4096")
            
            map_elem = ET.SubElement(visual, "map")
            map_elem.set("zfar", "30")
        
        worldbody = root.find("worldbody")
        
        existing_lights = worldbody.findall("light")
        if len(existing_lights) == 0:
            light1 = ET.SubElement(worldbody, "light")
            light1.set("directional", "true")
            light1.set("pos", "0 0 3")
            light1.set("dir", "0 0 -1")
            light1.set("diffuse", "0.8 0.8 0.8")
            light1.set("specular", "0.3 0.3 0.3")
            
            light2 = ET.SubElement(worldbody, "light")
            light2.set("directional", "true")
            light2.set("pos", "2 2 3")
            light2.set("dir", "-1 -1 -1")
            light2.set("diffuse", "0.4 0.4 0.4")
            
            light3 = ET.SubElement(worldbody, "light")
            light3.set("directional", "true")
            light3.set("pos", "-2 -2 3")
            light3.set("dir", "1 1 -1")
            light3.set("diffuse", "0.4 0.4 0.4")
        
        root_body = None
        for body in worldbody.findall("body"):
            if body.get("name") != "stick":
                root_body = body
                break
        
        if root_body is not None:
            freejoint = root_body.find("freejoint")
            if freejoint is not None:
                print("[INFO] Removing <freejoint> -> Robot is now pinned.")
                root_body.remove(freejoint)
            
            base_pos = self.q0_full[0:3]

            quat_pin = np.asarray(self.q0_full[3:7], dtype=float)

            qn = float(np.linalg.norm(quat_pin))
            if qn < 1e-6:
                print("[WARN] q0_full quaternion ~0 — falling back to identity")
                quat_pin = np.array([0.0, 0.0, 0.0, 1.0])
            else:
                quat_pin = quat_pin / qn

            base_quat_wxyz = [quat_pin[3], quat_pin[0], quat_pin[1], quat_pin[2]]
            
            root_body.set("pos", f"{base_pos[0]:.6f} {base_pos[1]:.6f} {base_pos[2]:.6f}")
            root_body.set("quat", f"{base_quat_wxyz[0]:.6f} {base_quat_wxyz[1]:.6f} {base_quat_wxyz[2]:.6f} {base_quat_wxyz[3]:.6f}")
            
            if "euler" in root_body.attrib:
                del root_body.attrib["euler"]
            
            print(f"[INFO] Pinned robot at mocap base pose:")
            print(f"  pos: {base_pos}")
            print(f"  quat (MuJoCo wxyz): {base_quat_wxyz}")

            def color_arm_geoms(body, arm_side):
                """Recursively color all geoms in an arm"""
                body_name = body.get("name", "")
                
                for geom in body.findall("geom"):
                    if arm_side == "right":
                        geom.set("rgba", "0.8 0.0 0.0 0.6")
                    elif arm_side == "left":
                        geom.set("rgba", "0.0 0.8 0.0 0.6")
                
                for child in body.findall("body"):
                    color_arm_geoms(child, arm_side)
            
            def find_and_color_arms(body):
                """Find arm bodies and color them"""
                body_name = body.get("name", "")
                
                if "right_shoulder" in body_name or "right_upperarm" in body_name:
                    color_arm_geoms(body, "right")
                elif "left_shoulder" in body_name or "left_upperarm" in body_name:
                    color_arm_geoms(body, "left")
                
                for child in body.findall("body"):
                    find_and_color_arms(child)
            
            find_and_color_arms(root_body)

            if hasattr(self, 'p_stick'):
                stick_pos_world = self.p_stick.copy()
                
                axis_vector = self.p_end_world - self.p_start_world
                z_axis = axis_vector / np.linalg.norm(axis_vector)
                
                x_axis = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
                y_axis = np.cross(z_axis, x_axis)
                y_axis /= np.linalg.norm(y_axis)
                x_axis = np.cross(y_axis, z_axis)
                
                R_stick = np.column_stack([x_axis, y_axis, z_axis])
                quat_stick = pin.Quaternion(R_stick)
                
                stick_body = ET.SubElement(worldbody, "body")
                stick_body.set("name", "stick")
                stick_body.set("pos", f"{stick_pos_world[0]:.6f} {stick_pos_world[1]:.6f} {stick_pos_world[2]:.6f}")
                stick_body.set("quat", f"{quat_stick.w:.6f} {quat_stick.x:.6f} {quat_stick.y:.6f} {quat_stick.z:.6f}")
                if not self.stick_static:
                    fj = ET.SubElement(stick_body, "freejoint")
                    fj.set("name", "stick_freejoint")
                
                axis_len = float(np.linalg.norm(axis_vector))
                task = str(self.args.get('task', ''))
                tool_length = self.args.get('tool_length', None)
                if tool_length is not None:
                    stick_half_len = float(tool_length) / 2.0
                elif task.endswith('_long'):
                    stick_half_len = 0.45
                elif task == "down_short":
                    stick_half_len = axis_len / 2.0 + 0.025
                else:
                    stick_half_len = axis_len / 2.0
                stick_geom = ET.SubElement(stick_body, "geom")
                stick_geom.set("name", "stick_geom")
                stick_geom.set("type", "cylinder")
                stick_geom.set("size", f"0.02 {stick_half_len:.4f}")
                stick_geom.set("rgba", "0.55 0.40 0.25 1.0")
                stick_geom.set("contype", "1")
                stick_geom.set("conaffinity", "1")
                stick_geom.set("friction", f"{self.mu} 0.005 0.0001")

                
                print(f"[INFO] Added stick to MuJoCo XML at pos={stick_pos_world} (matching Pinocchio)")
            else:
                print("[WARN] self.p_stick not found - stick not added to MuJoCo XML")
            
            
            def find_body_recursive(body, name):
                """Recursively search for body by name"""
                if body.get("name") == name:
                    return body
                for child in body.findall("body"):
                    result = find_body_recursive(child, name)
                    if result is not None:
                        return result
                return None
            
            right_hand_body = find_body_recursive(root_body, "right_hand")
            
            if right_hand_body is not None:
                if self.full_pin_model.existJointName("rock_fixed_joint_full"):
                    full_jid = self.full_pin_model.getJointId("rock_fixed_joint_full")
                    rock_offset = self.full_pin_model.jointPlacements[full_jid].translation
                    rock_rot    = self.full_pin_model.jointPlacements[full_jid].rotation
                else:
                    rock_placement = self.pin_model.jointPlacements[self.pin_model.getJointId("rock_fixed_joint")]
                    rock_offset = rock_placement.translation
                    rock_rot    = rock_placement.rotation

                import scipy.spatial.transform as sst
                q_xyzw = sst.Rotation.from_matrix(rock_rot).as_quat()
                rock_quat_mj = [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]

                rock_body = ET.SubElement(right_hand_body, "body")
                rock_body.set("name", "rock")
                rock_body.set("pos", f"{rock_offset[0]:.6f} {rock_offset[1]:.6f} {rock_offset[2]:.6f}")
                rock_body.set("quat", f"{rock_quat_mj[0]:.6f} {rock_quat_mj[1]:.6f} {rock_quat_mj[2]:.6f} {rock_quat_mj[3]:.6f}")
                
                rock_inertial = ET.SubElement(rock_body, "inertial")
                rock_inertial.set("pos", "0 0 0")
                rock_inertial.set("mass", "0.25")
                rock_inertial.set("diaginertia", "0.00015 0.00010 0.00018")
                
                rock_geom_visual = ET.SubElement(rock_body, "geom")
                rock_geom_visual.set("type", "mesh")
                rock_geom_visual.set("mesh", "rock_mesh")
                rock_geom_visual.set("rgba", "0.6 0.6 0.6 1.0")
                rock_geom_visual.set("contype", "0")
                rock_geom_visual.set("conaffinity", "0")
                rock_geom_visual.set("group", "1")
                
                rock_geom_collision = ET.SubElement(rock_body, "geom")
                rock_geom_collision.set("name", "rock_sphere")
                if self.args.get('primitive_rock', False):
                    _Rrock = float(self.args.get('rock_radius', 0.03))
                    rock_geom_collision.set("type", "sphere")
                    rock_geom_collision.set("size", f"{_Rrock:.6f}")
                else:
                    rock_geom_collision.set("type", "mesh")
                    rock_geom_collision.set("mesh", "rock_mesh")
                rock_geom_collision.set("rgba", "0.6 0.6 0.6 0.0")
                rock_geom_collision.set("friction", f"{self.mu} 0.005 0.0001")
                
                print(f"[INFO] Added rock to right_hand at offset {rock_offset}")
                
                assets = root.find("asset")
                if assets is None:
                    assets = ET.SubElement(root, "asset")
                
                rock_mesh_exists = False
                for mesh in assets.findall("mesh"):
                    if mesh.get("name") == "rock_mesh":
                        rock_mesh_exists = True
                        break
                
                if not rock_mesh_exists:
                    rock_mesh_path = "../tool/sculpt_convex.stl"
                    rock_mesh = ET.SubElement(assets, "mesh")
                    rock_mesh.set("name", "rock_mesh")
                    rock_mesh.set("file", rock_mesh_path)
                    rock_mesh.set("scale", "0.0001 0.0001 0.0001")
                    print(f"[INFO] Added rock mesh asset from {rock_mesh_path}")
            else:
                print("[WARN] Could not find right_hand body - rock not added")
        
        rock_geom_collision.set("contype", "1")
        rock_geom_collision.set("conaffinity", "1")

        actuator_elem = root.find("actuator")
        if actuator_elem is None:
            actuator_elem = ET.SubElement(root, "actuator")
        
        for motor in actuator_elem.findall("motor"):
            actuator_elem.remove(motor)

        active_joint_names = list(ACTIVE_ARM_JOINTS)

        human_model_path = str(REPO / "human_model/urdf/human.urdf")
        import xml.etree.ElementTree as ET_READ
        urdf_tree = ET_READ.parse(human_model_path)
        urdf_root = urdf_tree.getroot()

        default_limits = {
            "middle_thoracic_X": 190.0,
            "right_clavicle_joint_X": 100.0,
            "right_shoulder_Z": 92.0,
            "right_shoulder_X": 71.0,
            "right_shoulder_Y": 52.0,
            "right_elbow_Z": 77.0,
            "right_elbow_Y": 15.0,
            "right_wrist_Z": 100.0,
            "right_wrist_X": 100.0,
        }

        print("\n=== Setting Actuator Limits from human.urdf ===")
        for jnt_name in active_joint_names:
            urdf_joint = urdf_root.find(f".//joint[@name='{jnt_name}']")
            effort_limit = None
            
            if urdf_joint is not None:
                limit_elem = urdf_joint.find("limit")
                if limit_elem is not None and "effort" in limit_elem.attrib:
                    effort_str = limit_elem.get("effort")
                    try:
                        effort_limit = float(effort_str)
                    except ValueError:
                        pass
            
            if effort_limit is None or effort_limit <= 0.0:
                effort_limit = default_limits.get(jnt_name, 50.0)
                print(f"  [WARN] {jnt_name}: Invalid URDF effort, using default {effort_limit} Nm")
            
            motor = ET.SubElement(actuator_elem, "motor")
            motor.set("name", f"{jnt_name}_ctrl")
            motor.set("joint", jnt_name)
            motor.set("gear", "1")
            motor.set("ctrllimited", "true")
            motor.set("ctrlrange", f"-{effort_limit} {effort_limit}")
            
            print(f"  {jnt_name:<25}: ctrlrange = [-{effort_limit:5.1f}, {effort_limit:5.1f}] Nm")

        sensor_elem = root.find("sensor")
        if sensor_elem is None:
            sensor_elem = ET.SubElement(root, "sensor")
        if sensor_elem.find("force[@name='rock_force_sensor']") is None:
            force_sensor = ET.SubElement(sensor_elem, "force")
            force_sensor.set("name", "rock_force_sensor")
            force_sensor.set("site", "rock_contact_point")

        equality_elem = root.find("equality")
        if equality_elem is None:
            equality_elem = ET.SubElement(root, "equality")

        all_joint_names = set()
        for joint_elem in root.iter("joint"):
            jname = joint_elem.get("name")
            jtype = joint_elem.get("type", "hinge")
            if jname and jtype == "hinge":
                all_joint_names.add(jname)

        locked_names = all_joint_names - set(active_joint_names)
        for jname in sorted(locked_names):
            eq = ET.SubElement(equality_elem, "joint")
            eq.set("joint1", jname)
            eq.set("polycoef", "0 0 0 0 0")

        print(f"[INFO] Added {len(locked_names)} equality constraints for inactive joints")

        # rock-stick penetration; contact pair stiffness must be softened
        import re as _re
        import os as _os
        script_dir = _os.path.dirname(_os.path.abspath(__file__))
        project_root = _os.path.dirname(script_dir)
        urdf_path_for_axes = (self.urdf_path if self.urdf_path is not None
                              else _os.path.join(project_root,
                                   f"config/scaled_registered_models/{self.date}/{self.subject_id}.urdf"))
        try:
            urdf_text = open(urdf_path_for_axes).read()
            urdf_axis = {}
            for m in _re.finditer(
                r'<joint name="([^"]+)"[^/]*?>.*?<axis xyz="([^"]+)"',
                urdf_text, _re.DOTALL):
                urdf_axis[m.group(1)] = m.group(2)
            patched = 0
            for joint_elem in root.iter("joint"):
                jn = joint_elem.get("name")
                if jn in urdf_axis and joint_elem.get("type", "hinge") == "hinge":
                    if joint_elem.get("axis") != urdf_axis[jn]:
                        joint_elem.set("axis", urdf_axis[jn])
                        patched += 1
            print(f"[INFO] Patched {patched} joint axes to match {urdf_path_for_axes}")
        except FileNotFoundError:
            print(f"[WARN] URDF not found at {urdf_path_for_axes}; joint axes left as-is")

        tree.write(output_file)
        print(f"[INFO] Saved pinned model to {output_file}")
        return output_file

    def launch_gui(self):
        """Launch MuJoCo passive viewer (non-blocking)"""
        self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
        self.viewer.cam.azimuth = 90
        self.viewer.cam.elevation = -20
        self.viewer.cam.distance = 3.0
        print("[INFO] MuJoCo viewer launched (passive).")

    def run_loop(self):
        print("\n[INFO] Starting MPPI loop.")

        mujoco.mj_resetData(self.mj_model, self.mj_data)
        for i, name in enumerate(["right_shoulder_Z","right_shoulder_X","right_shoulder_Y",
                                "right_elbow_Z","right_elbow_Y","right_wrist_Z","right_wrist_X"]):
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.mj_data.qpos[self.mj_model.jnt_qposadr[jid]] = self.q0[i]
        for qpos_id, dof_id, lock_val in self.locked_joint_constraints:
            self.mj_data.qpos[qpos_id] = lock_val
            self.mj_data.qvel[dof_id] = 0.0
            
        mujoco.mj_forward(self.mj_model, self.mj_data)

        f0 = self.controller._get_contact_force(self.mj_data)
        print(f"[INFO] Initial contact force: {f0:.4f} N  |  ncon: {self.mj_data.ncon}")
        for i in range(self.mj_data.ncon):
            c = self.mj_data.contact[i]
            b1 = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.mj_model.geom_bodyid[c.geom1])
            b2 = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.mj_model.geom_bodyid[c.geom2])
            print(f"         contact {i}: {b1} <-> {b2}")

        step = 0
        max_steps = 1000
        goal_tolerance = 0.02
        while step < max_steps:   

            if self.viewer is not None and not self.viewer.is_running():
                print("[INFO] Viewer closed.")
                break

            for qpos_id, dof_id, lock_val in self.locked_joint_constraints:
                self.mj_data.qpos[qpos_id] = lock_val
                self.mj_data.qvel[dof_id] = 0.0

            for qpos_id, dof_id, lock_val in self.locked_joint_constraints:
                self.mj_data.qpos[qpos_id] = lock_val
                self.mj_data.qvel[dof_id] = 0.0

            u_mppi = self.controller.step(
            self.mj_data.qpos.copy(),
            self.mj_data.qvel.copy()
            )
            self.mj_data.ctrl[:] = u_mppi

            if (not self.stick_static and self.stick_traj is not None
                    and self._stick_qposadr is not None):
                site_id_tmp = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "rock_contact_point")
                p_arm = self.mj_data.site_xpos[site_id_tmp]
                rail_vec = self.p_end_world - self.p_start_world
                rail_len = np.linalg.norm(rail_vec)
                if rail_len > 1e-6:
                    progress = np.dot(p_arm - self.p_start_world, rail_vec / rail_len) / rail_len
                    progress = np.clip(progress, 0.0, 1.0)
                    stick_idx = int(progress * (len(self.stick_traj) - 1))
                    self._update_stick_pos(stick_idx)
                else:
                    self._update_stick_pos(min(step, len(self.stick_traj) - 1))
            mujoco.mj_step(self.mj_model, self.mj_data)

            if self.viewer is not None:
                scn = self.viewer.user_scn
                scn.ngeom = 0

                def add_sphere(pos, radius, rgba):
                    if scn.ngeom >= scn.maxgeom: return
                    mujoco.mjv_initGeom(
                        scn.geoms[scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([radius, 0, 0]),
                        np.array(pos, dtype=np.float64),
                        np.eye(3).flatten(),
                        np.array(rgba, dtype=np.float32)
                    )
                    scn.ngeom += 1

                def add_line(p0, p1, width, rgba):
                    if scn.ngeom >= scn.maxgeom: return
                    mujoco.mjv_connector(
                        scn.geoms[scn.ngeom],
                        mujoco.mjtGeom.mjGEOM_LINE,
                        width, p0, p1
                    )
                    scn.geoms[scn.ngeom].rgba[:] = np.array(rgba, dtype=np.float32)
                    scn.ngeom += 1

                if hasattr(self.controller, 'all_trajs') and \
                   self.controller.all_trajs is not None:
                    for traj_k in self.controller.all_trajs:
                        for i in range(len(traj_k) - 1):
                            add_line(traj_k[i], traj_k[i+1],
                                     0.002, [0.6, 0.6, 0.6, 0.3])

                if hasattr(self.controller, 'best_traj') and \
                   self.controller.best_traj is not None:
                    traj = self.controller.best_traj
                    for i in range(len(traj) - 1):
                        add_line(traj[i], traj[i+1],
                                 0.006, [1, 1, 0, 1])

                site_id = mujoco.mj_name2id(
                    self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "rock_contact_point")
                if site_id >= 0:
                    add_sphere(self.mj_data.site_xpos[site_id], 0.015, [1, 1, 1, 1])

                add_sphere(self.p_start_world, 0.02, [0, 1, 0, 1])
                add_sphere(self.p_end_world,   0.02, [1, 0, 0, 1])

                self.viewer.sync()

            if step % 20 == 0:
                site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "rock_contact_point")
                p_site  = self.mj_data.site_xpos[site_id]
                force   = self.controller._get_contact_force(self.mj_data)
                dist_to_goal = np.linalg.norm(p_site - self.p_end_world)
                print(f"[step {step:4d}] force: {force:6.2f} N  |  dist_to_goal: {dist_to_goal*100:.2f} cm  |  ncon: {self.mj_data.ncon}")

            p_contact = self.mj_data.site_xpos[site_id].copy()
            dist_to_goal = np.linalg.norm(p_contact - self.p_end_world)
            
            if dist_to_goal < goal_tolerance:
                print(f"[SUCCESS] Goal reached at step {step}!")
                print(f"  Final distance: {dist_to_goal*100:.2f}cm")
                break
    
            step += 1

    def get_path_from_states(self, state_traj):
        """Converts a sequence of [q, v] states into [x, y, z] positions."""
        positions = []
        tmp_data = mujoco.MjData(self.mj_model) 
        
        for state in state_traj:
            tmp_data.qpos[:] = state[:self.mj_model.nq]
            mujoco.mj_forward(self.mj_model, tmp_data)
            hand_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "right_hand")
            positions.append(tmp_data.xpos[hand_id].copy())
        return positions
    

    def _estimate_stick_trajectory(self, q_traj):
        """
        Pre-compute stick center positions for each simulation step.

        The stick translates perpendicular to its long axis to track the
        contact point's off-axis drift, maintaining 90° contact with the
        (non-rotating) rock.  Along-stick motion (the shaving stroke) is
        NOT tracked — only the component normal to the stick axis.

        Returns (T+1, 3) array of stick center world positions, resampled
        to match self.T simulation steps.
        """
        z_stick = self.p_end_world - self.p_start_world
        z_stick = z_stick / np.linalg.norm(z_stick)

        temp_data = self.pin_model.createData()
        contact_positions = []
        for q in q_traj:
            pin.framesForwardKinematics(self.pin_model, temp_data, q)
            pin.updateFramePlacements(self.pin_model, temp_data)
            contact_positions.append(
                temp_data.oMf[self.contact_frame_id].translation.copy()
            )
        contact_positions = np.array(contact_positions)

        p_c0        = contact_positions[0]
        deltas      = contact_positions - p_c0
        deltas_perp = deltas - (deltas @ z_stick)[:, None] * z_stick
        stick_positions = self.p_stick + deltas_perp

        N      = len(stick_positions)
        T_sim  = self.T + 1
        t_demo = np.linspace(0, 1, N)
        t_sim  = np.linspace(0, 1, T_sim)
        return np.column_stack([
            np.interp(t_sim, t_demo, stick_positions[:, ax]) for ax in range(3)
        ])

    def _measure_stick_contact_offset(self, q_demo0):
        """
        Measure rock-stick penetration by replicating what the kinematic replay
        does: teleport arm joints to q_demo0, place stick at p_stick, run
        mj_step (contacts need the solver, not just mj_forward), read contact.

        Uses a scratch MjData so the real state is untouched.
        q_demo0 is q_traj[0] with shape (8,) — first 7 are arm joints in order:
          shoulder_Z, shoulder_X, shoulder_Y, elbow_Z, elbow_Y, wrist_Z, wrist_X
        """
        TARGET_PEN = -0.000

        ARM_JOINTS = list(ACTIVE_ARM_JOINTS)

        d = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, d)

        for i, jname in enumerate(ARM_JOINTS):
            mj_jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if mj_jid >= 0 and i < len(q_demo0):
                d.qpos[self.mj_model.jnt_qposadr[mj_jid]] = q_demo0[i]

        if hasattr(self, 'controller') and hasattr(self.controller, 'locked_joint_constraints'):
            for qpos_id, dof_id, lock_val in self.controller.locked_joint_constraints:
                d.qpos[qpos_id] = lock_val
                d.qvel[dof_id]  = 0.0

        if self._stick_qposadr is not None and hasattr(self, 'p_stick'):
            adr = self._stick_qposadr
            d.qpos[adr:adr + 3] = self.p_stick
            d.qvel[adr:adr + 6] = 0.0

        mujoco.mj_step(self.mj_model, d)

        rock_body_id  = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "rock")
        stick_body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "stick")
        if rock_body_id < 0 or stick_body_id < 0:
            print("[WARN] _measure_stick_contact_offset: rock or stick body not found")
            return np.zeros(3)

        print(f"[DEBUG] offset_meas: ncon={d.ncon}, "
              f"rock_pos={np.round(d.xpos[rock_body_id], 4)}, "
              f"stick_pos={np.round(d.xpos[stick_body_id], 4)}")

        for ci in range(d.ncon):
            c  = d.contact[ci]
            b1 = self.mj_model.geom_bodyid[c.geom1]
            b2 = self.mj_model.geom_bodyid[c.geom2]
            if {b1, b2} == {rock_body_id, stick_body_id}:
                pen = float(c.dist)
                if pen >= TARGET_PEN:
                    print(f"[INFO] stick_traj offset: no correction needed (pen={pen*1000:.1f} mm)")
                    return np.zeros(3)
                normal = c.frame[:3].copy()
                if b1 == rock_body_id:
                    normal = -normal
                correction_mag = abs(pen) - abs(TARGET_PEN)
                offset = -normal * correction_mag
                print(f"[INFO] stick_traj offset: {pen*1000:.1f} mm penetration → "
                      f"correcting by {np.linalg.norm(offset)*1000:.1f} mm "
                      f"(target residual={TARGET_PEN*1000:.1f} mm)")
                return offset

        print("[WARN] _measure_stick_contact_offset: no rock-stick contact at q0 — offset = 0")
        return np.zeros(3)

    def _update_stick_pos(self, t, data=None):
        """
        Set the stick freejoint position to stick_traj[t].
        No-op when stick_static=True or trajectory not computed.
        Pass data= to update a specific MjData (e.g. for force rollouts).
        """
        if self.stick_static or self.stick_traj is None or self._stick_qposadr is None:
            return
        data = data if data is not None else self.mj_data
        t_c  = min(int(t), len(self.stick_traj) - 1)
        adr  = self._stick_qposadr
        data.qpos[adr:adr + 3] = self.stick_traj[t_c]
        data.qvel[adr:adr + 3] = 0
        data.qvel[adr+3:adr+6] = 0
        # quaternion part (adr+3 : adr+7) is set by mj_resetData from XML and never changed

    def _extract_pin_state(self):
        """Read active joint (q, v) from MuJoCo into Pinocchio-ordered vectors."""
        active_names = list(ACTIVE_ARM_JOINTS)
        q = self.q0.copy()
        v = np.zeros(self.nv)
        for i, name in enumerate(active_names):
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                q[i] = self.mj_data.qpos[self.mj_model.jnt_qposadr[jid]]
                v[i] = self.mj_data.qvel[self.mj_model.jnt_dofadr[jid]]
        return q, v

    def set_warmstart_trajectory(self, xs_ref, method="id", desired_force=0.0,
                                 noise_fraction=1.0):
        """
        Warm-start MPPI from a reference trajectory.

        Parameters
        ----------
        xs_ref : (T_ref+1, nq+nv) reference trajectory (Pinocchio reduced state)
        method : 'grav', 'id' (inverse dynamics + optional J^T F),
                 or 'ct' (computed torque with PD tracking — forward-simulates)
        desired_force : contact force in N along rock→stick normal (for 'id' / 'ct')
        noise_fraction : fraction of joint torque limits used as noise sigma (default 1.0)

        Returns
        -------
        xs_mj : (H+1, nq+nv) MuJoCo trajectory produced during warm-start.
        """
        H  = self.controller.H
        nu = self.controller.nu

        active_names = list(ACTIVE_ARM_JOINTS)

        q_adrs, v_adrs = [], []
        for name in active_names:
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            q_adrs.append(self.mj_model.jnt_qposadr[jid])
            v_adrs.append(self.mj_model.jnt_dofadr[jid])
        q_adrs = np.array(q_adrs)
        v_adrs = np.array(v_adrs)

        xs_ref = np.asarray(xs_ref)
        ref_dt = self.dt
        dq_ref_all = xs_ref[:, self.nq : self.nq + nu]
        ddq_ref_all = np.gradient(dq_ref_all, ref_dt, axis=0)

        _LIMITS_BY_NAME_2 = {
            "middle_thoracic_X":      190.0, "right_clavicle_joint_X": 100.0,
            "right_shoulder_Z":        92.0, "right_shoulder_X":        71.0,
            "right_shoulder_Y":        52.0, "right_elbow_Z":           77.0,
            "right_elbow_Y":           15.0, "right_wrist_Z":          100.0,
            "right_wrist_X":          100.0,
        }
        limits = np.array([_LIMITS_BY_NAME_2[j] for j in ACTIVE_ARM_JOINTS])

        mujoco.mj_resetData(self.mj_model, self.mj_data)
        for i, qa in enumerate(q_adrs):
            self.mj_data.qpos[qa] = xs_ref[0, i]
        for qpos_id, dof_id, lock_val in self.locked_joint_constraints:
            self.mj_data.qpos[qpos_id] = lock_val
            self.mj_data.qvel[dof_id] = 0.0
        if not self.stick_static and self.stick_traj is not None \
                and self._stick_qposadr is not None:
            adr = self._stick_qposadr
            self.mj_data.qpos[adr:adr + 3] = self.stick_traj[0]
            self.mj_data.qvel[adr:adr + 6] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

        U_out = np.zeros((H, nu))
        xs_mj = []
        q, v = self._extract_pin_state()
        xs_mj.append(np.concatenate([q, v]))

        if method == "id":
            for t in range(H):
                ref_idx = min(t, len(xs_ref) - 1)
                q_ref   = xs_ref[ref_idx, :nu]
                dq_ref  = dq_ref_all[ref_idx]
                ddq_ref = ddq_ref_all[ref_idx]

                for i, (qa, va) in enumerate(zip(q_adrs, v_adrs)):
                    self.mj_data.qpos[qa] = q_ref[i]
                    self.mj_data.qvel[va] = dq_ref[i]
                    self.mj_data.qacc[va] = ddq_ref[i]

                for qpos_id, dof_id, lock_val in self.locked_joint_constraints:
                    self.mj_data.qpos[qpos_id] = lock_val
                    self.mj_data.qvel[dof_id] = 0.0
                    self.mj_data.qacc[dof_id] = 0.0

                if not self.stick_static and self.stick_traj is not None \
                        and self._stick_qposadr is not None:
                    frac = ref_idx / max(len(xs_ref) - 1, 1)
                    si = min(int(frac * (len(self.stick_traj) - 1)),
                             len(self.stick_traj) - 1)
                    adr = self._stick_qposadr
                    self.mj_data.qpos[adr:adr + 3] = self.stick_traj[si]
                    self.mj_data.qvel[adr:adr + 6] = 0.0

                mujoco.mj_forward(self.mj_model, self.mj_data)
                mujoco.mj_inverse(self.mj_model, self.mj_data)
                tau = self.mj_data.qfrc_inverse[v_adrs].copy()

                if desired_force > 0:
                    rock_site_id = mujoco.mj_name2id(
                        self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "rock_contact_point")
                    if rock_site_id >= 0:
                        rock_pos = self.mj_data.site_xpos[rock_site_id]
                        stick_bid = mujoco.mj_name2id(
                            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "stick")
                        stick_pos = self.mj_data.xpos[stick_bid]
                        d = stick_pos - rock_pos
                        n = np.linalg.norm(d)
                        contact_dir = d / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
                        J_full = np.zeros((3, self.mj_model.nv))
                        mujoco.mj_jacSite(self.mj_model, self.mj_data,
                                          J_full, None, rock_site_id)
                        J_arm = J_full[:, v_adrs]
                        tau += J_arm.T @ (contact_dir * desired_force)

                U_out[t] = np.clip(tau, -limits, limits)
                xs_mj.append(np.concatenate([
                    xs_ref[ref_idx, :self.nq], xs_ref[ref_idx, self.nq:]]))

        elif method == "ct":
            mj_dt  = self.mj_model.opt.timestep
            N_SUB  = max(1, round(self.dt / mj_dt))

            M_full_ws = np.zeros((self.mj_model.nv, self.mj_model.nv))
            mujoco.mj_fullM(self.mj_model, M_full_ws, self.mj_data.qM)
            M_diag = np.array([M_full_ws[d, d] for d in v_adrs])
            _arm_ix = np.ix_(v_adrs, v_adrs)

            KP_CT = np.minimum(limits * 5.0, 0.5 * 4.0 * M_diag / mj_dt**2)
            KD_CT = np.minimum(KP_CT * 0.15, 0.7 * 2.0 * M_diag / mj_dt)

            act_ids = []
            for jname in ACTIVE_ARM_JOINTS:
                aid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
                if aid < 0:
                    for suffix in ("_ctrl", "_motor", "_act", "_torque"):
                        aid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname + suffix)
                        if aid >= 0:
                            break
                act_ids.append(aid)
            act_ids = np.array(act_ids)

            ctrl_min = self.mj_model.actuator_ctrlrange[:, 0]
            ctrl_max = self.mj_model.actuator_ctrlrange[:, 1]

            rock_site_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "rock_contact_point")

            for t in range(H):
                ref_idx = min(t, len(xs_ref) - 1)
                q_ref   = xs_ref[ref_idx, :nu]
                dq_ref  = dq_ref_all[ref_idx]
                ddq_ref = ddq_ref_all[ref_idx]

                tau_last = np.zeros(nu)
                for _ in range(N_SUB):
                    q_cur  = self.mj_data.qpos[q_adrs]
                    dq_cur = self.mj_data.qvel[v_adrs]

                    desired_qacc = ddq_ref + KP_CT * (q_ref - q_cur) + KD_CT * (dq_ref - dq_cur)

                    mujoco.mj_fullM(self.mj_model, M_full_ws, self.mj_data.qM)
                    M_arm = M_full_ws[_arm_ix]
                    h = self.mj_data.qfrc_bias[v_adrs]
                    c = self.mj_data.qfrc_constraint[v_adrs]
                    tau = M_arm @ desired_qacc + h - c

                    if desired_force > 0 and rock_site_id >= 0:
                        J_full = np.zeros((3, self.mj_model.nv))
                        mujoco.mj_jacSite(self.mj_model, self.mj_data,
                                          J_full, None, rock_site_id)
                        J_arm = J_full[:, v_adrs]
                        stick_bid = mujoco.mj_name2id(
                            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "stick")
                        rock_pos  = self.mj_data.site_xpos[rock_site_id]
                        stick_pos = self.mj_data.xpos[stick_bid]
                        d = stick_pos - rock_pos
                        n = np.linalg.norm(d)
                        contact_dir = d / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
                        tau += J_arm.T @ (contact_dir * desired_force)

                    tau_last = np.clip(tau, -limits, limits)
                    for i, ai in enumerate(act_ids):
                        if ai >= 0:
                            self.mj_data.ctrl[ai] = np.clip(tau[i], ctrl_min[ai], ctrl_max[ai])

                    if not self.stick_static and self.stick_traj is not None \
                            and self._stick_qposadr is not None:
                        frac = ref_idx / max(len(xs_ref) - 1, 1)
                        si = min(int(frac * (len(self.stick_traj) - 1)),
                                 len(self.stick_traj) - 1)
                        adr = self._stick_qposadr
                        self.mj_data.qpos[adr:adr + 3] = self.stick_traj[si]
                        self.mj_data.qvel[adr:adr + 6] = 0.0

                    for qpos_id, dof_id, lock_val in self.locked_joint_constraints:
                        self.mj_data.qpos[qpos_id] = lock_val
                        self.mj_data.qvel[dof_id] = 0.0

                    mujoco.mj_step(self.mj_model, self.mj_data)

                U_out[t] = tau_last
                q, v = self._extract_pin_state()
                xs_mj.append(np.concatenate([q, v]))

            print(f"  [ct] N_SUBSTEPS={N_SUB}  KP range=[{KP_CT.min():.0f}, {KP_CT.max():.0f}]  "
                  f"KD range=[{KD_CT.min():.0f}, {KD_CT.max():.0f}]")

        else:
            _q_traj = xs_ref[:, :nu]
            U_out = self.controller.warmstart_from_demo(
                _q_traj,
                locked_joint_constraints=self.locked_joint_constraints,
                stick_traj=self.stick_traj,
                stick_qposadr=self._stick_qposadr,
            )
            for t in range(H):
                ref_idx = min(t, len(xs_ref) - 1)
                xs_mj.append(np.concatenate([
                    xs_ref[ref_idx, :self.nq], xs_ref[ref_idx, self.nq:]]))

        self.controller.U = U_out
        self.controller.u_prev = U_out[0].copy()

        mujoco.mj_resetData(self.mj_model, self.mj_data)
        dq0 = xs_ref[0, self.nq : self.nq + nu]
        for i, (qa, va) in enumerate(zip(q_adrs, v_adrs)):
            self.mj_data.qpos[qa] = xs_ref[0, i]
            self.mj_data.qvel[va] = dq0[i]
        for qpos_id, dof_id, lock_val in self.locked_joint_constraints:
            self.mj_data.qpos[qpos_id] = lock_val
            self.mj_data.qvel[dof_id] = 0.0
        if not self.stick_static and self.stick_traj is not None \
                and self._stick_qposadr is not None:
            adr = self._stick_qposadr
            self.mj_data.qpos[adr:adr + 3] = self.stick_traj[0]
            self.mj_data.qvel[adr:adr + 6] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self._warmstart_set = True

        self.controller.set_noise_from_limits(noise_fraction, limits)

        print(f"[warmstart-{method}] done — H={H} | "
              f"U range=[{U_out.min():.1f}, {U_out.max():.1f}] Nm"
              f" | desired_force={desired_force:.0f}N")
        return np.array(xs_mj)

    def replay_warmstart(self, xs_ref=None, desired_force=0.0):
        """
        Re-run the closed-loop CT controller in the viewer to verify the
        warmstart visually.  This is the same PD tracking loop that
        set_warmstart_trajectory(..., method='ct') uses internally.

        Parameters
        ----------
        xs_ref : (T_ref+1, nq+nv) reference trajectory — required for CT replay.
        desired_force : J^T F feedforward magnitude (same as warmstart call).
        """
        import time as _time

        if xs_ref is None:
            raise ValueError("replay_warmstart requires xs_ref for the CT reference")

        xs_ref = np.asarray(xs_ref)
        nu = len(ACTIVE_ARM_JOINTS)

        active_names = list(ACTIVE_ARM_JOINTS)
        q_adrs = np.array([self.mj_model.jnt_qposadr[
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in active_names])
        v_adrs = np.array([self.mj_model.jnt_dofadr[
            mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            for n in active_names])

        act_ids = []
        for jname in active_names:
            aid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname)
            if aid < 0:
                for suffix in ("_ctrl", "_motor", "_act", "_torque"):
                    aid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_ACTUATOR, jname + suffix)
                    if aid >= 0:
                        break
            act_ids.append(aid)
        act_ids = np.array(act_ids)
        ctrl_min = self.mj_model.actuator_ctrlrange[:, 0]
        ctrl_max = self.mj_model.actuator_ctrlrange[:, 1]
        _LIMITS_BY_NAME_2 = {
            "middle_thoracic_X":      190.0, "right_clavicle_joint_X": 100.0,
            "right_shoulder_Z":        92.0, "right_shoulder_X":        71.0,
            "right_shoulder_Y":        52.0, "right_elbow_Z":           77.0,
            "right_elbow_Y":           15.0, "right_wrist_Z":          100.0,
            "right_wrist_X":          100.0,
        }
        limits = np.array([_LIMITS_BY_NAME_2[j] for j in ACTIVE_ARM_JOINTS])

        mujoco.mj_resetData(self.mj_model, self.mj_data)
        for i, qa in enumerate(q_adrs):
            self.mj_data.qpos[qa] = xs_ref[0, i]
        dq0 = xs_ref[0, self.nq:self.nq + nu]
        for i, va in enumerate(v_adrs):
            self.mj_data.qvel[va] = dq0[i]
        for qpos_id, dof_id, lock_val in self.locked_joint_constraints:
            self.mj_data.qpos[qpos_id] = lock_val
            self.mj_data.qvel[dof_id] = 0.0
        if not self.stick_static and self.stick_traj is not None \
                and self._stick_qposadr is not None:
            adr = self._stick_qposadr
            self.mj_data.qpos[adr:adr + 3] = self.stick_traj[0]
            self.mj_data.qvel[adr:adr + 6] = 0.0
        mujoco.mj_forward(self.mj_model, self.mj_data)

        mj_dt = self.mj_model.opt.timestep
        N_SUB = max(1, round(self.dt / mj_dt))
        M_full = np.zeros((self.mj_model.nv, self.mj_model.nv))
        mujoco.mj_fullM(self.mj_model, M_full, self.mj_data.qM)
        M_diag = np.array([M_full[d, d] for d in v_adrs])
        _arm_ix = np.ix_(v_adrs, v_adrs)
        KP_CT = np.minimum(limits * 5.0, 0.5 * 4.0 * M_diag / mj_dt**2)
        KD_CT = np.minimum(KP_CT * 0.15, 0.7 * 2.0 * M_diag / mj_dt)

        dq_ref_all  = xs_ref[:, self.nq:self.nq + nu]
        ddq_ref_all = np.gradient(dq_ref_all, self.dt, axis=0)

        H = self.controller.H
        rock_site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "rock_contact_point")

        print(f"[replay-ct] N_SUB={N_SUB}  KP=[{KP_CT.min():.0f},{KP_CT.max():.0f}]  "
              f"KD=[{KD_CT.min():.0f},{KD_CT.max():.0f}]")

        with mujoco.viewer.launch_passive(self.mj_model, self.mj_data) as viewer:
            for t in range(H):
                ref_idx = min(t, len(xs_ref) - 1)
                q_ref   = xs_ref[ref_idx, :nu]
                dq_ref  = dq_ref_all[ref_idx]
                ddq_ref = ddq_ref_all[ref_idx]

                for _ in range(N_SUB):
                    q_cur  = self.mj_data.qpos[q_adrs]
                    dq_cur = self.mj_data.qvel[v_adrs]

                    desired_qacc = ddq_ref + KP_CT * (q_ref - q_cur) + KD_CT * (dq_ref - dq_cur)

                    mujoco.mj_fullM(self.mj_model, M_full, self.mj_data.qM)
                    M_arm = M_full[_arm_ix]
                    h = self.mj_data.qfrc_bias[v_adrs]
                    c = self.mj_data.qfrc_constraint[v_adrs]
                    tau = M_arm @ desired_qacc + h - c

                    if desired_force > 0 and rock_site_id >= 0:
                        J_full = np.zeros((3, self.mj_model.nv))
                        mujoco.mj_jacSite(self.mj_model, self.mj_data,
                                          J_full, None, rock_site_id)
                        J_arm = J_full[:, v_adrs]
                        stick_bid = mujoco.mj_name2id(
                            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "stick")
                        rock_pos  = self.mj_data.site_xpos[rock_site_id]
                        stick_pos = self.mj_data.xpos[stick_bid]
                        d = stick_pos - rock_pos
                        n = np.linalg.norm(d)
                        contact_dir = d / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])
                        tau += J_arm.T @ (contact_dir * desired_force)

                    for i, ai in enumerate(act_ids):
                        if ai >= 0:
                            self.mj_data.ctrl[ai] = np.clip(tau[i], ctrl_min[ai], ctrl_max[ai])

                    if not self.stick_static and self.stick_traj is not None \
                            and self._stick_qposadr is not None:
                        frac = ref_idx / max(len(xs_ref) - 1, 1)
                        si = min(int(frac * (len(self.stick_traj) - 1)),
                                 len(self.stick_traj) - 1)
                        adr = self._stick_qposadr
                        self.mj_data.qpos[adr:adr + 3] = self.stick_traj[si]
                        self.mj_data.qvel[adr:adr + 6] = 0.0

                    for qpos_id, dof_id, lock_val in self.locked_joint_constraints:
                        self.mj_data.qpos[qpos_id] = lock_val
                        self.mj_data.qvel[dof_id] = 0.0

                    mujoco.mj_step(self.mj_model, self.mj_data)

                if t % max(1, H // 10) == 0:
                    f = self.controller._get_contact_force(self.mj_data)
                    q_cur = self.mj_data.qpos[q_adrs]
                    err = np.max(np.abs(q_ref - q_cur))
                    print(f"  [replay t={t:4d}] force={f:.2f}N  max_q_err={err:.4f}")

                viewer.sync()
                _time.sleep(self.dt)

            print("[replay-ct] done — close viewer window to continue")
            while viewer.is_running():
                viewer.sync()
                _time.sleep(0.05)

    def solve(self, xs_init=None, us_init=None, w_run_windows=None, w_term_windows=None, visualize=False):
        if w_run_windows is None and hasattr(self, '_w_run_windows'):
            w_run_windows = self._w_run_windows
            w_term_windows = self._w_term_windows

        self.controller.current_t = 0
        if not getattr(self, '_warmstart_set', False):
            self.controller.U = np.zeros((self.controller.H, self.controller.nu))

            mujoco.mj_resetData(self.mj_model, self.mj_data)
            for i, name in enumerate(ACTIVE_ARM_JOINTS):
                jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
                self.mj_data.qpos[self.mj_model.jnt_qposadr[jid]] = self.q0[i]
            for qpos_id, dof_id, lock_val in self.locked_joint_constraints:
                self.mj_data.qpos[qpos_id] = lock_val
                self.mj_data.qvel[dof_id] = 0.0
            if not self.stick_static and self.stick_traj is not None \
                    and self._stick_qposadr is not None:
                adr = self._stick_qposadr
                self.mj_data.qpos[adr:adr + 3] = self.stick_traj[0]
                self.mj_data.qvel[adr:adr + 6] = 0.0
            mujoco.mj_forward(self.mj_model, self.mj_data)
        self._warmstart_set = False

        xs = []
        us = []

        n_w = len(w_run_windows) if w_run_windows is not None else 1
        window_size = max(1, self.T // n_w)

        q, v = self._extract_pin_state()
        xs.append(np.concatenate([q, v]))

        _viewer = None
        if visualize:
            _viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)
            _viewer.cam.azimuth = 90
            _viewer.cam.elevation = -20
            _viewer.cam.distance = 3.0

        force_log = []
        cost_log = {k: [] for k in self.controller.w_run}
        site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, "rock_contact_point")
        rail_vec = self.p_end_world - self.p_start_world
        rail_len = np.linalg.norm(rail_vec)
        rail_unit = rail_vec / rail_len if rail_len > 1e-6 else np.zeros(3)

        for t in range(self.T):
            if w_run_windows is not None:
                k = min(t // window_size, n_w - 1)
                w_run_k = dict(zip(self.keys_run, w_run_windows[k]))
                w_term_k = dict(zip(self.keys_term, w_term_windows[k])) if w_term_windows is not None else self.w_term
                self.update_solver_weights(w_run_k, w_term_k)

            for qpos_id, dof_id, lock_val in self.locked_joint_constraints:
                self.mj_data.qpos[qpos_id] = lock_val
                self.mj_data.qvel[dof_id] = 0.0

            u = self.controller.step(
                self.mj_data.qpos.copy(),
                self.mj_data.qvel.copy()
            )
            self.mj_data.ctrl[:] = u
            us.append(u.copy())

            if not self.stick_static and self.stick_traj is not None \
                    and self._stick_qposadr is not None:
                sidx = min(t, len(self.stick_traj) - 1)
                adr = self._stick_qposadr
                self.mj_data.qpos[adr:adr + 3] = self.stick_traj[sidx]
                self.mj_data.qvel[adr:adr + 6] = 0.0

            mujoco.mj_step(self.mj_model, self.mj_data)

            if _viewer is not None and _viewer.is_running():
                scn = _viewer.user_scn
                scn.ngeom = 0
                if hasattr(self.controller, 'best_traj') and self.controller.best_traj is not None:
                    traj = self.controller.best_traj
                    for i in range(len(traj) - 1):
                        if scn.ngeom < scn.maxgeom:
                            mujoco.mjv_connector(scn.geoms[scn.ngeom],
                                                 mujoco.mjtGeom.mjGEOM_LINE,
                                                 0.006, traj[i], traj[i+1])
                            scn.geoms[scn.ngeom].rgba[:] = np.array([1,1,0,1], dtype=np.float32)
                            scn.ngeom += 1
                if site_id >= 0:
                    if scn.ngeom < scn.maxgeom:
                        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                            np.array([0.015,0,0]), self.mj_data.site_xpos[site_id].copy(),
                                            np.eye(3).flatten(), np.array([1,1,1,1], dtype=np.float32))
                        scn.ngeom += 1
                for p, rgba in [(self.p_start_world, [0,1,0,1]), (self.p_end_world, [1,0,0,1])]:
                    if scn.ngeom < scn.maxgeom:
                        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                                            np.array([0.02,0,0]), np.array(p, dtype=np.float64),
                                            np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
                        scn.ngeom += 1
                _viewer.sync()
            elif _viewer is not None and not _viewer.is_running():
                print(f"[solve] Viewer closed at t={t}")
                _viewer = None

            w = self.controller.w_run
            u_prev = us[-2] if len(us) >= 2 else u
            raw = self.controller._raw_run_features(self.mj_data, u, u_prev, t)
            for k in cost_log:
                cost_log[k].append(w.get(k, 0.0) * raw.get(k, 0.0))

            f = self.controller._get_contact_force(self.mj_data)
            force_log.append(f)

            if t % 50 == 0:
                p_site = self.mj_data.site_xpos[site_id] if site_id >= 0 else np.zeros(3)
                progress_pct = np.dot(p_site - self.p_start_world, rail_unit) / rail_len * 100 if rail_len > 1e-6 else 0.0
                top3 = sorted(((k, cost_log[k][-1]) for k in cost_log), key=lambda x: abs(x[1]), reverse=True)[:3]
                cost_str = "  ".join(f"{k}={v:.3g}" for k, v in top3)
                print(f"  [t={t:4d}] force={f:6.2f}N  progress={progress_pct:.1f}%  |u|={np.linalg.norm(u):.1f}  costs: {cost_str}")

            q, v = self._extract_pin_state()
            xs.append(np.concatenate([q, v]))

        self._last_xs = np.array(xs)
        self._last_us = np.array(us)
        self._last_forces = np.array(force_log)
        self._last_costs = {k: np.array(v) for k, v in cost_log.items()}

        if _viewer is not None:
            _viewer.close()

        f_arr = np.array(force_log)
        print(f"\n[solve] done — force: mean={f_arr.mean():.2f}N, max={f_arr.max():.2f}N, "
              f"nonzero={np.count_nonzero(f_arr > 0.1)}/{len(f_arr)} steps")
        return np.array(xs), np.array(us)

    def _simulate_press_forces(self, xs, us):
        """
        Forward-simulate the trajectory with mj_step to collect contact forces.
        mj_forward alone cannot produce contact forces — the simulation must be
        stepped so that the solver resolves contact impulses.
        Returns array of shape (T,) with the rock-stick normal force at each step.
        """
        T = min(len(us), len(xs) - 1)
        forces = np.zeros(T)
        d_sim = mujoco.MjData(self.mj_model)
        mujoco.mj_resetData(self.mj_model, d_sim)
        q0 = xs[0][:self.nq]
        v0 = xs[0][self.nq:]
        self.controller._set_state(d_sim, q0, v0)
        self._update_stick_pos(0, data=d_sim)
        mujoco.mj_forward(self.mj_model, d_sim)
        for t in range(T):
            u = np.array(us[t])[:self.mj_model.nu]
            d_sim.ctrl[:self.mj_model.nu] = u
            if not self.stick_static and self.stick_traj is not None and self._stick_qposadr is not None:
                ref_frac  = t / max(T - 1, 1)
                stick_idx = min(int(ref_frac * (len(self.stick_traj) - 1)), len(self.stick_traj) - 1)
                adr = self._stick_qposadr
                d_sim.qpos[adr:adr + 3] = self.stick_traj[stick_idx]
                d_sim.qvel[adr:adr + 6] = 0.0
            mujoco.mj_step(self.mj_model, d_sim)
            forces[t] = self.controller._get_contact_force(d_sim)
        return forces

    def get_irl_gradient(self, xs_demo, us_demo, n_w=1, w_run_windows=None, w_term_windows=None,
                         grad_clip=1e3, n_mppi_iters=1):
        """
        Compute IRL gradient using full-horizon MPPI.

        The MPPI horizon is set to self.T so each rollout covers the entire trajectory.
        TV weights (one per window) are applied inside the rollout at the correct timestep.
        update() is called n_mppi_iters times from x0 to optimise U before gradient extraction.

        Gradient (per window k):
            g_k = phi_demo_k - E_w[phi_k]

        where phi_demo_k / E_w[phi_k] are the summed features for timesteps in window k.

        For n_w=1 returns (grad_run, grad_term) of shape (nr_run,) / (nr_term,).
        For n_w>1 returns stacked (n_w*nr_run,) / (n_w*nr_term,).
        """
        phi_demo_full, Phis_demo, _, _ = self.get_traj_features(xs_demo, us_demo)

        T           = min(len(us_demo), self.T)
        nr          = self.nr
        window_size = max(1, T // n_w)

        phi_demo_windows = np.zeros((n_w, nr))
        for t, phi_t in enumerate(Phis_demo[:T]):
            k = min(t // window_size, n_w - 1)
            phi_demo_windows[k] += phi_t * self.dt
        phi_demo_windows[-1] += Phis_demo[-1]

        if w_run_windows is not None:
            w_run_dicts  = [dict(zip(self.keys_run,  w_run_windows[k]))  for k in range(n_w)]
            w_term_dicts = [dict(zip(self.keys_term, w_term_windows[k])) for k in range(n_w)] \
                           if w_term_windows is not None else None
            self.controller.set_tv_weights(w_run_dicts, w_term_dicts)
        else:
            self.controller.set_tv_weights(None)

        orig_H = self.controller.H
        self.controller.H = T
        if self.controller.U.shape[0] != T:
            U_new = np.zeros((T, self.nu))
            n_copy = min(self.controller.U.shape[0], T)
            U_new[:n_copy] = self.controller.U[:n_copy]
            self.controller.U = U_new
        self.controller.noise_schedule = self.controller._build_noise_schedule()
        if self._stick_traj_ctrl is not None and self._stick_qposadr_ctrl is not None:
            self.controller.set_stick_traj(self._stick_traj_ctrl, self._stick_qposadr_ctrl)

        mujoco.mj_resetData(self.mj_model, self.mj_data)
        for i, name in enumerate(ACTIVE_ARM_JOINTS):
            jid = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self.mj_data.qpos[self.mj_model.jnt_qposadr[jid]] = self.q0[i]
                self.mj_data.qvel[self.mj_model.jnt_dofadr[jid]]  = 0.0
        for qpos_id, dof_id, lock_val in self.locked_joint_constraints:
            self.mj_data.qpos[qpos_id] = lock_val
            self.mj_data.qvel[dof_id]  = 0.0
        self._update_stick_pos(0)
        mujoco.mj_forward(self.mj_model, self.mj_data)

        q0_pos = self.mj_data.qpos.copy()
        q0_vel = self.mj_data.qvel.copy()
        for _ in range(n_mppi_iters):
            E_phi_full = self.controller.update(q0_pos, q0_vel)

        # update() accumulates phi over the full rollout, but doesn't break by window.
        if n_w == 1:
            E_phi_windows = np.array([E_phi_full])
        else:
            E_phi_windows = np.zeros((n_w, nr))
            for k in range(n_w):
                if w_run_dicts is not None:
                    self.controller.set_tv_weights([w_run_dicts[k]] * n_w,
                                                   [w_term_dicts[k]] * n_w if w_term_dicts else None)
                E_phi_k = self.controller.update(q0_pos, q0_vel)
                E_phi_windows[k] = E_phi_k

        local_grad = phi_demo_windows - E_phi_windows

        self._last_gradient_info = {
            'phi_demo_windows':  phi_demo_windows.copy(),
            'E_phi_windows':     E_phi_windows.copy(),
            'diff_windows':      local_grad.copy(),
            'phi_keys':          self.keys_run + self.keys_term,
        }

        grad_run  = np.clip(local_grad[:, :self.nr_run].flatten(),  -grad_clip, grad_clip)
        grad_term = np.clip(local_grad[:, self.nr_run:].flatten(), -grad_clip, grad_clip)

        self.controller.H = orig_H
        self.controller.noise_schedule = self.controller._build_noise_schedule()
        self.controller.set_tv_weights(None)
        self.controller.set_stick_traj(
            self._stick_traj_ctrl, self._stick_qposadr_ctrl
        ) if self._stick_traj_ctrl is not None else None

        return grad_run, grad_term

    def set_demo_forces(self, forces, xs_demo):
        """
        Inject measured contact forces for the demo trajectory.
        When get_traj_features() is called on a trajectory whose initial
        AND terminal states match this demo's, it uses these cached
        forces instead of re-simulating.

        Comparing both endpoints (not just x0) is important — every IRL
        challenger trajectory in MPPI/Crocoddyl starts from the same x0
        as the demo, so an x0-only check would misclassify challengers
        as the demo, give them the demo's cached forces, and zero out
        any press_force IRL gradient.
        """
        self._demo_forces = np.array(forces)
        self._demo_xs0 = np.array(xs_demo[0])
        self._demo_xsT = np.array(xs_demo[-1])

    def set_force_target(self, profile):
        """Per-timestep target for the press_force feature.

        profile : (T,) array — overrides the scalar self.target_force per
                  step inside the press_force cost. Pass None to clear and
                  fall back to the scalar.

        Also propagates to the kinematic solver (self._kin_mppi) if one has
        been installed, so the MPPI rollout's per-horizon target_sched
        matches the demo's per-step target. Without this, demo and rollout
        would compare against different targets (demo per-step profile,
        rollout constant) — phantom IRL signal on press_force.
        """
        prof = (None if profile is None
                else np.asarray(profile, dtype=np.float64))
        self.target_force_profile = prof
        kin = getattr(self, '_kin_mppi', None)
        if kin is not None and hasattr(kin, 'set_force_target_profile'):
            kin.set_force_target_profile(prof)

    def get_tau_from_trajectory(self, xs, force_profile=None, force_gain=0.25,
                                force_dist_threshold=0.05,
                                record_contact_aware=False,
                                return_qtrack=False,
                                force_regulate=False, force_reg_kp=0.4,
                                force_reg_ki=0.15):
        """
        Compute joint torques by PD-tracking the trajectory in MuJoCo,
        with optional contact force feedforward.

        Uses computed torque control: tau = M*a_desired + h - c + J^T*f
        where:
          - M*a_desired + h: dynamics (gravity, Coriolis, inertia)
          - -c: cancel MuJoCo's contact constraint (prevents double-counting)
          - J^T*f: force feedforward toward stick (from measured profile)

        Parameters
        ----------
        xs            : (T+1, 2*nq) state trajectory [q_arm|rock, dq_arm|rock]
        force_profile : (T,) measured normal force per timestep (positive = pressing),
                        or None for no force feedforward
        force_gain    : float, scale factor for force feedforward (default 0.25)
        force_dist_threshold : float, only apply force when rock is within this
                               distance of stick (meters)

        Returns
        -------
        us     : (T, nu_mj) recorded ctrl values (joint torques)
        forces : (T,) contact force at each step (from sensor)
        """
        T = len(xs) - 1
        nu_mj = self.mj_model.nu
        nj = nu_mj
        model = self.mj_model
        data = mujoco.MjData(model)
        data_ref = mujoco.MjData(model)
        arm_dof = self.controller.arm_dofadr
        arm_qpos = self.controller.arm_qposadr

        act_ids = []
        for j in range(nj):
            jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT,
                                          model.actuator_trnid[j, 0])
            act_ids.append(j)

        ctrl_min = model.actuator_ctrlrange[:, 0]
        ctrl_max = model.actuator_ctrlrange[:, 1]

        _LIMITS_BY_NAME = {
            "middle_thoracic_X":      190.0,
            "right_clavicle_joint_X": 100.0,
            "right_shoulder_Z":        92.0,
            "right_shoulder_X":        71.0,
            "right_shoulder_Y":        52.0,
            "right_elbow_Z":           77.0,
            "right_elbow_Y":           15.0,
            "right_wrist_Z":          100.0,
            "right_wrist_X":          100.0,
        }
        _LIMITS = np.array([_LIMITS_BY_NAME[j] for j in ACTIVE_ARM_JOINTS])
        M_full = np.zeros((model.nv, model.nv))
        mj_dt = model.opt.timestep
        N_SUBSTEPS = max(1, round(self.dt / mj_dt))

        nq_xs = xs.shape[1] // 2

        data.qpos[:] = self.mujoco_init_q.copy()
        data.qvel[:] = 0.0
        for i in range(nj):
            data.qpos[arm_qpos[i]] = xs[0, i]
            data.qvel[arm_dof[i]] = xs[0, nq_xs + i]
        if not self.stick_static and self.stick_traj is not None \
                and self._stick_qposadr is not None:
            adr = self._stick_qposadr
            data.qpos[adr:adr + 3] = self.stick_traj[0]
            data.qvel[adr:adr + 6] = 0.0
        mujoco.mj_forward(model, data)

        mujoco.mj_fullM(model, M_full, data.qM)
        M_diag = np.array([M_full[d, d] for d in arm_dof])
        KP = np.minimum(_LIMITS * 5.0, 0.5 * 4.0 * M_diag / mj_dt**2)
        KD = np.minimum(KP * 0.15, 0.7 * 2.0 * M_diag / mj_dt)
        if getattr(self, '_no_pd', False):
            KP = KP * 0.0; KD = KD * 0.0

        ddq_demo = np.gradient(xs[:, nq_xs:nq_xs + nj], self.dt, axis=0)

        _locked_arm = np.where(np.asarray(model.dof_armature)[arm_dof] >= 1e6)[0]
        _q_lock = np.asarray(xs[0, :nj])[_locked_arm] if len(_locked_arm) else None

        _arm_ix = np.ix_(arm_dof, arm_dof)

        stick_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stick")
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                      "rock_contact_point")
        rock_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "rock_sphere")
        stick_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "stick_geom")

        us_out = np.zeros((T, nj))
        forces_out = np.zeros(T)
        _sdf_forces = np.zeros(T)
        td_ff = np.zeros((T, nj)); td_pd = np.zeros((T, nj)); td_press = np.zeros((T, nj))
        _q_home_null = np.asarray(getattr(self, '_contact_pd_home', xs[0, :nj]), dtype=float)
        qtrack_out = np.zeros((T, nj))
        vtrack_out = np.zeros((T, nj))

        _integ = 0.0

        for t in range(T):
            q_ref = xs[t, :nj]
            dq_ref = xs[t, nq_xs:nq_xs + nj]
            ddq_ref = ddq_demo[t] if t < len(ddq_demo) else ddq_demo[-1]
            if len(_locked_arm):
                q_ref = np.asarray(q_ref).copy(); q_ref[_locked_arm] = _q_lock
                dq_ref = np.asarray(dq_ref).copy(); dq_ref[_locked_arm] = 0.0
                ddq_ref = np.asarray(ddq_ref).copy(); ddq_ref[_locked_arm] = 0.0

            _reg_F = None
            if force_regulate and force_profile is not None:
                _ftgt = force_profile[t] if t < len(force_profile) else force_profile[-1]
                if t > 0:
                    _a = float(getattr(self, '_force_reg_meas_ema', 0.5))
                    self._fmeas_ema = (1 - _a) * getattr(self, '_fmeas_ema', forces_out[0]) + _a * forces_out[t - 1]
                else:
                    self._fmeas_ema = _ftgt
                _err = _ftgt - self._fmeas_ema
                _integ += force_reg_ki * _err
                _reg_F = float(np.clip(_ftgt + force_reg_kp * _err + _integ, 0.0, 100.0))

            for _ in range(N_SUBSTEPS):
                q_cur = data.qpos[arm_qpos]
                dq_cur = data.qvel[arm_dof]

                if force_profile is not None and stick_body_id >= 0:
                    rock_pos = data.site_xpos[site_id].copy()
                    stick_center = data.xpos[stick_body_id].copy()
                    stick_axis = data.xmat[stick_body_id].reshape(3, 3)[:, 2]
                    t_proj = np.clip(np.dot(rock_pos - stick_center, stick_axis),
                                     -0.4, 0.4)
                    nearest = stick_center + t_proj * stick_axis
                    toward_stick = nearest - rock_pos
                    dist = np.linalg.norm(toward_stick)
                    force_dir = toward_stick / dist if dist > 1e-6 else np.array([0, 0, -1.0])

                    f_desired_t = force_profile[t] if t < len(force_profile) else force_profile[-1]
                    if dist < force_dist_threshold and f_desired_t > 0.5:
                        jac_p = np.zeros((3, model.nv))
                        mujoco.mj_jacSite(model, data, jac_p, None, site_id)
                        J_arm = jac_p[:, arm_dof]
                        _appliedF = _reg_F if _reg_F is not None else force_gain * f_desired_t
                        tau_force = J_arm.T @ (_appliedF * force_dir)
                    else:
                        tau_force = np.zeros(nj)
                else:
                    tau_force = np.zeros(nj)

                mujoco.mj_fullM(model, M_full, data.qM)
                M_arm = M_full[_arm_ix]
                h = data.qfrc_bias[arm_dof]
                c = data.qfrc_constraint[arm_dof]
                _dik = float(getattr(self, '_diff_ik', 0.0))
                _cpd = float(getattr(self, '_contact_pd', 0.0))
                if _dik > 0.0 and site_id >= 0:
                    data_ref.qpos[:] = data.qpos
                    data_ref.qpos[arm_qpos] = q_ref
                    mujoco.mj_forward(model, data_ref)
                    x_ref = data_ref.site_xpos[site_id].copy()
                    x_cur = data.site_xpos[site_id].copy()
                    _jc = np.zeros((3, model.nv))
                    mujoco.mj_jacSite(model, data, _jc, None, site_id)
                    J_c = _jc[:, arm_dof]
                    _lam = float(getattr(self, '_diff_ik_lambda', 0.01))
                    _dqik = J_c.T @ np.linalg.solve(J_c @ J_c.T + _lam ** 2 * np.eye(3), x_ref - x_cur)
                    _pd_term = M_arm @ (_dik * KP * _dqik - KD * dq_cur)
                    desired_qacc = ddq_ref
                elif _cpd > 0.0 and site_id >= 0:
                    data_ref.qpos[:] = data.qpos
                    data_ref.qpos[arm_qpos] = q_ref
                    mujoco.mj_forward(model, data_ref)
                    x_ref = data_ref.site_xpos[site_id].copy()
                    x_cur = data.site_xpos[site_id].copy()
                    _jc = np.zeros((3, model.nv))
                    mujoco.mj_jacSite(model, data, _jc, None, site_id)
                    J_c = _jc[:, arm_dof]
                    v_cur = J_c @ dq_cur
                    F_task = _cpd * (x_ref - x_cur) - float(getattr(self, '_contact_pd_kd', 0.0)) * v_cur
                    _pd_term = J_c.T @ F_task
                    _kpn = float(getattr(self, '_contact_pd_nullkp', 0.0))
                    _kdn = float(getattr(self, '_contact_pd_nullkd', 0.0))
                    if _kpn > 0.0 or _kdn > 0.0:
                        _reg = float(getattr(self, '_contact_pd_reg', 1e-3))
                        _Minv = np.linalg.inv(M_arm + 1e-6 * np.eye(M_arm.shape[0]))
                        _Lam = np.linalg.inv(J_c @ _Minv @ J_c.T + _reg * np.eye(3))
                        _Jbar = _Minv @ J_c.T @ _Lam
                        _NT = np.eye(J_c.shape[1]) - J_c.T @ _Jbar.T
                        _tau_post = _kpn * (_q_home_null - q_cur) - _kdn * dq_cur
                        _pd_term = _pd_term + _NT @ _tau_post
                    desired_qacc = ddq_ref
                else:
                    desired_qacc = ddq_ref
                    _pd_term = M_arm @ (KP * (q_ref - q_cur) + KD * (dq_ref - dq_cur))
                _vdz_src = getattr(self, '_virtual_dz', 0.0)
                _vdz = float(_vdz_src[min(t, len(_vdz_src) - 1)]) if hasattr(_vdz_src, '__len__') else float(_vdz_src)
                if _vdz > 0.0 and stick_geom_id >= 0 and rock_geom_id >= 0:
                    _rsum = float(model.geom_size[rock_geom_id][0] + model.geom_size[stick_geom_id][0])
                    _shalf = float(model.geom_size[stick_geom_id][1])
                    _rp = data.geom_xpos[rock_geom_id].copy(); _sc = data.geom_xpos[stick_geom_id].copy()
                    _sa = data.geom_xmat[stick_geom_id].reshape(3, 3)[:, 2]
                    _tp = np.clip(np.dot(_rp - _sc, _sa), -_shalf, _shalf)
                    _av = _rp - (_sc + _tp * _sa); _ad = np.linalg.norm(_av)
                    # ANALYTIC SDF (point-to-cylinder): continuous, always defined. We DO NOT read
                    _sgap = _ad - _rsum
                    _err = _sgap + _vdz
                    if _sgap < force_dist_threshold and _err > 0.0 and _ad > 1e-6:
                        _nout = _av / _ad
                        _vtkp = float(getattr(self, '_vt_kp', 2500.0))
                        _jvt = np.zeros((3, model.nv)); mujoco.mj_jacSite(model, data, _jvt, None, site_id)
                        _Jc = _jvt[:, arm_dof]
                        _MiJn = np.linalg.solve(M_arm, _Jc.T @ _nout)
                        _lam = float(_nout @ (_Jc @ _MiJn))
                        _meff = (1.0 / _lam) if _lam > 1e-9 else 1.0
                        _vtkd = float(getattr(self, '_vt_kd', -1.0))
                        if _vtkd < 0.0:
                            _vtkd = 2.0 * np.sqrt(_vtkp * _meff)
                        _vn = float(_nout @ (_Jc @ dq_cur))
                        _Fmag = _vtkp * _err + _vtkd * _vn
                        _Fmag = max(_Fmag, 0.0)                  # spring only pushes IN, never pulls
                        _sdf_forces[t] = _Fmag
                        tau_force = tau_force + (_Jc.T @ (_Fmag * (-_nout)))
                        if getattr(self, '_vt_project_tangential', False):
                            _g = _Jc.T @ _nout
                            if len(_locked_arm):                 # NEVER project through locked joints
                                _g[_locked_arm] = 0.0            # (thorax armature 1e10 → huge J col → blows up)
                            _g2 = float(_g @ _g)
                            if _g2 > 1e-12:
                                _pd_term = _pd_term - (float(_pd_term @ _g) / _g2) * _g
                tau = M_arm @ desired_qacc + h - c + _pd_term + tau_force
                _ff_term = M_arm @ ddq_ref + h - c
                _press_term = tau_force

                for i in range(nj):
                    data.ctrl[i] = np.clip(tau[i], ctrl_min[i], ctrl_max[i])

                if not self.stick_static and self.stick_traj is not None \
                        and self._stick_qposadr is not None:
                    sidx = min(t, len(self.stick_traj) - 1)
                    adr = self._stick_qposadr
                    data.qpos[adr:adr + 3] = self.stick_traj[sidx]
                    data.qvel[adr:adr + 6] = 0.0

                mujoco.mj_step(model, data)

            if record_contact_aware and (force_profile is not None or getattr(self, 'contact_aware_emergent', True)):
                mujoco.mj_fullM(model, M_full, data.qM)
                M_arm_r = M_full[_arm_ix]
                h_r = data.qfrc_bias[arm_dof]
                tau_ca = M_arm_r @ ddq_ref + h_r
                if getattr(self, 'contact_aware_emergent', True):
                    tau_ca = tau_ca - np.asarray(data.qfrc_constraint)[arm_dof]
                else:
                    f_meas = force_profile[t] if t < len(force_profile) else force_profile[-1]
                    if stick_body_id >= 0 and f_meas > 0.5:
                        rock_pos = data.site_xpos[site_id].copy()
                        stick_center = data.xpos[stick_body_id].copy()
                        stick_axis = data.xmat[stick_body_id].reshape(3, 3)[:, 2]
                        t_proj = np.clip(np.dot(rock_pos - stick_center, stick_axis),
                                         -0.4, 0.4)
                        nearest = stick_center + t_proj * stick_axis
                        toward_stick = nearest - rock_pos
                        dist = np.linalg.norm(toward_stick)
                        if dist < force_dist_threshold:
                            force_dir = toward_stick / dist if dist > 1e-6 else np.array([0, 0, -1.0])
                            jac_p = np.zeros((3, model.nv))
                            mujoco.mj_jacSite(model, data, jac_p, None, site_id)
                            J_arm = jac_p[:, arm_dof]
                            tau_ca = tau_ca + J_arm.T @ (f_meas * force_dir)
                us_out[t] = tau_ca
            else:
                us_out[t] = data.ctrl[:nj].copy()

            td_ff[t] = _ff_term; td_pd[t] = _pd_term; td_press[t] = _press_term

            if _vdz > 0.0:
                forces_out[t] = _sdf_forces[t]
            elif self.force_sensor_adr >= 0 and site_id >= 0:
                f_site = data.sensordata[self.force_sensor_adr:self.force_sensor_adr + 3]
                R_site = data.site_xmat[site_id].reshape(3, 3)
                f_world = R_site @ f_site
                forces_out[t] = np.linalg.norm(f_world)
            else:
                forces_out[t] = 0.0

            qtrack_out[t] = data.qpos[arm_qpos]
            vtrack_out[t] = data.qvel[arm_dof]

        self._td_ff = td_ff; self._td_pd = td_pd; self._td_press = td_press
        if return_qtrack:
            return us_out, forces_out, qtrack_out, vtrack_out
        return us_out, forces_out

    def _sync_mujoco(self, q, v, u, t=None):
        """Sync MuJoCo state and run forward dynamics — shared by all feature computations."""
        self.mj_data.qpos[:] = self.mujoco_init_q.copy()
        self.mj_data.qvel[:] = 0.0
        arm_qpos = self.controller.arm_qposadr
        arm_dof = self.controller.arm_dofadr
        nj = min(len(q), len(arm_qpos))
        for i in range(nj):
            self.mj_data.qpos[arm_qpos[i]] = q[i]
        nv = min(len(v), len(arm_dof))
        for i in range(nv):
            self.mj_data.qvel[arm_dof[i]] = v[i]
        for qpos_id, dof_id, lock_val in self.locked_joint_constraints:
            self.mj_data.qpos[qpos_id] = lock_val
            self.mj_data.qvel[dof_id] = 0.0
        if not self.stick_static and self.stick_traj is not None \
                and self._stick_qposadr is not None:
            sidx = min(t, len(self.stick_traj) - 1) if t is not None else 0
            adr = self._stick_qposadr
            self.mj_data.qpos[adr:adr + 3] = self.stick_traj[sidx]
            self.mj_data.qvel[adr:adr + 6] = 0.0
        u_ctrl = np.array(u)
        if len(u_ctrl) < self.mj_model.nu:
            u_ctrl = np.concatenate([u_ctrl, np.zeros(self.mj_model.nu - len(u_ctrl))])
        self.mj_data.ctrl[:self.mj_model.nu] = u_ctrl[:self.mj_model.nu]
        mujoco.mj_forward(self.mj_model, self.mj_data)

    def _compute_run_feature(self, key, x, u, t=None):
        """Compute scalar cost for a single running feature, matching Crocoddyl's 0.5*||r||^2 convention.
        Uses MuJoCo arm DOFs for velocity/acceleration features; Pinocchio for JTC and Geo
        (which have no direct MuJoCo equivalent)."""
        q   = x[:self.nq]
        v   = x[self.nq:]
        u_arm = np.array(u)[:self.mj_model.nu]

        self._sync_mujoco(q, v, u_arm, t=t)

        arm = self.controller.arm_dofadr

        # MPPI's "normalized L2" convention — these formulas MUST mirror
        if key == 'Tau':
            return float(np.sqrt(np.sum(u_arm**2))) / 50.0

        elif key == 'JV':
            return float(np.sqrt(np.sum(self.mj_data.qvel[arm]**2)))

        elif key == 'JA':
            if (t is not None and getattr(self, '_ddq_cache', None) is not None
                    and t < len(self._ddq_cache)):
                return float(np.sqrt(np.sum(self._ddq_cache[t] ** 2))) / 100.0
            return float(np.sqrt(np.sum(self.mj_data.qacc[arm] ** 2))) / 100.0

        elif key == 'JTC':
            return 0.0

        elif key == 'Geo':
            model = self.mj_model
            data = self.mj_data
            M_full = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, M_full, data.qM)
            v_full = np.asarray(data.qvel, dtype=float).copy()
            _locked = np.where(np.asarray(model.dof_armature) >= 1e6)[0]
            if len(_locked):
                v_full[_locked] = 0.0
            return float(np.sqrt(abs(v_full @ M_full @ v_full)))

        elif key == 'Eng':
            # in KEYS_RUN, so IRL never reads it via this path.
            return 0.5 * float(np.sum((self.mj_data.qvel[arm] * u_arm)**2))

        elif key.startswith('Eng_'):
            # Per-joint-group power norm — MUST match mppi_cpu.py phi_eng
            from mppi_cpu import ENG_GROUPS
            grp = key[len('Eng_'):]
            gidx = np.array(ENG_GROUPS.get(grp, []), dtype=int)
            if gidx.size == 0:
                return 0.0
            power = self.mj_data.qvel[arm] * u_arm
            return float(np.linalg.norm(power[gidx])) / 20.0

        elif key.startswith('Tau_'):
            # Per-segment torque split (CSQP tausplit). MUST match
            from mppi_cpu import ENG_GROUPS
            grp = key[len('Tau_'):]
            gidx = np.array(ENG_GROUPS.get(grp, []), dtype=int)
            if gidx.size == 0:
                return 0.0
            return float(np.linalg.norm(u_arm[gidx])) / 50.0

        elif key.startswith('JV_'):
            # identifiable posture handle. MUST match mppi_mjx_kinematic phi_jv_g:
            from mppi_cpu import ENG_GROUPS
            grp = key[len('JV_'):]
            gidx = np.array(ENG_GROUPS.get(grp, []), dtype=int)
            if gidx.size == 0:
                return 0.0
            return float(np.linalg.norm(self.mj_data.qvel[arm][gidx]))

        elif key == 'ElbowTrack':
            # explicit elbow-ANGLE tracking (PROOF feature). MUST match mppi_mjx_kinematic
            from mppi_cpu import ELBOW_TRACK_REF, ELBOW_LOCAL
            _qadr = self.controller.arm_qposadr[ELBOW_LOCAL]
            return float(np.sum((self.mj_data.qpos[_qadr] - ELBOW_TRACK_REF) ** 2))

        if key in ('press_force', 'press_capacity'):
            T_ps = getattr(self, 'T_press_start', None)
            if T_ps is not None and t is not None and t < T_ps:
                return 0.0
            if t is not None and self._press_forces is not None:
                force = self._press_forces[t]
            else:
                force = self.controller._get_contact_force(self.mj_data)
            if getattr(self, 'press_emergent', False):
                _fmax = float(self.args.get('force_max', 80.0))
                if key == 'press_capacity':
                    # SQUARED capacity (f_n - Fmax)^2/Fmax^2. MUST be squared, not
                    _f = float(np.clip(force, -2.0 * _fmax, 2.0 * _fmax))
                    return (_f - _fmax) ** 2 / max(_fmax, 1.0) ** 2
                return abs(float(force)) / max(_fmax, 1.0)
            _2c = bool(self.args.get('force_two_cost', False))
            if key == 'press_capacity':
                target = float(self.args.get('force_max', 80.0))
            elif _2c:
                target = 0.0
            elif (t is not None
                  and getattr(self, 'target_force_profile', None) is not None
                  and t < len(self.target_force_profile)):
                target = float(self.target_force_profile[t])
            else:
                target = float(self.target_force)
            # Denominator must MATCH the rollout side (_running_phi). Fixed
            if _2c or key == 'press_capacity':
                _fmax = float(self.args.get('force_max', 80.0))
                _f = float(np.clip(force, -2.0 * _fmax, 2.0 * _fmax))
                return abs(_f - target) / _fmax
            _pref = getattr(self, 'press_ref', 0.0) or 0.0
            _denom = _pref if _pref > 0 else max(abs(target), 1.0)
            return abs(float(force) - target) / _denom

        elif key == 'progress_vel':
            rail_vec = self.p_end_world - self.p_start_world
            rail_len = np.linalg.norm(rail_vec)
            if rail_len > 1e-9 and t is not None and hasattr(self, '_traveled_cache') and self._traveled_cache is not None:
                if t == 0:
                    rail_vel = (self._traveled_cache[1] - self._traveled_cache[0]) / self.dt
                else:
                    rail_vel = (self._traveled_cache[t] - self._traveled_cache[t - 1]) / self.dt
                if getattr(self, 'progress_vel_track', False):
                    _vavg = getattr(self, 'progress_vel_cap', None)
                    if _vavg is None:
                        _vavg = rail_len / max(len(self._traveled_cache) * self.dt, 1e-9)
                    return (rail_vel - float(_vavg)) ** 2
                if getattr(self, 'progress_vel_sq', False):
                    return rail_vel ** 2
                # progress_vel_cap. MUST mirror the MJX/CPU rollout
                pv_cap = getattr(self, 'progress_vel_cap', None)
                if pv_cap is not None:
                    return -min(rail_vel, float(pv_cap))
                return -rail_vel
            return 0.0

        elif key == 'rail_lat':
            ctrl = getattr(self, '_kin_mppi', None) or self.controller
            stick_radius = float(getattr(ctrl, '_stick_radius', 0.02))
            pos = self.controller._get_contact_position(self.mj_data)
            rail_vec = self.p_end_world - self.p_start_world
            rail_len = np.linalg.norm(rail_vec)
            if rail_len > 1e-9:
                rail_unit = rail_vec / rail_len
                diff = pos - self.p_start_world
                lateral = diff - np.dot(diff, rail_unit) * rail_unit
                return float(np.linalg.norm(lateral)) / stick_radius
            return 0.0

        elif key == 'surface':
            ctrl = getattr(self, '_kin_mppi', None) or self.controller
            stick_radius = float(getattr(ctrl, '_stick_radius', 0.02))
            gap = self.controller._rock_stick_gap(self.mj_data)
            return float(max(0.0, gap)) / stick_radius

        elif key == 'rock_ori':
            ctrl = getattr(self, '_kin_mppi', None) or self.controller
            if hasattr(ctrl, '_rock_body_id') and ctrl._rock_body_id is not None \
                    and ctrl._rock_body_id >= 0 and hasattr(ctrl, '_rock_quat_ref') \
                    and ctrl._rock_quat_ref is not None:
                q_cur = self.mj_data.xquat[ctrl._rock_body_id]
                dot = float(np.dot(q_cur, ctrl._rock_quat_ref))
                return 1.0 - dot ** 2
            return 0.0

        elif key == 'brace':
            # normal (LOW = braced). MUST mirror mppi_mjx_kinematic step_fn: J_n_geom = jacp_rock @
            ctrl = getattr(self, '_kin_mppi', None) or self.controller
            _sbid = getattr(ctrl, '_stick_body_id', None)
            _rgid = getattr(ctrl, '_rock_geom_id', None)
            _rbid = getattr(ctrl, '_rock_body_id', None)
            _arm = self.controller.arm_dofadr
            if (_sbid is not None and _sbid >= 0 and _rgid is not None and _rgid >= 0
                    and _rbid is not None and _rbid >= 0):
                _rc = self.mj_data.geom_xpos[_rgid]
                _sc = self.mj_data.xpos[_sbid]
                _sa = self.mj_data.xmat[_sbid].reshape(3, 3)[:, 2]
                _hl = getattr(ctrl, '_stick_half_len', 0.4)
                _tp = np.clip(np.dot(_rc - _sc, _sa), -_hl, _hl)
                _nv = _rc - (_sc + _tp * _sa)
                _n = _nv / max(np.linalg.norm(_nv), 1e-9)
                _J = np.zeros((3, self.mj_model.nv))
                mujoco.mj_jac(self.mj_model, self.mj_data, _J, None, _rc, _rbid)
                return float(np.linalg.norm((_J.T @ _n)[_arm]))
            return 0.0

        elif key == 'approach':
            ctrl = getattr(self, '_kin_mppi', None) or self.controller
            stick_body_id = getattr(ctrl, '_stick_body_id', None)
            rock_geom_id = getattr(ctrl, '_rock_geom_id', None)
            if stick_body_id is not None and stick_body_id >= 0 \
                    and rock_geom_id is not None and rock_geom_id >= 0:
                rock_center = self.mj_data.geom_xpos[rock_geom_id]
                stick_center = self.mj_data.xpos[stick_body_id]
                stick_axis = self.mj_data.xmat[stick_body_id].reshape(3, 3)[:, 2]
                half_len = getattr(ctrl, '_stick_half_len', 0.4)
                stick_radius = getattr(ctrl, '_stick_radius', 0.02)
                t_proj = np.clip(np.dot(rock_center - stick_center, stick_axis),
                                 -half_len, half_len)
                nearest = stick_center + t_proj * stick_axis
                raw_dist = float(np.linalg.norm(rock_center - nearest))
                gap = raw_dist - stick_radius - 0.02
                return abs(gap) / stick_radius
            return 0.0

        elif key == 'traveled':
            rail_vec = self.p_end_world - self.p_start_world
            rail_len = np.linalg.norm(rail_vec)
            if rail_len > 1e-9:
                rail_unit = rail_vec / rail_len
                site_id = getattr(self.controller, 'contact_site_id', None) or \
                          getattr(self.controller, '_contact_site_id', None)
                if site_id is not None and site_id >= 0:
                    pos = self.mj_data.site_xpos[site_id]
                    traveled = float(np.dot(pos - self.p_start_world, rail_unit))
                    tref = getattr(self.controller, 'traveled_ref', None)
                    if getattr(self, 'use_trav_anchor', False) and tref is not None and t is not None:
                        tref = np.asarray(tref, dtype=float)
                        ti = min(int(t), len(tref) - 1)
                        return ((traveled - tref[ti]) / max(rail_len, 1e-9)) ** 2
                    return -traveled / rail_len
            return 0.0

        return 0.0

    def _compute_term_feature(self, key, x):
        """Compute scalar cost for a single terminal feature."""
        q = x[:self.nq]
        v = x[self.nq:]

        if key == 'JV':
            return float(np.sum(v**2))

        elif key == 'Geo':
            diff = pin.difference(self.pin_model, self.q0, q)
            return float(np.sum(diff**2))

        elif key == 'JTC':
            if hasattr(self, '_prev_u'):
                return float(np.sum(self._prev_u**2))
            return 0.0

        elif key == 'JA':
            pin.computeGeneralizedGravity(self.pin_model, self.pin_data, q)
            return float(np.sum(self.pin_data.g**2))

        return 0.0

    @property
    def keys_run(self):
        return sorted(k for k in self.w_run if k not in self.phi_exclude)

    @property
    def keys_term(self):
        return sorted(self.w_term.keys())

    @property
    def nr_run(self):
        return len(self.keys_run)

    @property
    def nr_term(self):
        return len(self.keys_term)

    @property
    def nr(self):
        return self.nr_run + self.nr_term

    def _make_solver_shim(self):
        """
        Minimal object that satisfies IRL.py's interface:
          self.solver = self.model.solver
          self.T      = self.model.solver.problem.T
          self.model.solver.termination_tolerance = ...
          self.model.solver.with_callbacks = False
        """
        T = self.T

        class _ProblemShim:
            def __init__(self, T_):
                self.T = T_

        class _SolverShim:
            def __init__(self, T_):
                self.problem = _ProblemShim(T_)
                self.termination_tolerance = 1e-4
                self.with_callbacks        = False

            def solve(self, xs_init=None, us_init=None, maxiter=1000,
                      isFeasible=False, init_reg=None):

                xs, us = self._parent.solve(xs_init, us_init)
                self.xs = list(xs)
                self.us = list(us)
                return True

        shim = _SolverShim(T)
        shim._parent = self
        return shim
