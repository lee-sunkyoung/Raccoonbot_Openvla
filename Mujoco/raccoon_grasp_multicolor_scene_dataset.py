import os
os.environ["MUJOCO_GL"] = "egl"

import json
import math
import shutil
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image


class DatasetLogger:
    def __init__(self, root_dir="dataset_raw", keep_failed=False):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.keep_failed = keep_failed
        self.episode_dir = None
        self.meta = None

    def start_episode(self, episode_id, instruction, goal_xy, box_init_xy, box_init_yaw,
                      task_type="grasp", target_color=None, target_body_name=None, all_object_init_poses=None):
        episode_name = f"episode_{episode_id:06d}"
        self.episode_dir = self.root_dir / episode_name
        if self.episode_dir.exists():
            shutil.rmtree(self.episode_dir, ignore_errors=True)
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        self.meta = {
            "episode_id": int(episode_id),
            "instruction": str(instruction),
            "task_type": str(task_type),
            "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
            "box_init_xy": [float(box_init_xy[0]), float(box_init_xy[1])],
            "box_init_yaw": float(box_init_yaw),
            "success": False,
            "steps": [],
        }
        if target_color is not None:
            self.meta["target_color"] = str(target_color)
        if target_body_name is not None:
            self.meta["target_body_name"] = str(target_body_name)
        if all_object_init_poses is not None:
            self.meta["all_object_init_poses"] = all_object_init_poses

    def log_step(self, step_idx, image_rgb, joint_angles, gripper_state, object_pose, ee_pose,
                 action, is_first=False, is_last=False):
        image_file = f"frame_{step_idx:06d}.png"
        Image.fromarray(image_rgb).save(self.episode_dir / image_file)
        self.meta["steps"].append({
            "t": int(step_idx),
            "image_file": image_file,
            "joint_angles": [float(x) for x in joint_angles],
            "gripper_state": float(gripper_state),
            "object_pose": [float(x) for x in object_pose],
            "ee_pose": [float(x) for x in ee_pose],
            "action": [float(x) for x in action],
            "is_first": bool(is_first),
            "is_last": bool(is_last),
        })

    def finalize_episode(self, success, exception_text=None, debug_info=None):
        self.meta["success"] = bool(success)
        if exception_text is not None:
            self.meta["exception"] = str(exception_text)
        if debug_info is not None:
            self.meta["debug_info"] = debug_info
        with open(self.episode_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(self.meta, f, indent=2, ensure_ascii=False)
        if (not success) and (not self.keep_failed):
            shutil.rmtree(self.episode_dir, ignore_errors=True)

    def abort_episode(self):
        if self.episode_dir is not None and self.episode_dir.exists():
            shutil.rmtree(self.episode_dir, ignore_errors=True)


class SyncSimRaccoonDataset:
    MAX_SPEEDS = [2.2, 2.3, 2.3, 2.3]
    GRIPPER_SPEED = 15.0
    L1, L2, L3, L4 = 8.25, 10.0, 10.0, 8.0
    MODE_POSITION = 0
    MODE_VELOCITY = 1
    GRIP_OPEN = 0.15701
    GRIP_CLOSE = -0.85
    GRIP_MODE_FREE = 0
    GRIP_MODE_HORZ = 1
    GRIP_MODE_VERT = 2

    CYLINDER_BODY_BY_COLOR = {
        "red": "target_object",
        "blue": "target_object_blue",
        "green": "target_object_green",
        "yellow": "target_object_yellow",
    }
    CYLINDER_COLORS = tuple(CYLINDER_BODY_BY_COLOR.keys())
    BOX_BODY_NAME = "target_box"

    DEFAULT_OBJECT_X_RANGE = (-0.10, 0.10)
    DEFAULT_OBJECT_Y_RANGE = (0.16, 0.20)
    DEFAULT_MIN_OBJECT_DISTANCE = 0.035

    def __init__(self, xml_path, image_size=(256, 256), camera_name=None, use_viewer=False):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"xml 파일을 찾을 수 없습니다: {xml_path}")
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=image_size[1], width=image_size[0])
        self.camera_name = camera_name
        self.use_viewer = use_viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if use_viewer else None

        self.target_angles = [0.0] * 4
        self.current_setpoints = [0.0] * 5
        self.joint_velocities = [0.0] * 4
        self.joint_control_mode = [self.MODE_POSITION] * 4
        self.gripper_target = self.GRIP_OPEN
        self.gripper_mode = self.GRIP_MODE_FREE
        self.active_object_body_name = self.CYLINDER_BODY_BY_COLOR["red"]

        for i in range(4):
            self.joint_velocities[i] = self.MAX_SPEEDS[i] * 0.7

        self.reset_episode(self.make_default_object_specs(), target_color="red")

    def _calc_inv_kinematics(self, x, y, z):
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)) or not isinstance(z, (int, float)):
            return None
        if not ((-28.0 <= x <= 28.0) and (-15.0 <= y <= 28.0) and (0.0 <= z <= 36.25)):
            return None
        x, y = y, -x
        th1 = math.atan2(y, x)
        c1 = math.cos(th1)
        s1 = math.sin(th1)
        x = x - self.L4 * c1
        y = y - self.L4 * s1
        zL1 = z - self.L1
        c3 = (x * x + y * y + zL1 * zL1 - self.L2 * self.L2 - self.L3 * self.L3) / (2 * self.L2 * self.L3)
        if c3 < -1.0 or c3 > 1.0:
            return None
        s3 = -math.sqrt(max(0.0, 1.0 - c3 * c3))
        th3 = math.atan2(s3, c3)
        M1 = c3 * self.L3 + self.L2
        M2 = z - self.L1
        M3 = s3 * self.L3
        M4 = c1 * x + s1 * y
        c2 = M1 * M2 - M3 * M4
        s2 = -M2 * M3 - M1 * M4
        th2 = math.atan2(s2, c2)
        th1 = math.degrees(th1)
        th2 = math.degrees(th2)
        th3 = math.degrees(th3)
        th4 = -(th2 + th3) - 90
        if th1 < -120 or th1 > 120:
            return None
        if th2 < -90 or th2 > 30:
            return None
        if th3 < -150 or th3 > 0:
            return None
        return [th1, th2, th3, th4]

    def degree_to(self, joints, degrees, speed=70):
        j_list = joints if isinstance(joints, (list, tuple)) else [joints]
        d_list = degrees if isinstance(degrees, (list, tuple)) else [degrees]
        if len(d_list) == 1 and len(j_list) > 1:
            d_list = d_list * len(j_list)
        for j, deg in zip(j_list, d_list):
            idx = j - 1
            if 0 <= idx < 4:
                self.joint_control_mode[idx] = self.MODE_POSITION
                self.target_angles[idx] = np.radians(deg)
                self.joint_velocities[idx] = (np.clip(speed, 0.0, 100.0) / 100.0) * self.MAX_SPEEDS[idx]

    def move_to(self, x_cm, y_cm, z_cm, speed=70):
        angles = self._calc_inv_kinematics(x_cm, y_cm, z_cm)
        if angles is None:
            raise ValueError(f"도달할 수 없는 좌표입니다: ({x_cm:.2f}, {y_cm:.2f}, {z_cm:.2f}) cm")
        self.degree_to([1, 2, 3, 4], angles[:4], speed)

    def open_gripper(self):
        self.gripper_target = self.GRIP_OPEN

    def close_gripper(self):
        self.gripper_target = self.GRIP_CLOSE

    def lockh(self):
        self.gripper_mode = self.GRIP_MODE_HORZ

    def lockv(self):
        self.gripper_mode = self.GRIP_MODE_VERT

    def unlock(self):
        if self.gripper_mode != self.GRIP_MODE_FREE:
            self.target_angles[3] = self.data.qpos[3]
            self.gripper_mode = self.GRIP_MODE_FREE

    def execute_action(self, action, speed=70):
        target_x, target_y, target_z, gripper = action
        self.move_to(target_x * 100.0, target_y * 100.0, target_z * 100.0, speed=speed)
        if gripper >= 0.5:
            self.close_gripper()
        else:
            self.open_gripper()

    def _apply_controls_once(self):
        dt = self.model.opt.timestep
        for i in range(4):
            if i == 3 and self.gripper_mode != self.GRIP_MODE_FREE:
                base_angle = -(self.current_setpoints[1] + self.current_setpoints[2])
                desired = base_angle - np.radians(90 if self.gripper_mode == self.GRIP_MODE_HORZ else 180)
                error = desired - self.current_setpoints[i]
                self.current_setpoints[i] += np.clip(error, -self.MAX_SPEEDS[i] * dt, self.MAX_SPEEDS[i] * dt)
            else:
                if self.joint_control_mode[i] == self.MODE_VELOCITY:
                    self.current_setpoints[i] += self.joint_velocities[i] * dt
                else:
                    error = self.target_angles[i] - self.current_setpoints[i]
                    if abs(error) > 1e-4:
                        max_step = abs(self.joint_velocities[i]) * dt
                        self.current_setpoints[i] += np.clip(error, -max_step, max_step)
            joint_id = self.model.actuator_trnid[i, 0]
            rng = self.model.jnt_range[joint_id]
            self.current_setpoints[i] = np.clip(self.current_setpoints[i], rng[0], rng[1])
            self.data.ctrl[i] = self.current_setpoints[i]

        try:
            touch_L = self.data.sensor("sensor_L").data[0]
            touch_R = self.data.sensor("sensor_R").data[0]
            is_touched = (touch_L > 0.1) and (touch_R > 0.1)
        except Exception:
            is_touched = False
        if self.gripper_target == self.GRIP_CLOSE and is_touched:
            self.gripper_target = self.data.qpos[4] - 0.028
        g_err = self.gripper_target - self.current_setpoints[4]
        if abs(g_err) > 1e-4:
            g_step = self.GRIPPER_SPEED * dt
            self.current_setpoints[4] += np.clip(g_err, -g_step, g_step)
        self.data.ctrl[4] = self.current_setpoints[4]

    def step_n(self, n_steps):
        for _ in range(int(n_steps)):
            self._apply_controls_once()
            mujoco.mj_step(self.model, self.data)
            if self.viewer is not None and self.viewer.is_running():
                self.viewer.sync()

    def steps_for_seconds(self, seconds):
        return max(1, int(round(seconds / self.model.opt.timestep)))

    def settle_steps(self, seconds=2.0):
        self.step_n(self.steps_for_seconds(seconds))

    def get_robot_state(self):
        return {"joint_angles": [float(self.data.qpos[i]) for i in range(4)], "gripper_state": float(self.data.qpos[4])}

    def get_object_pose(self, body_name="target_object"):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")
        pos = self.data.xpos[body_id].copy()
        xmat = self.data.xmat[body_id].reshape(3, 3).copy()
        yaw = math.atan2(xmat[1, 0], xmat[0, 0])
        return np.array([pos[0], pos[1], pos[2], yaw], dtype=np.float32)

    def get_contact_summary(self, include_all=False):
        contacts = []
        for contact_idx in range(int(self.data.ncon)):
            contact = self.data.contact[contact_idx]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1, body2 = int(self.model.geom_bodyid[geom1]), int(self.model.geom_bodyid[geom2])
            geom1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom1) or ""
            geom2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom2) or ""
            body1_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1) or ""
            body2_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2) or ""
            if (not include_all) and (body1_name == "" or body2_name == ""):
                continue
            contacts.append({
                "geom1": geom1_name,
                "geom2": geom2_name,
                "body1": body1_name,
                "body2": body2_name,
            })
        return contacts

    def render_rgb(self):
        cam_id = self.camera_name if self.camera_name is not None else -1
        self.renderer.update_scene(self.data, camera=cam_id)
        return self.renderer.render().copy()

    def get_observation(self, object_body_name=None):
        if object_body_name is None:
            object_body_name = self.active_object_body_name
        rs = self.get_robot_state()
        obj = self.get_object_pose(object_body_name)
        img = self.render_rgb()
        link4_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Link4")
        if link4_id != -1:
            ee_pos = self.data.xpos[link4_id].copy()
            ee_pose = [float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])]
        else:
            ee_pose = [0.0, 0.0, 0.0]
        return {"image": img, "joint_angles": rs["joint_angles"], "gripper_state": rs["gripper_state"], "object_pose": obj, "ee_pose": ee_pose}

    def reset_object_pose(self, body_name="target_object", x=0.15, y=0.15, z=0.02, yaw=0.0):
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id == -1:
            raise ValueError(f"body not found: {body_name}")
        qpos_adr = self.model.jnt_qposadr[self.model.body_jntadr[body_id]]
        qw = math.cos(yaw / 2.0)
        qz = math.sin(yaw / 2.0)
        self.data.qpos[qpos_adr:qpos_adr + 7] = np.array([x, y, z, qw, 0.0, 0.0, qz], dtype=np.float64)
        qvel_adr = self.model.jnt_dofadr[self.model.body_jntadr[body_id]]
        self.data.qvel[qvel_adr:qvel_adr + 6] = 0.0

    @classmethod
    def make_default_object_specs(cls):
        x_values = np.linspace(cls.DEFAULT_OBJECT_X_RANGE[0] * 0.75, cls.DEFAULT_OBJECT_X_RANGE[1] * 0.75, len(cls.CYLINDER_COLORS))
        y_center = float(sum(cls.DEFAULT_OBJECT_Y_RANGE) / 2.0)
        return {color: {"body_name": cls.CYLINDER_BODY_BY_COLOR[color], "x": float(x_values[idx]), "y": y_center, "yaw": 0.0} for idx, color in enumerate(cls.CYLINDER_COLORS)}

    @classmethod
    def sample_object_specs(cls, rng, colors=None, x_range=None, y_range=None, yaw_range=(-np.pi / 4, np.pi / 4), min_distance=None, max_tries=1000):
        colors = tuple(colors or cls.CYLINDER_COLORS)
        x_range = x_range or cls.DEFAULT_OBJECT_X_RANGE
        y_range = y_range or cls.DEFAULT_OBJECT_Y_RANGE
        min_distance = cls.DEFAULT_MIN_OBJECT_DISTANCE if min_distance is None else min_distance
        specs = {}
        placed_xy = []
        order = list(colors)
        rng.shuffle(order)
        for color in order:
            for _ in range(max_tries):
                x = float(rng.uniform(x_range[0], x_range[1]))
                y = float(rng.uniform(y_range[0], y_range[1]))
                xy = np.array([x, y], dtype=np.float64)
                if all(np.linalg.norm(xy - other_xy) >= min_distance for other_xy in placed_xy):
                    specs[color] = {"body_name": cls.CYLINDER_BODY_BY_COLOR[color], "x": x, "y": y, "yaw": float(rng.uniform(yaw_range[0], yaw_range[1]))}
                    placed_xy.append(xy)
                    break
            else:
                raise RuntimeError("색상 cylinder를 배치하지 못했습니다.")
        return {color: specs[color] for color in colors}

    @staticmethod
    def specs_to_meta(object_specs):
        return {color: {"body_name": str(spec["body_name"]), "xy": [float(spec["x"]), float(spec["y"])], "yaw": float(spec["yaw"])} for color, spec in object_specs.items()}

    def reset_colored_objects(self, object_specs, target_color):
        if target_color not in object_specs:
            raise ValueError(f"target_color={target_color}가 object_specs에 없습니다.")
        self.active_object_body_name = object_specs[target_color]["body_name"]
        for color, spec in object_specs.items():
            self.reset_object_pose(spec["body_name"], x=spec["x"], y=spec["y"], z=0.02, yaw=spec["yaw"])

    def reset_episode(self, object_specs, target_color="red", active_object_body_name=None, box_pose=None):
        home = np.radians([0.0, -10.0, -140.0, 60.0])
        for i in range(4):
            self.data.qpos[i] = home[i]
            self.data.ctrl[i] = home[i]
            self.current_setpoints[i] = home[i]
            self.target_angles[i] = home[i]
            self.joint_control_mode[i] = self.MODE_POSITION
        self.data.qvel[:] = 0.0
        self.data.qpos[4] = self.GRIP_OPEN
        self.data.ctrl[4] = self.GRIP_OPEN
        self.current_setpoints[4] = self.GRIP_OPEN
        self.gripper_target = self.GRIP_OPEN
        self.gripper_mode = self.GRIP_MODE_FREE
        self.reset_colored_objects(object_specs=object_specs, target_color=target_color)
        if active_object_body_name is not None:
            self.active_object_body_name = active_object_body_name
        if box_pose is not None:
            if len(box_pose) == 4:
                box_body_name, box_x, box_y, box_yaw = box_pose
            elif len(box_pose) == 3:
                box_body_name = self.BOX_BODY_NAME
                box_x, box_y, box_yaw = box_pose
            else:
                raise ValueError(f"box_pose는 길이 3 또는 4여야 합니다. 현재: {len(box_pose)}")
            self.reset_object_pose(box_body_name, x=float(box_x), y=float(box_y), z=0.02, yaw=float(box_yaw))
        mujoco.mj_forward(self.model, self.data)
        self.step_n(20)

    def get_gripper_touch_state(self):
        try:
            return float(self.data.sensor("sensor_L").data[0]), float(self.data.sensor("sensor_R").data[0])
        except Exception:
            return 0.0, 0.0

    def is_grasp_success(self, touch_threshold=0.1, require_closed=True):
        touch_l, touch_r = self.get_gripper_touch_state()
        both_touched = (touch_l > touch_threshold) and (touch_r > touch_threshold)
        if not require_closed:
            return bool(both_touched)
        return bool(both_touched and float(self.data.qpos[4]) < (self.GRIP_OPEN - 0.01))

    def is_body_touching_robot(self, body_name, ignored_geom_names=("floor",)):
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if target_body_id == -1:
            raise ValueError(f"body not found: {body_name}")
        cylinder_body_ids = set()
        for cylinder_body_name in self.CYLINDER_BODY_BY_COLOR.values():
            cylinder_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cylinder_body_name)
            if cylinder_body_id != -1:
                cylinder_body_ids.add(cylinder_body_id)
        ignored_geom_names = set(ignored_geom_names or [])
        for contact_idx in range(int(self.data.ncon)):
            contact = self.data.contact[contact_idx]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            body1, body2 = int(self.model.geom_bodyid[geom1]), int(self.model.geom_bodyid[geom2])
            if target_body_id not in (body1, body2):
                continue
            other_geom = geom2 if body1 == target_body_id else geom1
            other_body = body2 if body1 == target_body_id else body1
            other_geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, other_geom) or ""
            if other_geom_name in ignored_geom_names:
                continue
            if other_body in cylinder_body_ids:
                continue
            return True
        return False

    def is_target_grasp_success(self, target_body_name, touch_threshold=0.1, require_closed=True):
        return bool(self.is_grasp_success(touch_threshold=touch_threshold, require_closed=require_closed) and self.is_body_touching_robot(target_body_name))

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def make_grasp_plan(self, box_x, box_y):
        return [[box_x, box_y, 0.10, 0], [box_x, box_y, 0.02, 0], [box_x, box_y, 0.02, 1]]

    def make_push_plan(self, box_x, box_y, goal_x, goal_y, slide=False):
        dx = goal_x - box_x
        dy = goal_y - box_y
        dist = float(np.hypot(dx, dy))
        if dist < 1e-6:
            push_dir = np.array([1.0, 0.0], dtype=np.float32)
        else:
            push_dir = np.array([dx / dist, dy / dist], dtype=np.float32)
        # Increase engagement and push distances so the box is moved farther.
        approach_x = box_x - float(push_dir[0]) * 0.06
        approach_y = box_y - float(push_dir[1]) * 0.06
        touch_x = box_x + float(push_dir[0]) * 0.03
        touch_y = box_y + float(push_dir[1]) * 0.03
        # Move the touch/goal_touch further along the push direction (proportional to distance)
        goal_touch_offset = max(0.08, dist * 0.5)
        goal_touch_x = box_x + float(push_dir[0]) * goal_touch_offset
        goal_touch_y = box_y + float(push_dir[1]) * goal_touch_offset
        # Final waypoint slightly beyond the nominal goal to impart extra shove
        final_x = goal_x + float(push_dir[0]) * 0.04
        final_y = goal_y + float(push_dir[1]) * 0.04

        if not slide:
            return [
                [approach_x, approach_y, 0.07, 0],
                [touch_x, touch_y, 0.018, 0],
                [goal_touch_x, goal_touch_y, 0.018, 0],
                [final_x, final_y, 0.018, 0],
            ]

        # Grasp-drag plan: come from behind, close once, then keep the gripper closed while moving
        # straight ahead at a constant height. This reduces twisting compared to an open push.
        grasp_z = 0.02
        pre_grasp_z = 0.07
        pre_contact_x = box_x - float(push_dir[0]) * 0.015
        pre_contact_y = box_y - float(push_dir[1]) * 0.015
        contact_x = box_x - float(push_dir[0]) * 0.005
        contact_y = box_y - float(push_dir[1]) * 0.005
        mid1 = box_x + float(push_dir[0]) * min(dist * 0.45, 0.14)
        mid1y = box_y + float(push_dir[1]) * min(dist * 0.45, 0.14)
        mid2 = box_x + float(push_dir[0]) * min(dist * 0.85, 0.28)
        mid2y = box_y + float(push_dir[1]) * min(dist * 0.85, 0.28)
        return [
            [approach_x, approach_y, pre_grasp_z, 0],
            [pre_contact_x, pre_contact_y, pre_grasp_z, 0],
            [contact_x, contact_y, grasp_z, 0],
            [contact_x, contact_y, grasp_z, 1],
            [mid1, mid1y, grasp_z, 1],
            [mid2, mid2y, grasp_z, 1],
            [final_x, final_y, grasp_z, 1],
        ]

    def is_object_at_goal(self, body_name, goal_x, goal_y, tolerance=0.03):
        object_pose = self.get_object_pose(body_name)
        dist = np.linalg.norm(np.array(object_pose[:2], dtype=np.float32) - np.array([goal_x, goal_y], dtype=np.float32))
        return bool(dist < tolerance)


TASK_KEYWORDS = {
    "grasp": ("grasp", "pick up", "grab","catch", "take"),
    "push": ("push","slide", "move"),
}

def detect_task_type_from_instruction(instruction):
    text = instruction.lower().strip()
    push_hits = sum(1 for keyword in TASK_KEYWORDS["push"] if keyword in text)
    grasp_hits = sum(1 for keyword in TASK_KEYWORDS["grasp"] if keyword in text)
    return "push" if push_hits > grasp_hits else "grasp"


def sample_box_pose(rng, object_specs, box_x_range=(-0.12, 0.12), box_y_range=(0.10, 0.18), min_object_distance=0.04, max_tries=1000):
    occupied_xy = [np.array([spec["x"], spec["y"]], dtype=np.float64) for spec in object_specs.values()]
    for _ in range(max_tries):
        box_x = float(rng.uniform(box_x_range[0], box_x_range[1]))
        box_y = float(rng.uniform(box_y_range[0], box_y_range[1]))
        box_xy = np.array([box_x, box_y], dtype=np.float64)
        if any(np.linalg.norm(box_xy - other_xy) < min_object_distance for other_xy in occupied_xy):
            continue
        return {"body_name": SyncSimRaccoonDataset.BOX_BODY_NAME, "x": box_x, "y": box_y, "yaw": float(rng.uniform(-np.pi / 4, np.pi / 4))}
    raise RuntimeError("box pose를 샘플링하지 못했습니다. 범위나 min_object_distance를 확인하세요.")


def sample_goal_xy(rng, target_x, target_y, occupied_xy, goal_x_range=(-0.16, 0.16), goal_y_range=(0.17, 0.28),
                   min_target_goal_distance=0.06, min_object_distance=0.04, max_tries=1000):
    target_xy = np.array([float(target_x), float(target_y)], dtype=np.float64)
    for _ in range(max_tries):
        goal_x = float(np.clip(target_x + rng.uniform(-0.08, 0.08), goal_x_range[0], goal_x_range[1]))
        goal_y = float(np.clip(target_y + rng.uniform(0.05, 0.12), goal_y_range[0], goal_y_range[1]))
        goal_xy = np.array([goal_x, goal_y], dtype=np.float64)
        if np.linalg.norm(goal_xy - target_xy) < min_target_goal_distance:
            continue
        if any(np.linalg.norm(goal_xy - other_xy) < min_object_distance for other_xy in occupied_xy):
            continue
        return [goal_x, goal_y]
    raise RuntimeError("goal 위치를 샘플링하지 못했습니다. 범위나 min_object_distance를 확인하세요.")


def sample_push_scene(rng, object_specs, box_x_range=(-0.12, 0.12), box_y_range=(0.10, 0.18),
                      goal_x_range=(-0.16, 0.16), goal_y_range=(0.17, 0.28),
                      min_object_distance=0.04, max_tries=1000):
    push_box_spec = sample_box_pose(
        rng=rng,
        object_specs=object_specs,
        box_x_range=box_x_range,
        box_y_range=box_y_range,
        min_object_distance=min_object_distance,
        max_tries=max_tries,
    )
    occupied_xy = [np.array([spec["x"], spec["y"]], dtype=np.float64) for spec in object_specs.values()]
    push_goal_xy = sample_goal_xy(
        rng=rng,
        target_x=push_box_spec["x"],
        target_y=push_box_spec["y"],
        occupied_xy=occupied_xy,
        goal_x_range=goal_x_range,
        goal_y_range=goal_y_range,
        min_target_goal_distance=0.02,
        min_object_distance=min_object_distance,
        max_tries=max_tries,
    )
    return push_box_spec, push_goal_xy


def build_instruction(rng, task_type, target_kind="cylinder", target_color=None):
    if target_kind not in {"cylinder", "box"}:
        raise ValueError(f"Unsupported target_kind: {target_kind}")
    if task_type == "grasp":
        verbs = ["grasp", "pick up", "grab", "catch", "take"]
        prefixes = ["please", "can you", "carefully", "gently", ""]
        if target_kind == "box":
            nouns = ["box", "cube", "block", "object", "target box"]
            templates = ["{prefix} {verb} the {noun}", "{prefix} {verb} the {noun} up", "{prefix} {verb} the {noun} for me"]
            return rng.choice(templates).format(prefix=rng.choice(prefixes).strip(), verb=rng.choice(verbs), noun=rng.choice(nouns)).strip()
        nouns = ["cylinder", "object", "item", "target"]
        templates = ["{prefix} {verb} the {color} {noun}", "{prefix} {verb} the {color} {noun} up", "{prefix} {verb} the {color} {noun} for me"]
        return rng.choice(templates).format(prefix=rng.choice(prefixes).strip(), verb=rng.choice(verbs), color=target_color, noun=rng.choice(nouns)).strip()
    if task_type == "push":
        verbs = ["push", "slide", "move"]
        prefixes = ["please", "carefully", "gently", ""]
        goals = ["goal", "target spot", "destination", "marked point", "final position"]
        nouns = ["box", "cube", "block", "object", "target box"] if target_kind == "box" else ["cylinder", "object", "item", "target cylinder"]
        templates = ["{prefix} {verb} the {noun} to the {goal}", "{prefix} {verb} the {noun} toward the {goal}", "{prefix} move the {noun} to the {goal}"]
        return rng.choice(templates).format(prefix=rng.choice(prefixes).strip(), verb=rng.choice(verbs), noun=rng.choice(nouns), goal=rng.choice(goals)).strip()
    raise ValueError(f"Unsupported task_type: {task_type}")


def _balanced_target_counts(num_episodes, keys):
    base = num_episodes // len(keys)
    remainder = num_episodes % len(keys)
    return {key: base + (1 if idx < remainder else 0) for idx, key in enumerate(keys)}


def _sample_remaining_color(rng, target_counts, success_counts):
    remaining_colors = []
    remaining_weights = []
    for color, target_count in target_counts.items():
        remaining = target_count - success_counts[color]
        if remaining > 0:
            remaining_colors.append(color)
            remaining_weights.append(remaining)
    if not remaining_colors:
        return None
    remaining_weights = np.asarray(remaining_weights, dtype=np.float64)
    remaining_weights /= remaining_weights.sum()
    return str(rng.choice(remaining_colors, p=remaining_weights))


def _sample_remaining_combo(rng, target_counts, success_counts):
    remaining_keys = []
    remaining_weights = []
    for key, target_count in target_counts.items():
        remaining = target_count - success_counts[key]
        if remaining > 0:
            remaining_keys.append(key)
            remaining_weights.append(remaining)
    if not remaining_keys:
        return None
    remaining_weights = np.asarray(remaining_weights, dtype=np.float64)
    remaining_weights /= remaining_weights.sum()
    idx = int(rng.choice(len(remaining_keys), p=remaining_weights))
    return remaining_keys[idx]


def run_episode_and_record(rc, logger, episode_id, instruction, object_specs, task_type="grasp", target_kind="cylinder",
                           target_color="red", target_box_pose=None, push_box_spec=None, push_goal_xy=None,
                           speed=70, settle_seconds_per_action=2.0, initial_settle_seconds=0.3, hz=10, touch_threshold=0.1):
    task_type = task_type.lower().strip()
    target_kind = target_kind.lower().strip()
    if target_kind not in {"cylinder", "box"}:
        raise ValueError(f"Unsupported target_kind: {target_kind}")
    if task_type == "grasp":
        if target_kind == "cylinder":
            if target_color not in object_specs:
                raise ValueError(f"target_color={target_color}가 object_specs에 없습니다.")
            target_spec = object_specs[target_color]
            target_body_name = target_spec["body_name"]
            box_init_xy = [float(target_spec["x"]), float(target_spec["y"])]
            box_init_yaw = float(target_spec["yaw"])
            goal_xy = box_init_xy
            plan = rc.make_grasp_plan(box_init_xy[0], box_init_xy[1])
            success_fn = lambda: rc.is_target_grasp_success(target_body_name=target_body_name, touch_threshold=touch_threshold)
        else:
            if target_box_pose is None:
                raise ValueError("box grasp task에는 target_box_pose가 필요합니다.")
            target_body_name = SyncSimRaccoonDataset.BOX_BODY_NAME
            box_init_xy = [float(target_box_pose[0]), float(target_box_pose[1])]
            box_init_yaw = float(target_box_pose[2])
            goal_xy = box_init_xy
            plan = rc.make_grasp_plan(box_init_xy[0], box_init_xy[1])
            success_fn = lambda: rc.is_target_grasp_success(target_body_name=target_body_name, touch_threshold=touch_threshold)
    elif task_type == "push":
        if target_kind == "cylinder":
            raise ValueError("push task는 box target_kind에서만 지원합니다.")
        else:
            if push_box_spec is None or push_goal_xy is None:
                raise ValueError("push task에는 push_box_spec와 push_goal_xy가 필요합니다.")
            target_body_name = push_box_spec["body_name"]
            box_init_xy = [float(push_box_spec["x"]), float(push_box_spec["y"])]
            box_init_yaw = float(push_box_spec["yaw"])
            goal_xy = [float(push_goal_xy[0]), float(push_goal_xy[1])]
            plan = rc.make_push_plan(box_init_xy[0], box_init_xy[1], goal_xy[0], goal_xy[1], slide=True)
            success_fn = lambda: rc.is_object_at_goal(target_body_name, goal_xy[0], goal_xy[1], tolerance=0.08)
    else:
        raise ValueError(f"Unsupported task_type: {task_type}")

    rc.reset_episode(
        object_specs=object_specs,
        target_color=target_color if target_color is not None else list(object_specs.keys())[0],
        active_object_body_name=target_body_name,
        box_pose=(target_body_name, box_init_xy[0], box_init_xy[1], box_init_yaw)
        if target_body_name == SyncSimRaccoonDataset.BOX_BODY_NAME else None,
    )
    rc.lockh()
    if initial_settle_seconds > 0:
        rc.settle_steps(seconds=initial_settle_seconds)

    logger.start_episode(
        episode_id=episode_id,
        instruction=instruction,
        task_type=task_type,
        goal_xy=goal_xy,
        box_init_xy=box_init_xy,
        box_init_yaw=box_init_yaw,
        target_color=target_color if target_kind == "cylinder" else None,
        target_body_name=target_body_name,
        all_object_init_poses=SyncSimRaccoonDataset.specs_to_meta(object_specs),
    )

    try:
        obs = rc.get_observation()
        dt = 1.0 / hz
        step_counter = 0
        for action in plan:
            rc.execute_action(action, speed=speed)
            num_frames = int(settle_seconds_per_action * hz)
            for _ in range(num_frames):
                logger.log_step(
                    step_idx=step_counter,
                    image_rgb=obs["image"],
                    joint_angles=obs["joint_angles"],
                    gripper_state=obs["gripper_state"],
                    object_pose=obs["object_pose"],
                    ee_pose=obs["ee_pose"],
                    action=action,
                    is_first=(step_counter == 0),
                    is_last=False,
                )
                rc.settle_steps(seconds=dt)
                obs = rc.get_observation()
                step_counter += 1
        logger.log_step(
            step_idx=step_counter,
            image_rgb=obs["image"],
            joint_angles=obs["joint_angles"],
            gripper_state=obs["gripper_state"],
            object_pose=obs["object_pose"],
            ee_pose=obs["ee_pose"],
            action=plan[-1],
            is_first=False,
            is_last=True,
        )
        success = success_fn()
        debug_info = None
        if not success:
            debug_info = {
                "contacts": rc.get_contact_summary(include_all=False),
                "target_pose": [float(x) for x in rc.get_object_pose(target_body_name)],
                "current_object_poses": {
                    name: [float(x) for x in rc.get_object_pose(name)]
                    for name in list(object_specs[key]["body_name"] for key in object_specs)
                    if mujoco.mj_name2id(rc.model, mujoco.mjtObj.mjOBJ_BODY, name) != -1
                },
            }
        logger.finalize_episode(success=success, debug_info=debug_info)
        return success
    except Exception:
        logger.abort_episode()
        raise


def collect_dataset(xml_path="Raccoon_colored_cylinder.xml", dataset_root="raccoon_grasp_colored_cylinder", num_episodes=100,
                    colors=("red", "blue", "green", "yellow"), task_types=("grasp", "push"), target_kinds=("cylinder", "box"),
                    keep_failed=False, use_viewer=False, camera_name="front_view", speed=150, settle_seconds_per_action=0.8,
                    initial_settle_seconds=0.3, hz=10, touch_threshold=0.1, seed=None, max_attempts=None,
                    object_x_range=(-0.10, 0.10), object_y_range=(0.16, 0.20), grasp_min_object_distance=0.035,
                    push_min_object_distance=0.04, enforce_push_box_only=True, max_stall_attempts=300):
    colors = tuple(colors)
    task_types = tuple(task_types)
    target_kinds = tuple(target_kinds)

    valid_colors = set(SyncSimRaccoonDataset.CYLINDER_BODY_BY_COLOR.keys())
    unknown_colors = [color for color in colors if color not in valid_colors]
    if unknown_colors:
        raise ValueError(f"지원하지 않는 색상입니다: {unknown_colors}. 지원 색상: {sorted(valid_colors)}")
    if len(colors) == 0 or len(task_types) == 0 or len(target_kinds) == 0:
        raise ValueError("colors/task_types/target_kinds는 비어 있을 수 없습니다.")
    if any(task not in {"grasp", "push"} for task in task_types):
        raise ValueError("지원하지 않는 task_types입니다.")
    if any(kind not in {"cylinder", "box"} for kind in target_kinds):
        raise ValueError("지원하지 않는 target_kinds입니다.")

    combo_keys = [(task, kind) for task in task_types for kind in target_kinds]
    if enforce_push_box_only:
        combo_keys = [combo for combo in combo_keys if not (combo[0] == "push" and combo[1] != "box")]
    if not combo_keys:
        raise ValueError("유효한 task/object 조합이 없습니다. (현재 설정에서는 push는 box에서만 허용)")
    combo_counts = _balanced_target_counts(num_episodes, combo_keys)
    combo_color_target_counts = {
        combo: _balanced_target_counts(combo_count, colors)
        for combo, combo_count in combo_counts.items()
        if combo[1] == "cylinder"
    }
    combo_color_success_counts = {
        combo: {color: 0 for color in colors}
        for combo in combo_color_target_counts
    }

    rng = np.random.default_rng(seed)
    if max_attempts is None:
        max_attempts = max(num_episodes * 20, num_episodes + 100)

    rc = SyncSimRaccoonDataset(xml_path=xml_path, image_size=(256, 256), camera_name=camera_name, use_viewer=use_viewer)
    logger = DatasetLogger(root_dir=dataset_root, keep_failed=keep_failed)
    combo_success_counts = {combo: 0 for combo in combo_keys}
    attempt_count = 0
    last_success_attempt = 0

    print(f"Target task/object counts: {combo_counts}")
    for combo, color_counts in combo_color_target_counts.items():
        print(f"Target color counts for {combo}: {color_counts}")

    def _plan_is_reachable(plan):
        for x_m, y_m, z_m, _ in plan:
            if rc._calc_inv_kinematics(x_m * 100.0, y_m * 100.0, z_m * 100.0) is None:
                return False
        return True

    try:
        while sum(combo_success_counts.values()) < num_episodes and attempt_count < max_attempts:
            attempt_count += 1
            if max_stall_attempts is not None and (attempt_count - last_success_attempt) > max_stall_attempts:
                print(
                    f"[Attempt {attempt_count:04d}] no success for {max_stall_attempts} attempts. "
                    "Stopping early to avoid long stall."
                )
                break
            combo = _sample_remaining_combo(rng, combo_counts, combo_success_counts)
            if combo is None:
                break
            task_type, target_kind = combo
            target_color = None
            target_box_pose = None
            push_goal_xy = None
            push_box_spec = None
            instruction = None
            chosen_plan = None
            object_specs = None

            for scene_try in range(100):
                candidate_object_specs = SyncSimRaccoonDataset.sample_object_specs(
                    rng=rng,
                    colors=colors,
                    x_range=object_x_range,
                    y_range=object_y_range,
                    min_distance=grasp_min_object_distance,
                )

                candidate_target_color = None
                candidate_target_box_pose = None
                candidate_push_goal_xy = None
                candidate_push_box_spec = None

                try:
                    if target_kind == "cylinder":
                        candidate_target_color = _sample_remaining_color(
                            rng,
                            combo_color_target_counts[combo],
                            combo_color_success_counts[combo],
                        )
                        if candidate_target_color is None:
                            break
                        candidate_target_spec = candidate_object_specs[candidate_target_color]
                        if task_type != "grasp":
                            continue
                        candidate_plan = rc.make_grasp_plan(candidate_target_spec["x"], candidate_target_spec["y"])
                    else:
                        if task_type == "grasp":
                            candidate_target_box_pose = sample_box_pose(
                                rng=rng,
                                object_specs=candidate_object_specs,
                                min_object_distance=grasp_min_object_distance,
                            )
                            candidate_plan = rc.make_grasp_plan(
                                candidate_target_box_pose["x"],
                                candidate_target_box_pose["y"],
                            )
                        else:
                            candidate_push_box_spec, candidate_push_goal_xy = sample_push_scene(
                                rng=rng,
                                object_specs=candidate_object_specs,
                                min_object_distance=push_min_object_distance,
                            )
                            candidate_plan = rc.make_push_plan(
                                candidate_push_box_spec["x"],
                                candidate_push_box_spec["y"],
                                candidate_push_goal_xy[0],
                                candidate_push_goal_xy[1],
                                slide=True,
                            )
                except Exception:
                    continue

                if not _plan_is_reachable(candidate_plan):
                    continue

                object_specs = candidate_object_specs
                target_color = candidate_target_color
                target_box_pose = candidate_target_box_pose
                push_goal_xy = candidate_push_goal_xy
                push_box_spec = candidate_push_box_spec
                chosen_plan = candidate_plan
                break

            if object_specs is None or chosen_plan is None:
                print(f"[Attempt {attempt_count:04d}] task_type='{task_type}' | target_kind='{target_kind}' | skipped: no reachable scene found")
                continue

            instruction = build_instruction(rng, task_type, target_kind=target_kind, target_color=target_color)

            episode_id = attempt_count if keep_failed else (sum(combo_success_counts.values()) + 1)
            try:
                success = run_episode_and_record(
                    rc=rc,
                    logger=logger,
                    episode_id=episode_id,
                    instruction=instruction,
                    object_specs=object_specs,
                    task_type=task_type,
                    target_kind=target_kind,
                    target_color=target_color if target_color is not None else colors[0],
                    target_box_pose=(target_box_pose["x"], target_box_pose["y"], target_box_pose["yaw"]) if target_box_pose is not None else None,
                    push_box_spec=push_box_spec,
                    push_goal_xy=push_goal_xy,
                    speed=speed,
                    settle_seconds_per_action=settle_seconds_per_action,
                    initial_settle_seconds=initial_settle_seconds,
                    hz=hz,
                    touch_threshold=touch_threshold,
                )
                if success:
                    combo_success_counts[combo] += 1
                    last_success_attempt = attempt_count
                    if target_kind == "cylinder" and target_color is not None:
                        combo_color_success_counts[combo][target_color] += 1
                print(
                    f"[Attempt {attempt_count:04d}] episode_id={episode_id:06d} | task_type='{task_type}' | target_kind='{target_kind}' | target='{target_color if target_color is not None else 'box'}' | instruction='{instruction}' | success={success} | task_success_counts={combo_success_counts}"
                )
            except Exception as e:
                print(
                    f"[Attempt {attempt_count:04d}] task_type='{task_type}' | target_kind='{target_kind}' | target='{target_color if target_color is not None else 'box'}' | exception: {e}"
                )
    finally:
        rc.close()

    total_success = sum(combo_success_counts.values())
    print(f"완료: success episodes = {total_success}/{num_episodes}, attempts = {attempt_count}")
    print(f"task/object 조합별 성공 episode 수: {combo_success_counts}")
    for combo, color_counts in combo_color_success_counts.items():
        print(f"{combo} 색상별 성공 episode 수: {color_counts}")
    if total_success < num_episodes:
        print(
            "주의: max_attempts에 도달해서 목표 episode 수를 모두 채우지 못했습니다. "
            "max_attempts를 늘리거나 grasp/push 성공 조건과 동작 파라미터를 확인하세요."
        )


if __name__ == "__main__":
    collect_dataset(
        xml_path="Raccoon_colored_cylinder.xml",
        dataset_root="raccoon_grasp_colored_cylinder",
        num_episodes=400,
        colors=("red", "blue", "green", "yellow"),
        task_types=("grasp", "push"),
        target_kinds=("cylinder", "box"),
        keep_failed=False,
        use_viewer=False,
        camera_name="front_view",
        initial_settle_seconds=0.1,
        object_x_range=(-0.10, 0.10),
        object_y_range=(0.16, 0.25),
        grasp_min_object_distance=0.035,
        push_min_object_distance=0.04,
    )
