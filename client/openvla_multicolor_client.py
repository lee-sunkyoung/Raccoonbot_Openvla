import argparse
import base64
import csv
import io
import json
import importlib
import math
import os
import re
import time
from contextlib import nullcontext
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import mujoco
import numpy as np
import requests
from PIL import Image

from raccoon_env import SyncSimRaccoonEnv


CYLINDER_BODY_BY_COLOR = {
    "red": "target_object",
    "blue": "target_object_blue",
    "green": "target_object_green",
    "yellow": "target_object_yellow",
}
CYLINDER_COLORS = tuple(CYLINDER_BODY_BY_COLOR.keys())
BOX_BODY_NAME = "target_box"

# Dataset collection code와 동일한 기본 배치 조건.
# 이전 단일 object range였던 x=(-0.18, 0.18), y=(0.10, 0.18)보다
# x는 좁게, y는 조금 더 앞으로 제한한다.
DEFAULT_OBJECT_X_RANGE = (-0.10, 0.10)
DEFAULT_OBJECT_Y_RANGE = (0.16, 0.25)
DEFAULT_MIN_OBJECT_DISTANCE = 0.035
DEFAULT_YAW_RANGE = (-math.pi / 4, math.pi / 4)
DEFAULT_TASK_TYPE = "grasp"
DEFAULT_TARGET_KIND = "cylinder"
DEFAULT_BOX_XY = (0.00, 0.205)
DEFAULT_BOX_YAW = 0.0
DEFAULT_BOX_YAW_RANGE = (-0.20, 0.20)
EXECUTION_LOG_FIELDNAMES = [
    "step_index",
    "timestamp_perf_counter",
    "execution_time_s",
    "raw_action",
    "executed_action",
    "applied_delta",
    "executed_delta",
    "target_position",
    "actual_move",
    "retries",
    "raw_action_norm",
    "applied_delta_norm",
    "executed_action_norm",
    "action_delta_norm",
    "done",
    "success",
    "use_action_smoothing",
    "smooth_alpha",
    "max_action_abs",
    "clipped_delta",
    "smoothed_delta",
    "raw_delta_norm",
    "clipped_delta_norm",
    "smoothed_delta_norm",
    "smoothing_delta_norm",
    "step_delta_norm",
]
EVAL_LOG_FIELDNAMES = [
    "target_color",
    "target_x",
    "target_y",
    "target_z_initial",
    "ee_x",
    "ee_y",
    "ee_z",
    "target_z_current",
    "xy_error_to_target",
    "gripper_cmd",
    "assist_xy_alignment_before_close",
    "assist_xy_gain",
    "assist_xy_max_step",
    "assist_xy_deadband",
    "assist_xy_total_correction",
    "assist_xy_steps",
    "post_close_lift_controller",
    "post_close_hold_steps",
    "post_close_lift_dz",
    "post_close_lift_steps",
    "post_close_keep_xy",
    "post_close_controller_steps",
    "post_close_forced_lift_steps",
    "post_close_prevented_negative_z_steps",
]
ROLLOUT_LOG_FIELDNAMES = EXECUTION_LOG_FIELDNAMES + EVAL_LOG_FIELDNAMES


def image_to_b64(image_rgb: np.ndarray, image_format: str = "jpeg", jpeg_quality: int = 50) -> Tuple[str, int, str]:
    buffer = io.BytesIO()
    pil_image = Image.fromarray(image_rgb)
    fmt = image_format.lower().strip()
    if fmt in ("jpg", "jpeg"):
        pil_image.save(buffer, format="JPEG", quality=int(np.clip(jpeg_quality, 1, 95)), optimize=True)
        used_format = "jpeg"
    elif fmt == "png":
        pil_image.save(buffer, format="PNG")
        used_format = "png"
    else:
        raise ValueError(f"Unsupported image_format: {image_format}. Use 'jpeg' or 'png'.")

    raw = buffer.getvalue()
    return base64.b64encode(raw).decode("utf-8"), len(raw), used_format


def request_action(
    server_url: str,
    instruction: str,
    image_rgb: np.ndarray,
    unnorm_key: Optional[str],
    image_format: str = "jpeg",
    jpeg_quality: int = 50,
    timeout: float = 60.0,
) -> Tuple[Dict[str, Any], int, str]:
    image_b64, encoded_size_bytes, used_format = image_to_b64(
        image_rgb,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
    )
    payload = {
        "instruction": instruction,
        "image_b64": image_b64,
        "unnorm_key": unnorm_key,
        "do_sample": False,
    }
    response = requests.post(f"{server_url.rstrip('/')}/predict", json=payload, timeout=timeout)
    if not response.ok:
        print(f"[SERVER ERROR] {response.status_code} | {response.text}")
        response.raise_for_status()
    return response.json(), encoded_size_bytes, used_format


def resolve_ssh_password(args: argparse.Namespace) -> Optional[str]:
    if args.ssh_password:
        return args.ssh_password
    env_password = os.environ.get("OPENVLA_SSH_PASSWORD")
    if env_password:
        return env_password
    if args.use_ssh_tunnel and args.ssh_ask_password:
        return getpass("SSH password: ")
    return None


def open_ssh_tunnel(args: argparse.Namespace):
    try:
        sshtunnel = importlib.import_module("sshtunnel")
    except ImportError:
        raise ImportError(
            "sshtunnel is required only when --use_ssh_tunnel is enabled. "
            "Install it with `pip install sshtunnel` or run without --use_ssh_tunnel."
        )

    SSHTunnelForwarder = sshtunnel.SSHTunnelForwarder
    ssh_password = resolve_ssh_password(args)
    tunnel = SSHTunnelForwarder(
        ssh_address_or_host=(args.ssh_host, args.ssh_port),
        ssh_username=args.ssh_user,
        ssh_password=ssh_password,
        remote_bind_address=(args.remote_server_host, args.remote_server_port),
        local_bind_address=(args.local_server_host, args.local_server_port),
    )
    tunnel.start()
    return tunnel


def build_server_url(args: argparse.Namespace, tunnel: Optional[object]) -> str:
    if tunnel is not None:
        return f"http://{args.local_server_host}:{tunnel.local_bind_port}"
    if not args.server_url:
        raise ValueError("--server_url is required when --use_ssh_tunnel is not enabled.")
    return args.server_url


def maybe_tunnel_context(args: argparse.Namespace):
    if args.use_ssh_tunnel:
        return open_ssh_tunnel(args)
    return nullcontext(None)


def print_success_log(step_idx: int, exec_info: Dict[str, Any]) -> None:
    final_delta_xyz = [round(float(v), 4) for v in exec_info["final_delta_xyz"]]
    move_xyz = [round(float(v), 4) for v in exec_info["actual_move_xyz"]]
    target_xyz = [round(float(v), 4) for v in exec_info["target_xyz"]]
    gripper = float(exec_info["gripper_cmd"])
    retries = int(exec_info["retry_count"])
    print(
        f"[{step_idx:03d}] OK | final_delta={final_delta_xyz} | "
        f"move={move_xyz} | target={target_xyz} | "
        f"gripper={gripper:.1f} | retries={retries}"
    )


def print_fail_log(step_idx: int, exc: Exception) -> None:
    print(f"[{step_idx:03d}] FAIL | {exc}")


def get_gripper_cmd_from_action(action: Any, exec_info: Dict[str, Any]) -> float:
    try:
        if len(action) > 6:
            return float(action[6])
    except (TypeError, ValueError):
        pass
    return float(exec_info.get("gripper_cmd", 0.0))


def get_actual_move_xyz(exec_info: Dict[str, Any]) -> np.ndarray:
    actual_move = exec_info.get("actual_move")
    if actual_move is None:
        actual_move = exec_info.get("actual_move_xyz")
    if actual_move is None:
        actual_move = exec_info.get("executed_delta")
    if actual_move is None:
        actual_move = [0.0, 0.0, 0.0]

    actual_move_xyz = np.asarray(actual_move, dtype=np.float64).reshape(-1)
    if actual_move_xyz.size < 3:
        actual_move_xyz = np.pad(actual_move_xyz, (0, 3 - actual_move_xyz.size))
    return actual_move_xyz[:3]


def json_dumps_for_csv(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    return json.dumps(value)


def action_norm(action: Any) -> Optional[float]:
    if action is None:
        return None
    return float(np.linalg.norm(np.asarray(action, dtype=np.float64).reshape(-1)))


def make_rollout_log_row(
    exec_info: Dict[str, Any],
    eval_info: Dict[str, Any],
    raw_action: Any = None,
    executed_action: Any = None,
) -> Dict[str, Any]:
    raw_action_for_log = exec_info.get("raw_action") if raw_action is None else raw_action
    executed_action_for_log = exec_info.get("executed_action", exec_info.get("raw_action")) if executed_action is None else executed_action
    row = {
        "step_index": exec_info.get("step_index"),
        "timestamp_perf_counter": exec_info.get("timestamp_perf_counter"),
        "execution_time_s": exec_info.get("execution_time_s"),
        "raw_action": json_dumps_for_csv(raw_action_for_log),
        "executed_action": json_dumps_for_csv(executed_action_for_log),
        "applied_delta": json_dumps_for_csv(exec_info.get("applied_delta", exec_info.get("applied_delta_xyz"))),
        "executed_delta": json_dumps_for_csv(exec_info.get("executed_delta", exec_info.get("final_delta_xyz"))),
        "target_position": json_dumps_for_csv(exec_info.get("target_position", exec_info.get("target_xyz"))),
        "actual_move": json_dumps_for_csv(exec_info.get("actual_move", exec_info.get("actual_move_xyz"))),
        "retries": exec_info.get("retries", exec_info.get("retry_count")),
        "raw_action_norm": action_norm(raw_action_for_log),
        "applied_delta_norm": exec_info.get("applied_delta_norm"),
        "executed_action_norm": exec_info.get("executed_action_norm"),
        "action_delta_norm": exec_info.get("action_delta_norm"),
        "done": exec_info.get("done"),
        "success": exec_info.get("success"),
        "use_action_smoothing": exec_info.get("use_action_smoothing"),
        "smooth_alpha": exec_info.get("smooth_alpha"),
        "max_action_abs": exec_info.get("max_action_abs"),
        "clipped_delta": json_dumps_for_csv(exec_info.get("clipped_delta")),
        "smoothed_delta": json_dumps_for_csv(exec_info.get("smoothed_delta")),
        "raw_delta_norm": exec_info.get("raw_delta_norm"),
        "clipped_delta_norm": exec_info.get("clipped_delta_norm"),
        "smoothed_delta_norm": exec_info.get("smoothed_delta_norm"),
        "smoothing_delta_norm": exec_info.get("smoothing_delta_norm"),
        "step_delta_norm": exec_info.get("step_delta_norm"),
    }
    row.update(eval_info)
    return row


def default_summary_json_path(log_csv: Optional[str], summary_json: Optional[str]) -> Optional[Path]:
    if summary_json:
        return Path(summary_json)
    if log_csv:
        return Path(log_csv).with_suffix(".summary.json")
    return None


def open_rollout_log_csv(log_csv: Optional[str]) -> Tuple[Optional[Any], Optional[csv.DictWriter]]:
    if log_csv is None:
        return None, None

    path = Path(log_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(path, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(log_file, fieldnames=ROLLOUT_LOG_FIELDNAMES)
    writer.writeheader()
    log_file.flush()
    return log_file, writer


def maybe_save_frame(image_rgb: np.ndarray, out_dir: Path, frame_name: str, step_idx: int, no_save_images: bool, save_every_n_frames: int) -> None:
    if no_save_images:
        return
    if step_idx % save_every_n_frames != 0:
        return
    Image.fromarray(image_rgb).save(out_dir / frame_name)


def infer_color_from_instruction(instruction: Optional[str]) -> Optional[str]:
    """Return the single color word found in an instruction, or None."""
    if not instruction:
        return None

    text = instruction.lower()
    matches = []
    for color in CYLINDER_COLORS:
        if re.search(rf"\b{re.escape(color)}\b", text):
            matches.append(color)

    if len(matches) > 1:
        raise ValueError(f"instruction에 여러 색상이 들어 있습니다: {matches} | instruction={instruction!r}")
    return matches[0] if matches else None


def build_instruction(
    rng: np.random.Generator,
    task_type: str,
    target_kind: str = "cylinder",
    target_color: Optional[str] = None,
) -> str:
    task_type = task_type.lower().strip()
    target_kind = target_kind.lower().strip()
    if target_kind not in {"cylinder", "box"}:
        raise ValueError(f"Unsupported target_kind: {target_kind}")

    if task_type == "grasp":
        verbs = ["grasp", "pick up", "grab", "catch", "take"]
        prefixes = ["please", "can you", "carefully", "gently", ""]
        if target_kind == "box":
            nouns = ["box", "cube", "block", "object", "target box"]
            templates = [
                "{prefix} {verb} the {noun}",
                "{prefix} {verb} the {noun} up",
                "{prefix} {verb} the {noun} for me",
            ]
            return rng.choice(templates).format(
                prefix=rng.choice(prefixes).strip(),
                verb=rng.choice(verbs),
                noun=rng.choice(nouns),
            ).strip()

        nouns = ["cylinder", "object", "item", "target"]
        templates = [
            "{prefix} {verb} the {color} {noun}",
            "{prefix} {verb} the {color} {noun} up",
            "{prefix} {verb} the {color} {noun} for me",
        ]
        return rng.choice(templates).format(
            prefix=rng.choice(prefixes).strip(),
            verb=rng.choice(verbs),
            color=target_color,
            noun=rng.choice(nouns),
        ).strip()

    if task_type == "push":
        verbs = ["push", "slide", "move"]
        prefixes = ["please", "carefully", "gently", ""]
        goals = ["goal", "target spot", "destination", "marked point", "final position"]
        nouns = ["box", "cube", "block", "object", "target box"] if target_kind == "box" else ["cylinder", "object", "item", "target cylinder"]
        templates = [
            "{prefix} {verb} the {noun} to the {goal}",
            "{prefix} {verb} the {noun} toward the {goal}",
            "{prefix} move the {noun} to the {goal}",
        ]
        return rng.choice(templates).format(
            prefix=rng.choice(prefixes).strip(),
            verb=rng.choice(verbs),
            noun=rng.choice(nouns),
            goal=rng.choice(goals),
        ).strip()

    raise ValueError(f"Unsupported task_type: {task_type}")


def resolve_target_color_and_instruction(
    instruction: Optional[str],
    target_color_arg: str,
    rng: np.random.Generator,
    task_type: str,
    target_kind: str,
) -> Tuple[str, str]:
    """
    Keep the OpenVLA prompt and the physical target color synchronized.

    Priority:
      1. If instruction already contains exactly one color, use that color.
      2. Else if --target_color is one of red/blue/green/yellow, use it.
            3. Else choose a random color and generate a dataset-style instruction.
    """
    instruction_color = infer_color_from_instruction(instruction)

    if instruction_color is not None:
        target_color = instruction_color
        if target_color_arg in CYLINDER_COLORS and target_color_arg != instruction_color:
            raise ValueError(
                f"--instruction 색상({instruction_color})과 --target_color({target_color_arg})가 다릅니다. "
                "OpenVLA prompt와 실제 target이 어긋나지 않도록 둘 중 하나를 수정하세요."
            )
    elif target_color_arg in CYLINDER_COLORS:
        target_color = target_color_arg
    elif target_color_arg in ("auto", "random"):
        target_color = str(rng.choice(CYLINDER_COLORS))
    else:
        raise ValueError(f"지원하지 않는 --target_color 값입니다: {target_color_arg}")

    if instruction is None or instruction.strip() == "":
        instruction = build_instruction(
            rng=rng,
            task_type=task_type,
            target_kind=target_kind,
            target_color=target_color,
        )

    return target_color, instruction


def make_default_object_specs() -> Dict[str, Dict[str, float]]:
    """Deterministic fallback used when randomization is disabled."""
    x_values = np.linspace(
        DEFAULT_OBJECT_X_RANGE[0] * 0.75,
        DEFAULT_OBJECT_X_RANGE[1] * 0.75,
        len(CYLINDER_COLORS),
    )
    y_center = float(sum(DEFAULT_OBJECT_Y_RANGE) / 2.0)
    return {
        color: {
            "body_name": CYLINDER_BODY_BY_COLOR[color],
            "x": float(x_values[idx]),
            "y": y_center,
            "yaw": 0.0,
        }
        for idx, color in enumerate(CYLINDER_COLORS)
    }


def make_default_box_spec(xy: Tuple[float, float] = DEFAULT_BOX_XY, yaw: float = DEFAULT_BOX_YAW) -> Dict[str, float]:
    return {
        "body_name": BOX_BODY_NAME,
        "x": float(xy[0]),
        "y": float(xy[1]),
        "yaw": float(yaw),
    }


def sample_box_spec(
    rng: np.random.Generator,
    x_range: Tuple[float, float] = DEFAULT_OBJECT_X_RANGE,
    y_range: Tuple[float, float] = DEFAULT_OBJECT_Y_RANGE,
    yaw_range: Tuple[float, float] = DEFAULT_BOX_YAW_RANGE,
) -> Dict[str, float]:
    if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
        raise ValueError(f"잘못된 box spawn range입니다: x_range={x_range}, y_range={y_range}")
    return {
        "body_name": BOX_BODY_NAME,
        "x": float(rng.uniform(x_range[0], x_range[1])),
        "y": float(rng.uniform(y_range[0], y_range[1])),
        "yaw": float(rng.uniform(yaw_range[0], yaw_range[1])),
    }


def sample_object_specs(
    rng: np.random.Generator,
    x_range: Tuple[float, float] = DEFAULT_OBJECT_X_RANGE,
    y_range: Tuple[float, float] = DEFAULT_OBJECT_Y_RANGE,
    yaw_range: Tuple[float, float] = DEFAULT_YAW_RANGE,
    min_distance: float = DEFAULT_MIN_OBJECT_DISTANCE,
    max_tries: int = 1000,
) -> Dict[str, Dict[str, float]]:
    """
    Dataset collection code와 동일한 조건으로 4개 색상 cylinder를 모두 배치한다.

    Defaults:
      - x_range=(-0.10, 0.10)
      - y_range=(0.16, 0.20)
      - min_object_distance=0.035
      - yaw_range=(-pi/4, pi/4)
    """
    if x_range[0] >= x_range[1] or y_range[0] >= y_range[1]:
        raise ValueError(f"잘못된 spawn range입니다: x_range={x_range}, y_range={y_range}")

    specs: Dict[str, Dict[str, float]] = {}
    placed_xy = []

    # 특정 색상이 항상 먼저 배치되어 유리/불리해지는 bias를 줄인다.
    placement_order = list(CYLINDER_COLORS)
    rng.shuffle(placement_order)

    for color in placement_order:
        for _ in range(max_tries):
            x = float(rng.uniform(x_range[0], x_range[1]))
            y = float(rng.uniform(y_range[0], y_range[1]))
            xy = np.array([x, y], dtype=np.float64)

            if all(np.linalg.norm(xy - other_xy) >= min_distance for other_xy in placed_xy):
                specs[color] = {
                    "body_name": CYLINDER_BODY_BY_COLOR[color],
                    "x": x,
                    "y": y,
                    "yaw": float(rng.uniform(yaw_range[0], yaw_range[1])),
                }
                placed_xy.append(xy)
                break
        else:
            raise RuntimeError(
                "색상 cylinder 4개를 겹치지 않게 배치하지 못했습니다. "
                f"x_range={x_range}, y_range={y_range}, min_distance={min_distance}를 확인하세요."
            )

    return {color: specs[color] for color in CYLINDER_COLORS}


def reset_freejoint_body_pose(env: SyncSimRaccoonEnv, body_name: str, x: float, y: float, z: float, yaw: float) -> None:
    """Set a MuJoCo freejoint body pose directly through env.model/env.data."""
    if not hasattr(env, "model") or not hasattr(env, "data"):
        raise AttributeError("SyncSimRaccoonEnv에 model/data 속성이 필요합니다.")

    body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id == -1:
        raise ValueError(f"body not found: {body_name}. XML이 Raccoon_colored_cylinder.xml인지 확인하세요.")

    jnt_adr = int(env.model.body_jntadr[body_id])
    jnt_num = int(env.model.body_jntnum[body_id])
    if jnt_num < 1:
        raise ValueError(f"{body_name} has no joint")

    joint_id = jnt_adr
    qpos_adr = int(env.model.jnt_qposadr[joint_id])

    # freejoint qpos = [x, y, z, qw, qx, qy, qz]
    qw = math.cos(yaw / 2.0)
    qz = math.sin(yaw / 2.0)
    env.data.qpos[qpos_adr:qpos_adr + 7] = np.array([x, y, z, qw, 0.0, 0.0, qz], dtype=np.float64)

    qvel_adr = int(env.model.jnt_dofadr[joint_id])
    env.data.qvel[qvel_adr:qvel_adr + 6] = 0.0


def reset_box_pose(env: SyncSimRaccoonEnv, box_spec: Dict[str, float]) -> None:
    reset_freejoint_body_pose(
        env=env,
        body_name=str(box_spec["body_name"]),
        x=float(box_spec["x"]),
        y=float(box_spec["y"]),
        z=0.02,
        yaw=float(box_spec["yaw"]),
    )


def reset_multicolor_scene(
    env: SyncSimRaccoonEnv,
    object_specs: Dict[str, Dict[str, float]],
    target_color: str,
    box_spec: Optional[Dict[str, float]] = None,
) -> None:
    """
    Reset the robot using the existing env.reset_episode(), then place all four
    colored cylinders in the scene. The prompted color is stored as env.active_object_body_name
    when the env supports that attribute, but inference only needs the rendered image.
    """
    if target_color not in object_specs:
        raise ValueError(f"target_color={target_color}가 object_specs에 없습니다.")

    target_spec = object_specs[target_color]

    # Existing raccoon_env expects a single target pose for reset_episode().
    # We use the prompted target pose to reset the robot/home state, then override
    # all four cylinder poses below.
    env.reset_episode(float(target_spec["x"]), float(target_spec["y"]), float(target_spec["yaw"]))

    for color, spec in object_specs.items():
        reset_freejoint_body_pose(
            env=env,
            body_name=str(spec["body_name"]),
            x=float(spec["x"]),
            y=float(spec["y"]),
            z=0.02,
            yaw=float(spec["yaw"]),
        )

    if box_spec is None:
        box_spec = make_default_box_spec()
    reset_box_pose(env, box_spec)

    target_body_name = str(target_spec["body_name"])
    if hasattr(env, "active_object_body_name"):
        env.active_object_body_name = target_body_name
    if hasattr(env, "target_body_name"):
        env.target_body_name = target_body_name
    if hasattr(env, "target_box_body_name"):
        env.target_box_body_name = BOX_BODY_NAME

    mujoco.mj_forward(env.model, env.data)


def object_specs_to_meta(object_specs: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, Any]]:
    return {
        color: {
            "body_name": str(spec["body_name"]),
            "xy": [float(spec["x"]), float(spec["y"])],
            "yaw": float(spec["yaw"]),
        }
        for color, spec in object_specs.items()
    }


def box_spec_to_meta(box_spec: Dict[str, float]) -> Dict[str, Any]:
    return {
        "body_name": str(box_spec["body_name"]),
        "xy": [float(box_spec["x"]), float(box_spec["y"])],
        "yaw": float(box_spec["yaw"]),
    }


def write_rollout_meta(
    out_dir: Path,
    instruction: str,
    target_color: str,
    object_specs: Dict[str, Dict[str, float]],
    box_spec: Dict[str, float],
    args: Dict[str, Any],
) -> None:
    meta = {
        "instruction": instruction,
        "target_color": target_color,
        "target_body_name": CYLINDER_BODY_BY_COLOR[target_color],
        "all_object_init_poses": object_specs_to_meta(object_specs),
        "box_init_pose": box_spec_to_meta(box_spec),
        "args": args,
    }
    with open(out_dir / "rollout_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)


def rollout(
    xml_path: str,
    server_url: str,
    instruction: Optional[str],
    unnorm_key: str,
    output_dir: str,
    episode_id: int = 1,
    max_steps: int = 1000000,
    use_viewer: bool = True,
    camera_name: str = "front_view",
    log_csv=None,
    use_action_smoothing=False,
    smooth_alpha=0.6,
    max_action_abs=None,
    speed: int = 90,
    settle_seconds_per_action: float = 0.1,
    initial_settle_seconds: float = 0.0,
    delta_scale: float = 1.0,
    randomize_objects: bool = True,
    randomize_box: bool = True,
    request_timeout: float = 60.0,
    image_format: str = "jpeg",
    jpeg_quality: int = 50,
    max_delta_xyz: float = 0.01,
    target_color_arg: str = "auto",
    task_type: str = DEFAULT_TASK_TYPE,
    target_kind: str = DEFAULT_TARGET_KIND,
    seed: Optional[int] = None,
    object_x_range: Tuple[float, float] = DEFAULT_OBJECT_X_RANGE,
    object_y_range: Tuple[float, float] = DEFAULT_OBJECT_Y_RANGE,
    min_object_distance: float = DEFAULT_MIN_OBJECT_DISTANCE,
    stop_after_gripper_close: bool = False,
    gripper_close_threshold: float = 0.5,
    no_motion_threshold: float = 0.003,
    no_motion_patience: int = 5,
    summary_json: Optional[str] = None,
    no_save_images: bool = False,
    save_every_n_frames: int = 1,
    stop_on_lift_success: bool = False,
    lift_success_threshold: float = 0.010,
    min_steps_after_close: int = 10,
    max_valid_lift_delta: float = 0.08,
    max_final_xy_error_for_success: float = 0.05,
    gate_close_by_xy: bool = False,
    close_xy_threshold: float = 0.015,
    preclose_min_z_when_xy_bad: Optional[float] = None,
    latch_gripper_after_close: bool = False,
    post_close_clamp_negative_z: bool = False,
    assist_xy_alignment_before_close: bool = False,
    assist_xy_gain: float = 0.5,
    assist_xy_max_step: float = 0.004,
    assist_xy_deadband: float = 0.002,
    post_close_lift_controller: bool = False,
    post_close_hold_steps: int = 3,
    post_close_lift_dz: float = 0.0025,
    post_close_lift_steps: int = 12,
    post_close_keep_xy: bool = False,
    box_xy: Tuple[float, float] = DEFAULT_BOX_XY,
    box_yaw: float = DEFAULT_BOX_YAW,
) -> None:
    if save_every_n_frames < 1:
        raise ValueError(f"save_every_n_frames must be >= 1, got {save_every_n_frames}")
    if min_steps_after_close < 0:
        raise ValueError(f"min_steps_after_close must be >= 0, got {min_steps_after_close}")
    if max_valid_lift_delta <= 0:
        raise ValueError(f"max_valid_lift_delta must be positive, got {max_valid_lift_delta}")
    if max_final_xy_error_for_success <= 0:
        raise ValueError(
            f"max_final_xy_error_for_success must be positive, got {max_final_xy_error_for_success}"
        )
    if close_xy_threshold <= 0:
        raise ValueError(f"close_xy_threshold must be positive, got {close_xy_threshold}")
    if assist_xy_gain < 0:
        raise ValueError(f"assist_xy_gain must be non-negative, got {assist_xy_gain}")
    if assist_xy_max_step < 0:
        raise ValueError(f"assist_xy_max_step must be non-negative, got {assist_xy_max_step}")
    if assist_xy_deadband < 0:
        raise ValueError(f"assist_xy_deadband must be non-negative, got {assist_xy_deadband}")
    if post_close_hold_steps < 0:
        raise ValueError(f"post_close_hold_steps must be >= 0, got {post_close_hold_steps}")
    if post_close_lift_dz < 0:
        raise ValueError(f"post_close_lift_dz must be non-negative, got {post_close_lift_dz}")
    if post_close_lift_steps < 0:
        raise ValueError(f"post_close_lift_steps must be >= 0, got {post_close_lift_steps}")

    out_dir = Path(output_dir) / f"episode_{episode_id:06d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not no_save_images:
        # 기존 이미지 삭제 후 새로 저장 시작
        clear_existing_images(out_dir)

    rng = np.random.default_rng(seed)
    target_color, instruction = resolve_target_color_and_instruction(
        instruction=instruction,
        target_color_arg=target_color_arg,
        rng=rng,
        task_type=task_type,
        target_kind=target_kind,
    )

    if randomize_objects:
        object_specs = sample_object_specs(
            rng=rng,
            x_range=object_x_range,
            y_range=object_y_range,
            min_distance=min_object_distance,
        )
    else:
        object_specs = make_default_object_specs()

    if randomize_box:
        box_spec = sample_box_spec(
            rng=rng,
            x_range=object_x_range,
            y_range=object_y_range,
            yaw_range=(box_yaw - 0.20, box_yaw + 0.20),
        )
    else:
        box_spec = make_default_box_spec(xy=box_xy, yaw=box_yaw)

    summary_json_path = default_summary_json_path(log_csv, summary_json)
    log_csv_file, log_csv_writer = open_rollout_log_csv(log_csv)
    target_body_name = str(CYLINDER_BODY_BY_COLOR[target_color])
    target_x = None
    target_y = None
    target_z_initial = None
    first_close_step = None
    xy_error_at_close = None
    target_z_at_close = None
    final_xy_error = None
    target_z_final = None
    total_xy_move = 0.0
    post_close_positive_z_steps = 0
    retry_sum = 0
    num_logged_steps = 0
    previous_target_z = None
    stop_reason = "error"
    invalid_success_reason = ""
    assist_xy_total_correction = 0.0
    assist_xy_steps = 0
    close_ee_xy = None
    post_close_controller_steps = 0
    post_close_forced_lift_steps = 0
    post_close_prevented_negative_z_steps = 0

    env = SyncSimRaccoonEnv(
        xml_path=xml_path,
        image_size=(256, 256),
        camera_name=camera_name,
        use_viewer=use_viewer,
    )

    try:
        reset_multicolor_scene(
            env=env,
            object_specs=object_specs,
            target_color=target_color,
            box_spec=box_spec,
        )

        env.lockh()
        env.debug_check_current_ee_reachable()

        # Dataset collector와 동일하게 첫 observation 전에 free-joint cylinder를 안정화한다.
        if initial_settle_seconds > 0:
            env.settle_steps(seconds=initial_settle_seconds)

        target_pose_initial = env.get_object_pose(target_body_name)
        target_x = float(target_pose_initial[0])
        target_y = float(target_pose_initial[1])
        target_z_initial = float(target_pose_initial[2])
        previous_target_z = target_z_initial

        write_rollout_meta(
            out_dir=out_dir,
            instruction=instruction,
            target_color=target_color,
            object_specs=object_specs,
            box_spec=box_spec,
            args={
                "xml_path": xml_path,
                "unnorm_key": unnorm_key,
                "camera_name": camera_name,
                "speed": speed,
                "settle_seconds_per_action": settle_seconds_per_action,
                "initial_settle_seconds": initial_settle_seconds,
                "image_format": image_format,
                "jpeg_quality": jpeg_quality,
                "delta_scale": delta_scale,
                "max_delta_xyz": max_delta_xyz,
                "seed": seed,
                "object_x_range": list(object_x_range),
                "object_y_range": list(object_y_range),
                "min_object_distance": min_object_distance,
                "box_xy": list(box_xy),
                "box_yaw": box_yaw,
                "randomize_box": randomize_box,
            },
        )

        print(
            f"[SCENE] instruction={instruction!r} | target_color={target_color!r} | "
            f"target_xy=({object_specs[target_color]['x']:.3f}, {object_specs[target_color]['y']:.3f}) | "
            f"objects={object_specs_to_meta(object_specs)}"
        )

        obs = env.get_observation()
        step_idx = 0
        stuck_after_grasp_count = 0

        while True:
            infer_start = time.perf_counter()
            response, image_payload_bytes, used_format = request_action(
                server_url=server_url,
                instruction=instruction,
                image_rgb=obs["image"],
                unnorm_key=unnorm_key,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
                timeout=request_timeout,
            )
            inference_time_s = time.perf_counter() - infer_start
            raw_action = [float(v) for v in response["action"]]
            executed_action = list(raw_action)

            try:
                ee_x_before, ee_y_before, ee_z_before = [float(v) for v in env.get_ee_pose()]
                xy_error_before = float(np.linalg.norm([ee_x_before - target_x, ee_y_before - target_y]))

                if len(executed_action) >= 7:
                    if first_close_step is None and assist_xy_alignment_before_close:
                        error_x = target_x - ee_x_before
                        error_y = target_y - ee_y_before
                        if xy_error_before > assist_xy_deadband:
                            assist_dx = float(
                                np.clip(
                                    assist_xy_gain * error_x,
                                    -assist_xy_max_step,
                                    assist_xy_max_step,
                                )
                            )
                            assist_dy = float(
                                np.clip(
                                    assist_xy_gain * error_y,
                                    -assist_xy_max_step,
                                    assist_xy_max_step,
                                )
                            )
                            executed_action[0] += assist_dx
                            executed_action[1] += assist_dy
                            assist_xy_total_correction += float(np.linalg.norm([assist_dx, assist_dy]))
                            assist_xy_steps += 1

                    if first_close_step is None and xy_error_before > close_xy_threshold:
                        if gate_close_by_xy and executed_action[6] > 0.5:
                            executed_action[6] = 0.0

                        if preclose_min_z_when_xy_bad is not None and executed_action[2] < 0.0:
                            if delta_scale > 0.0:
                                min_allowed_dz = min(0.0, (preclose_min_z_when_xy_bad - ee_z_before) / delta_scale)
                            else:
                                min_allowed_dz = 0.0
                            executed_action[2] = max(executed_action[2], min_allowed_dz)

                    if first_close_step is not None:
                        if latch_gripper_after_close:
                            executed_action[6] = max(executed_action[6], 1.0)
                        if post_close_clamp_negative_z and executed_action[2] < 0.0:
                            executed_action[2] = 0.0
                        if post_close_lift_controller:
                            post_close_controller_steps += 1
                            steps_after_close = step_idx - first_close_step
                            controller_dz_before = float(executed_action[2])

                            executed_action[6] = max(executed_action[6], 1.0)

                            in_lift_window = (
                                steps_after_close > post_close_hold_steps
                                and steps_after_close <= post_close_hold_steps + post_close_lift_steps
                            )
                            if in_lift_window and executed_action[2] < post_close_lift_dz:
                                executed_action[2] = post_close_lift_dz
                                post_close_forced_lift_steps += 1
                            elif executed_action[2] < 0.0:
                                executed_action[2] = 0.0

                            if controller_dz_before < 0.0:
                                post_close_prevented_negative_z_steps += 1

                            if post_close_keep_xy:
                                executed_action[0] = 0.0
                                executed_action[1] = 0.0

                exec_info = env.execute_delta_action7(
                    action=executed_action,
                    speed=speed,
                    delta_scale=delta_scale,
                    max_delta_xyz=max_delta_xyz,
                )
                exec_info["timestamp_perf_counter"] = float(time.perf_counter())
                exec_info["execution_time_s"] = float(inference_time_s)
                exec_info["executed_action"] = list(executed_action)

                print(
                    f"[TIMING] Step {step_idx:03d} | infer={inference_time_s:.3f}s | "
                    f"image={used_format.upper()} {image_payload_bytes / 1024.0:.1f}KB"
                )
                print_success_log(step_idx, exec_info)

                gripper_cmd = get_gripper_cmd_from_action(executed_action, exec_info)
                actual_move = get_actual_move_xyz(exec_info)
                move_norm = float(np.linalg.norm(actual_move[:3]))
                ee_x, ee_y, ee_z = [float(v) for v in env.get_ee_pose()]
                target_pose_current = env.get_object_pose(target_body_name)
                target_z_current = float(target_pose_current[2])
                xy_error_to_target = float(np.linalg.norm([ee_x - target_x, ee_y - target_y]))
                object_lift_delta_current = float(target_z_current - target_z_initial)
                total_xy_move += float(np.linalg.norm(actual_move[:2]))
                retry_sum += int(exec_info.get("retries", exec_info.get("retry_count", 0)) or 0)
                num_logged_steps += 1

                closing_now = gripper_cmd > 0.5
                if (first_close_step is not None or closing_now) and previous_target_z is not None:
                    if target_z_current > previous_target_z:
                        post_close_positive_z_steps += 1

                if first_close_step is None and closing_now:
                    first_close_step = step_idx
                    xy_error_at_close = xy_error_to_target
                    target_z_at_close = target_z_current
                    close_ee_xy = [ee_x, ee_y]

                previous_target_z = target_z_current
                final_xy_error = xy_error_to_target
                target_z_final = target_z_current

                if log_csv_writer is not None and log_csv_file is not None:
                    log_csv_writer.writerow(
                        make_rollout_log_row(
                            exec_info,
                            {
                                "target_color": target_color,
                                "target_x": target_x,
                                "target_y": target_y,
                                "target_z_initial": target_z_initial,
                                "ee_x": ee_x,
                                "ee_y": ee_y,
                                "ee_z": ee_z,
                                "target_z_current": target_z_current,
                                "xy_error_to_target": xy_error_to_target,
                                "gripper_cmd": gripper_cmd,
                                "assist_xy_alignment_before_close": assist_xy_alignment_before_close,
                                "assist_xy_gain": assist_xy_gain,
                                "assist_xy_max_step": assist_xy_max_step,
                                "assist_xy_deadband": assist_xy_deadband,
                                "assist_xy_total_correction": assist_xy_total_correction,
                                "assist_xy_steps": assist_xy_steps,
                                "post_close_lift_controller": post_close_lift_controller,
                                "post_close_hold_steps": post_close_hold_steps,
                                "post_close_lift_dz": post_close_lift_dz,
                                "post_close_lift_steps": post_close_lift_steps,
                                "post_close_keep_xy": post_close_keep_xy,
                                "post_close_controller_steps": post_close_controller_steps,
                                "post_close_forced_lift_steps": post_close_forced_lift_steps,
                                "post_close_prevented_negative_z_steps": post_close_prevented_negative_z_steps,
                            },
                            raw_action=raw_action,
                            executed_action=executed_action,
                        )
                    )
                    log_csv_file.flush()

                if stop_on_lift_success and first_close_step is not None:
                    steps_after_close = step_idx - first_close_step
                    if steps_after_close >= min_steps_after_close:
                        if object_lift_delta_current > max_valid_lift_delta:
                            invalid_success_reason = "object_lift_delta_too_large"
                            stop_reason = invalid_success_reason
                            print(
                                "[STOP] invalid lift delta "
                                f"(object_lift_delta={object_lift_delta_current:.6f}, "
                                f"max_valid={max_valid_lift_delta})"
                            )
                            break
                        if object_lift_delta_current >= lift_success_threshold:
                            if xy_error_to_target <= max_final_xy_error_for_success:
                                stop_reason = "lift_success"
                                print(
                                    "[STOP] lift success "
                                    f"(object_lift_delta={object_lift_delta_current:.6f}, "
                                    f"threshold={lift_success_threshold})"
                                )
                            else:
                                invalid_success_reason = "final_xy_error_too_large"
                                stop_reason = invalid_success_reason
                                print(
                                    "[STOP] invalid final xy error "
                                    f"(final_xy_error={xy_error_to_target:.6f}, "
                                    f"max_valid={max_final_xy_error_for_success})"
                                )
                            break

                if stop_after_gripper_close:
                    if gripper_cmd > gripper_close_threshold and move_norm < no_motion_threshold:
                        stuck_after_grasp_count += 1
                    else:
                        stuck_after_grasp_count = 0

                    if stuck_after_grasp_count >= no_motion_patience:
                        print(
                            "[STOP] gripper closed and movement is small for "
                            f"{stuck_after_grasp_count} consecutive steps "
                            f"(move_norm={move_norm:.6f}, threshold={no_motion_threshold})"
                        )
                        stop_reason = "gripper_closed_no_motion"
                        break

                env.settle_steps(seconds=settle_seconds_per_action)
                obs = env.get_observation()

                frame_name = f"frame_{step_idx:06d}.png"
                maybe_save_frame(obs["image"], out_dir, frame_name, step_idx, no_save_images, save_every_n_frames)

            except Exception as exc:
                print_fail_log(step_idx, exc)
                obs = env.get_observation()

                frame_name = f"frame_{step_idx:06d}_skipped.png"
                maybe_save_frame(obs["image"], out_dir, frame_name, step_idx, no_save_images, save_every_n_frames)

                step_idx += 1
                if step_idx >= max_steps:
                    print("[STOP] max_steps reached")
                    stop_reason = "max_steps"
                    break
                continue

            step_idx += 1
            if step_idx >= max_steps:
                print("[STOP] max_steps reached")
                stop_reason = "max_steps"
                break

    except KeyboardInterrupt:
        print("\n[STOP] interrupted by user")
        stop_reason = "interrupted"

    finally:
        if target_body_name is not None and target_z_initial is not None:
            try:
                final_pose = env.get_object_pose(target_body_name)
                target_z_final = float(final_pose[2])
                ee_x, ee_y, _ = [float(v) for v in env.get_ee_pose()]
                final_xy_error = float(np.linalg.norm([ee_x - target_x, ee_y - target_y]))
            except Exception:
                pass

        if summary_json_path is not None and target_z_initial is not None:
            object_lift_delta = None
            strict_lift_success = False
            post_close_lift_delta = None
            invalid_reasons = []
            if target_z_final is not None:
                object_lift_delta = float(target_z_final - target_z_initial)
                if target_z_at_close is not None:
                    post_close_lift_delta = float(target_z_final - target_z_at_close)
                if object_lift_delta > max_valid_lift_delta:
                    invalid_reasons.append("object_lift_delta_too_large")
                if final_xy_error is not None and final_xy_error > max_final_xy_error_for_success:
                    invalid_reasons.append("final_xy_error_too_large")

                invalid_success_reason = invalid_success_reason or ";".join(invalid_reasons)
                strict_lift_success = bool(
                    object_lift_delta >= lift_success_threshold
                    and object_lift_delta <= max_valid_lift_delta
                    and final_xy_error is not None
                    and final_xy_error <= max_final_xy_error_for_success
                )
                if invalid_success_reason and object_lift_delta >= lift_success_threshold:
                    stop_reason = invalid_success_reason

            summary = {
                "target_color": target_color,
                "instruction": instruction,
                "seed": seed,
                "num_steps": num_logged_steps,
                "target_x": target_x,
                "target_y": target_y,
                "first_close_step": first_close_step,
                "final_xy_error": final_xy_error,
                "xy_error_at_close": xy_error_at_close,
                "target_z_initial": target_z_initial,
                "target_z_final": target_z_final,
                "object_lift_delta": object_lift_delta,
                "strict_lift_success": strict_lift_success,
                "stop_reason": stop_reason,
                "invalid_success_reason": invalid_success_reason,
                "max_valid_lift_delta": max_valid_lift_delta,
                "max_final_xy_error_for_success": max_final_xy_error_for_success,
                "gate_close_by_xy": gate_close_by_xy,
                "close_xy_threshold": close_xy_threshold,
                "assist_xy_alignment_before_close": assist_xy_alignment_before_close,
                "assist_xy_gain": assist_xy_gain,
                "assist_xy_max_step": assist_xy_max_step,
                "assist_xy_deadband": assist_xy_deadband,
                "assist_xy_total_correction": assist_xy_total_correction,
                "assist_xy_steps": assist_xy_steps,
                "post_close_lift_controller": post_close_lift_controller,
                "post_close_hold_steps": post_close_hold_steps,
                "post_close_lift_dz": post_close_lift_dz,
                "post_close_lift_steps": post_close_lift_steps,
                "post_close_keep_xy": post_close_keep_xy,
                "post_close_controller_steps": post_close_controller_steps,
                "post_close_forced_lift_steps": post_close_forced_lift_steps,
                "post_close_prevented_negative_z_steps": post_close_prevented_negative_z_steps,
                "total_xy_move": total_xy_move,
                "post_close_lift_delta": post_close_lift_delta,
                "retry_sum": retry_sum,
                "post_close_positive_z_steps": post_close_positive_z_steps,
            }
            summary_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_json_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
            print(f"[SUMMARY] wrote {summary_json_path}")

        if log_csv_file is not None:
            log_csv_file.close()
        env.close()


def clear_existing_images(out_dir: Path) -> None:
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

    deleted_count = 0
    for file_path in out_dir.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_exts:
            file_path.unlink()
            deleted_count += 1

    print(f"[CLEANUP] removed {deleted_count} existing image files from {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml_path", type=str, default="Raccoon_colored_cylinder.xml")
    parser.add_argument("--server_url", type=str, default=None, help="Direct HTTP URL, e.g. http://127.0.0.1:8000")
    parser.add_argument(
        "--instruction",
        type=str,
        default=None,
        help="OpenVLA prompt. If omitted, generated with dataset-style randomized wording.",
    )
    parser.add_argument(
        "--target_color",
        type=str,
        default="auto",
        choices=["auto", "random", *CYLINDER_COLORS],
        help="Target color. 'auto' uses the color in --instruction, or random if instruction has no color.",
    )
    parser.add_argument("--task_type", type=str, default=DEFAULT_TASK_TYPE, choices=["grasp", "push"])
    parser.add_argument("--target_kind", type=str, default=DEFAULT_TARGET_KIND, choices=["cylinder", "box"])
    parser.add_argument("--unnorm_key", type=str, default="raccoon_pick_place")
    parser.add_argument("--output_dir", type=str, default="rollout_outputs")
    parser.add_argument("--episode_id", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=1000000)
    parser.add_argument("--speed", type=int, default=90)
    parser.add_argument("--settle_seconds_per_action", type=float, default=0.1)
    parser.add_argument("--image_format", type=str, default="jpeg", choices=["jpeg", "png"])
    parser.add_argument("--jpeg_quality", type=int, default=50)
    parser.add_argument("--initial_settle_seconds", type=float, default=0.0)
    parser.add_argument("--delta_scale", type=float, default=1.0)
    parser.add_argument("--max_delta_xyz", type=float, default=0.01)
    parser.add_argument("--request_timeout", type=float, default=60.0)
    parser.add_argument("--use_viewer", action="store_true")
    parser.add_argument("--camera_name", type=str, default="front_view")
    parser.add_argument("--no_save_images", action="store_true", help="Do not write rollout frame PNGs to output_dir.")
    parser.add_argument(
        "--save_every_n_frames",
        type=int,
        default=1,
        help="When saving images, write only every Nth rollout frame.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--object_x_range", type=float, nargs=2, default=DEFAULT_OBJECT_X_RANGE)
    parser.add_argument("--object_y_range", type=float, nargs=2, default=DEFAULT_OBJECT_Y_RANGE)
    parser.add_argument("--min_object_distance", type=float, default=DEFAULT_MIN_OBJECT_DISTANCE)
    parser.add_argument("--box_xy", type=float, nargs=2, default=DEFAULT_BOX_XY)
    parser.add_argument("--box_yaw", type=float, default=DEFAULT_BOX_YAW)
    parser.add_argument(
        "--no_randomize_box",
        action="store_true",
        help="Disable box randomization and keep the box fixed at --box_xy/--box_yaw.",
    )
    parser.add_argument(
        "--no_randomize_objects",
        action="store_true",
        help="Disables randomization for all four colored cylinders.",
    )

    parser.add_argument("--use_ssh_tunnel", action="store_true", help="Connect to the inference server through SSH local port forwarding")
    parser.add_argument("--ssh_host", type=str, default="qlak315.iptime.org")
    parser.add_argument("--ssh_port", type=int, default=23000)
    parser.add_argument("--ssh_user", type=str, default="root")
    parser.add_argument("--ssh_password", type=str, default=None, help="Prefer OPENVLA_SSH_PASSWORD or --ssh_ask_password")
    parser.add_argument("--ssh_ask_password", action="store_true", help="Prompt for the SSH password interactively")
    parser.add_argument("--remote_server_host", type=str, default="127.0.0.1")
    parser.add_argument("--remote_server_port", type=int, default=8000)
    parser.add_argument("--local_server_host", type=str, default="127.0.0.1")
    parser.add_argument("--local_server_port", type=int, default=8000)
    parser.add_argument("--log_csv", type=str, default=None, help="Optional CSV path for V3/V4 action execution logging.")
    parser.add_argument(
        "--summary_json",
        type=str,
        default=None,
        help="Optional rollout summary JSON path. Defaults to --log_csv with .summary.json.",
    )
    parser.add_argument("--use_action_smoothing", action="store_true", help="Enable V4 exponential smoothing for xyz movement delta.")
    parser.add_argument("--smooth_alpha", type=float, default=0.6, help="V4 smoothing alpha. Used only with --use_action_smoothing.")
    parser.add_argument("--max_action_abs", type=float, default=None, help="Optional V4 xyz movement clipping limit.")
    parser.add_argument(
        "--stop_after_gripper_close",
        action="store_true",
        help="Stop rollout when gripper is closed and movement stays small.",
    )
    parser.add_argument(
        "--gripper_close_threshold",
        type=float,
        default=0.5,
        help="Gripper command threshold used for early stop.",
    )
    parser.add_argument(
        "--no_motion_threshold",
        type=float,
        default=0.003,
        help="Movement norm threshold used for early stop.",
    )
    parser.add_argument(
        "--no_motion_patience",
        type=int,
        default=5,
        help="Number of consecutive low-motion gripper-closed steps before stopping.",
    )
    parser.add_argument(
        "--stop_on_lift_success",
        action="store_true",
        help="Stop rollout after the target object is lifted enough after gripper close.",
    )
    parser.add_argument(
        "--lift_success_threshold",
        type=float,
        default=0.010,
        help="Target object z increase needed for lift success.",
    )
    parser.add_argument(
        "--min_steps_after_close",
        type=int,
        default=10,
        help="Minimum steps after first gripper close before lift-success stop can trigger.",
    )
    parser.add_argument(
        "--max_valid_lift_delta",
        type=float,
        default=0.08,
        help="Maximum plausible object lift delta for a valid grasp-lift success.",
    )
    parser.add_argument(
        "--max_final_xy_error_for_success",
        type=float,
        default=0.05,
        help="Maximum final EE-to-target xy error for strict lift success.",
    )
    parser.add_argument("--gate_close_by_xy", action="store_true", help="Keep gripper open when close is predicted far from the target xy.")
    parser.add_argument(
        "--close_xy_threshold",
        type=float,
        default=0.015,
        help="XY error threshold used for close gating and pre-close z safety.",
    )
    parser.add_argument(
        "--preclose_min_z_when_xy_bad",
        type=float,
        nargs="?",
        const=0.045,
        default=None,
        help="Enable pre-close z safety and clamp downward motion below this z when xy is bad. Default when enabled: 0.045.",
    )
    parser.add_argument("--latch_gripper_after_close", action="store_true", help="Keep gripper command closed after the first executed close.")
    parser.add_argument(
        "--post_close_clamp_negative_z",
        action="store_true",
        help="Clamp negative z actions to 0 after the first executed close.",
    )
    parser.add_argument(
        "--post_close_lift_controller",
        action="store_true",
        help="After first close, hold briefly and then force a small upward lift action.",
    )
    parser.add_argument(
        "--post_close_hold_steps",
        type=int,
        default=3,
        help="Post-close controller hold steps before forced lift starts.",
    )
    parser.add_argument(
        "--post_close_lift_dz",
        type=float,
        default=0.0025,
        help="Minimum z action during the post-close forced lift window.",
    )
    parser.add_argument(
        "--post_close_lift_steps",
        type=int,
        default=12,
        help="Number of post-hold steps to force upward lift action.",
    )
    parser.add_argument(
        "--post_close_keep_xy",
        action="store_true",
        help="Zero x/y action while the post-close lift controller is active.",
    )
    parser.add_argument(
        "--assist_xy_alignment_before_close",
        action="store_true",
        help="Add a small client-side xy correction toward the target before the first close.",
    )
    parser.add_argument(
        "--assist_xy_gain",
        type=float,
        default=0.5,
        help="Gain for pre-close xy assist.",
    )
    parser.add_argument(
        "--assist_xy_max_step",
        type=float,
        default=0.004,
        help="Maximum absolute x/y assist added per step.",
    )
    parser.add_argument(
        "--assist_xy_deadband",
        type=float,
        default=0.002,
        help="XY error deadband below which pre-close assist is disabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with maybe_tunnel_context(args) as tunnel:
        server_url = build_server_url(args, tunnel)

        if tunnel is not None:
            print(
                f"[SSH] {args.local_server_host}:{tunnel.local_bind_port} -> "
                f"{args.remote_server_host}:{args.remote_server_port}"
            )

        rollout(
            xml_path=args.xml_path,
            server_url=server_url,
            instruction=args.instruction,
            unnorm_key=args.unnorm_key,
            output_dir=args.output_dir,
            episode_id=args.episode_id,
            max_steps=args.max_steps,
            use_viewer=args.use_viewer,
            log_csv=args.log_csv,
            use_action_smoothing=args.use_action_smoothing,
            smooth_alpha=args.smooth_alpha,
            max_action_abs=args.max_action_abs,
            camera_name=args.camera_name,
            no_save_images=args.no_save_images,
            save_every_n_frames=args.save_every_n_frames,
            speed=args.speed,
            settle_seconds_per_action=args.settle_seconds_per_action,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
            initial_settle_seconds=args.initial_settle_seconds,
            delta_scale=args.delta_scale,
            randomize_objects=not args.no_randomize_objects,
            randomize_box=not args.no_randomize_box,
            request_timeout=args.request_timeout,
            max_delta_xyz=args.max_delta_xyz,
            target_color_arg=args.target_color,
            task_type=args.task_type,
            target_kind=args.target_kind,
            seed=args.seed,
            object_x_range=tuple(args.object_x_range),
            object_y_range=tuple(args.object_y_range),
            min_object_distance=args.min_object_distance,
            box_xy=tuple(args.box_xy),
            box_yaw=args.box_yaw,
            stop_after_gripper_close=args.stop_after_gripper_close,
            gripper_close_threshold=args.gripper_close_threshold,
            no_motion_threshold=args.no_motion_threshold,
            no_motion_patience=args.no_motion_patience,
            summary_json=args.summary_json,
            stop_on_lift_success=args.stop_on_lift_success,
            lift_success_threshold=args.lift_success_threshold,
            min_steps_after_close=args.min_steps_after_close,
            max_valid_lift_delta=args.max_valid_lift_delta,
            max_final_xy_error_for_success=args.max_final_xy_error_for_success,
            gate_close_by_xy=args.gate_close_by_xy,
            close_xy_threshold=args.close_xy_threshold,
            preclose_min_z_when_xy_bad=args.preclose_min_z_when_xy_bad,
            latch_gripper_after_close=args.latch_gripper_after_close,
            post_close_clamp_negative_z=args.post_close_clamp_negative_z,
            assist_xy_alignment_before_close=args.assist_xy_alignment_before_close,
            assist_xy_gain=args.assist_xy_gain,
            assist_xy_max_step=args.assist_xy_max_step,
            assist_xy_deadband=args.assist_xy_deadband,
            post_close_lift_controller=args.post_close_lift_controller,
            post_close_hold_steps=args.post_close_hold_steps,
            post_close_lift_dz=args.post_close_lift_dz,
            post_close_lift_steps=args.post_close_lift_steps,
            post_close_keep_xy=args.post_close_keep_xy,
        )


if __name__ == "__main__":
    main()
