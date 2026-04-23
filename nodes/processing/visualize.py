# Copyright (c) 2025 Andrea Pozzetti
# SPDX-License-Identifier: MIT
"""
Visualization nodes for SAM 3D Body outputs.

Provides nodes for rendering and visualizing 3D mesh reconstructions.
"""

import os
import sys
import json
import numpy as np
import cv2
import torch
from pathlib import Path
import folder_paths
from ..base import numpy_to_comfy_image
from ..runtime_deps import ensure_runtime_dependencies

# Add sam-3d-body to Python path if it exists
_SAM3D_BODY_PATH = Path(__file__).parent.parent.parent.parent.parent.parent / "sam-3d-body"
if _SAM3D_BODY_PATH.exists() and str(_SAM3D_BODY_PATH) not in sys.path:
    sys.path.insert(0, str(_SAM3D_BODY_PATH))


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return value


def _normalize(vec, fallback=None):
    norm = np.linalg.norm(vec)
    if norm < 1e-8:
        if fallback is None:
            return vec
        return np.asarray(fallback, dtype=np.float32)
    return vec / norm


def _rotation_matrix_xyz(rx_deg, ry_deg, rz_deg):
    rx, ry, rz = np.radians([rx_deg, ry_deg, rz_deg])

    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)

    rx_mat = np.array([
        [1, 0, 0],
        [0, cx, -sx],
        [0, sx, cx],
    ], dtype=np.float32)
    ry_mat = np.array([
        [cy, 0, sy],
        [0, 1, 0],
        [-sy, 0, cy],
    ], dtype=np.float32)
    rz_mat = np.array([
        [cz, -sz, 0],
        [sz, cz, 0],
        [0, 0, 1],
    ], dtype=np.float32)

    return rz_mat @ ry_mat @ rx_mat


def _orbit_basis_from_yp(yaw_deg, pitch_deg):
    return _rotation_matrix_xyz(pitch_deg, yaw_deg, 0.0)


def _apply_roll_to_basis(basis, roll_deg):
    roll = np.radians(roll_deg)
    c = np.cos(roll)
    s = np.sin(roll)

    ref_right = (basis @ np.array([1.0, 0.0, 0.0], dtype=np.float32)).astype(np.float32)
    ref_down = (basis @ np.array([0.0, 1.0, 0.0], dtype=np.float32)).astype(np.float32)
    forward = (basis @ np.array([0.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)

    right = (ref_right * c) + (ref_down * s)
    down = (ref_down * c) - (ref_right * s)

    return np.array(
        [
            [right[0], down[0], forward[0]],
            [right[1], down[1], forward[1]],
            [right[2], down[2], forward[2]],
        ],
        dtype=np.float32,
    )


def _camera_pose_look_at(position, target, roll_deg=0.0, world_up=None):
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    position = np.asarray(position, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    z_axis = _normalize(position - target, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    x_axis = np.cross(world_up, z_axis)
    if np.linalg.norm(x_axis) < 1e-6:
        alt_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        x_axis = np.cross(alt_up, z_axis)
    x_axis = _normalize(x_axis, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    y_axis = _normalize(np.cross(z_axis, x_axis), fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))

    if abs(roll_deg) > 1e-6:
        roll = np.radians(roll_deg)
        c = np.cos(roll)
        s = np.sin(roll)
        x_new = (x_axis * c) + (y_axis * s)
        y_new = (-x_axis * s) + (y_axis * c)
        x_axis, y_axis = x_new, y_new

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = x_axis
    pose[:3, 1] = y_axis
    pose[:3, 2] = z_axis
    pose[:3, 3] = position
    return pose


def _transform_vertices_to_render_space(vertices):
    transformed = vertices.copy().astype(np.float32)
    transformed[:, 1] *= -1.0
    transformed[:, 2] *= -1.0
    return transformed


def _transform_points_to_render_space(points):
    transformed = points.copy().astype(np.float32)
    transformed[..., 1] *= -1.0
    transformed[..., 2] *= -1.0
    return transformed


def _prepare_pyrender_backend():
    """
    Ensure pyrender uses a Windows-safe backend.

    SAM3DBody's bundled renderer defaults PYOPENGL_PLATFORM=egl, which breaks on
    standard Windows setups that do not ship an EGL loader. On Windows we want
    pyrender to use its default hidden pyglet window backend instead.
    """
    if os.name == "nt" and os.environ.get("PYOPENGL_PLATFORM", "").lower() == "egl":
        os.environ.pop("PYOPENGL_PLATFORM", None)


def _spherical_offset(yaw_deg, pitch_deg, radius):
    yaw = np.radians(yaw_deg)
    pitch = np.radians(pitch_deg)
    x = radius * np.cos(pitch) * np.sin(yaw)
    y = radius * np.sin(pitch)
    z = radius * np.cos(pitch) * np.cos(yaw)
    return np.array([x, y, z], dtype=np.float32)


def _decompose_orbit_offset(offset, world_up=None):
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    offset = np.asarray(offset, dtype=np.float32)
    radius = float(np.linalg.norm(offset))
    if radius < 1e-8:
        return 0.0, 0.0, 1.0

    up_amount = float(np.dot(offset, world_up) / radius)
    up_amount = float(np.clip(up_amount, -1.0, 1.0))
    pitch_deg = float(np.degrees(np.arcsin(up_amount)))

    horizontal = offset - (world_up * np.dot(offset, world_up))
    horizontal_len = float(np.linalg.norm(horizontal))
    if horizontal_len < 1e-8:
        yaw_deg = 0.0
    else:
        yaw_deg = float(np.degrees(np.arctan2(horizontal[0], horizontal[2])))

    return yaw_deg, pitch_deg, radius


def _upright_orbit_position(base_position, pivot, orbit_pitch_deg, orbit_yaw_deg, world_up=None):
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    pivot = np.asarray(pivot, dtype=np.float32)
    base_position = np.asarray(base_position, dtype=np.float32)
    base_offset = base_position - pivot

    base_yaw, base_pitch, radius = _decompose_orbit_offset(base_offset, world_up=world_up)
    final_pitch = float(np.clip(base_pitch + float(orbit_pitch_deg), -89.0, 89.0))
    final_yaw = base_yaw + float(orbit_yaw_deg)

    return pivot + _spherical_offset(final_yaw, final_pitch, radius), base_yaw, base_pitch, final_yaw, final_pitch


def _orbit_basis_from_yp(yaw_deg, pitch_deg, world_up=None):
    if world_up is None:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    forward = _normalize(-_spherical_offset(yaw_deg, pitch_deg, 1.0), fallback=np.array([0.0, 0.0, -1.0], dtype=np.float32))
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        right = np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=np.float32))
    right = _normalize(right, fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32))
    down = _normalize(np.cross(right, forward), fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))

    basis = np.eye(3, dtype=np.float32)
    basis[:, 0] = right
    basis[:, 1] = down
    basis[:, 2] = forward
    return basis


def _apply_roll_to_basis(basis, roll_deg):
    basis = np.asarray(basis, dtype=np.float32)
    if abs(float(roll_deg)) < 1e-6:
        return basis

    roll = np.radians(roll_deg)
    c = np.cos(roll)
    s = np.sin(roll)

    right = basis[:, 0]
    down = basis[:, 1]
    forward = basis[:, 2]

    rolled = np.eye(3, dtype=np.float32)
    rolled[:, 0] = right * c - down * s
    rolled[:, 1] = right * s + down * c
    rolled[:, 2] = forward
    return rolled


def _resolve_lighting(preset, ambient, key, fill, rim):
    if preset == "flat":
        return max(ambient, 0.6), key * 0.55, max(fill, key * 0.45), rim * 0.15
    if preset == "dramatic":
        return ambient * 0.45, key * 1.35, fill * 0.35, max(rim, key * 0.9)
    return ambient, key, fill, rim


def _parse_interactive_state(state_text):
    default = {
        "pivot_x": 0.0,
        "pivot_y": 0.0,
        "pivot_z": 0.0,
        "yaw_deg": 0.0,
        "pitch_deg": 0.0,
        "roll_deg": 0.0,
        "distance": 0.0,
    }
    if not state_text:
        return default
    try:
        data = json.loads(state_text)
        if not isinstance(data, dict):
            return default
        for key in default:
            if key in data:
                default[key] = float(data[key])
        return default
    except Exception:
        return default


def _state_has_interactive_camera(state):
    return abs(float(state.get("distance", 0.0))) > 1e-5


def _camera_axes(yaw_deg, pitch_deg, roll_deg):
    rot = _apply_roll_to_basis(_orbit_basis_from_yp(yaw_deg, pitch_deg), roll_deg)
    right = rot @ np.array([1.0, 0.0, 0.0], dtype=np.float32)
    down = rot @ np.array([0.0, 1.0, 0.0], dtype=np.float32)
    forward = rot @ np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return right.astype(np.float32), down.astype(np.float32), forward.astype(np.float32)


def _state_to_camera_pose(state):
    pivot = np.array([state["pivot_x"], state["pivot_y"], state["pivot_z"]], dtype=np.float32)
    orbit_basis = _orbit_basis_from_yp(float(state["yaw_deg"]), float(state["pitch_deg"]))
    orbit_forward = (orbit_basis @ np.array([0.0, 0.0, 1.0], dtype=np.float32)).astype(np.float32)
    orbit_forward = _normalize(orbit_forward, fallback=np.array([0.0, 0.0, 1.0], dtype=np.float32))
    position = (pivot - orbit_forward * float(state["distance"])).astype(np.float32)

    right, down, forward = _camera_axes(
        float(state["yaw_deg"]),
        float(state["pitch_deg"]),
        float(state["roll_deg"]),
    )
    up = (-down).astype(np.float32)
    backward = (-forward).astype(np.float32)

    pose = np.eye(4, dtype=np.float32)
    pose[:3, 0] = right
    pose[:3, 1] = up
    pose[:3, 2] = backward
    pose[:3, 3] = position
    return position, pivot, pose


def _camera_state_from_parameters(pivot, cam_pos, yaw_deg, pitch_deg, roll_deg):
    return {
        "pivot_x": float(pivot[0]),
        "pivot_y": float(pivot[1]),
        "pivot_z": float(pivot[2]),
        "yaw_deg": float(yaw_deg),
        "pitch_deg": float(pitch_deg),
        "roll_deg": float(roll_deg),
        "distance": float(np.linalg.norm(np.asarray(cam_pos, dtype=np.float32) - np.asarray(pivot, dtype=np.float32))),
    }


def _sample_preview_points(vertices, max_points=6000):
    vertices = np.asarray(vertices, dtype=np.float32)
    if len(vertices) <= max_points:
        return vertices
    indices = np.linspace(0, len(vertices) - 1, num=max_points, dtype=np.int32)
    return vertices[indices]


class SAM3DBodyVisualize:
    """
    Visualizes SAM 3D Body mesh reconstruction results.

    Renders the 3D mesh onto the input image for visualization purposes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT", {
                    "tooltip": "Mesh data from SAM3DBodyProcess node"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Original input image to overlay mesh on"
                }),
                "render_mode": (["overlay", "mesh_only", "side_by_side"], {
                    "default": "overlay",
                    "tooltip": "How to display the mesh: overlay on image, mesh only, or side-by-side comparison"
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rendered_image",)
    FUNCTION = "visualize"
    CATEGORY = "CamShotToolkit/visualization"

    def visualize(self, mesh_data, image, render_mode="overlay"):
        """Visualize the 3D mesh reconstruction."""
        ensure_runtime_dependencies("Cam Shot Toolkit: Visualize Mesh")

        print(f"[SAM3DBody] Visualizing mesh with mode: {render_mode}")

        try:
            from ..base import comfy_image_to_numpy

            # Get original image
            img_bgr = comfy_image_to_numpy(image)

            # Extract mesh components
            vertices = mesh_data.get("vertices", None)
            faces = mesh_data.get("faces", None)
            camera = mesh_data.get("camera", None)
            raw_output = mesh_data.get("raw_output", {})

            if vertices is None or faces is None:
                print(f"[SAM3DBody] [WARNING] No mesh data available for visualization")
                return (image,)

            # Convert tensors to numpy if needed
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.cpu().numpy()
            if isinstance(faces, torch.Tensor):
                faces = faces.cpu().numpy()

            print(f"[SAM3DBody] Rendering mesh with {len(vertices)} vertices, {len(faces)} faces")

            # Try to use the original visualization tools
            try:
                from pathlib import Path
                import sys
                sam_3d_body_path = Path(__file__).parent.parent.parent.parent.parent.parent / "sam-3d-body"
                if sam_3d_body_path.exists():
                    sys.path.insert(0, str(sam_3d_body_path))
                    from tools.vis_utils import visualize_sample_together

                    rendered = visualize_sample_together(img_bgr, raw_output, faces)

                    if render_mode == "mesh_only":
                        # Return just the rendered mesh (would need separate rendering)
                        result_img = rendered
                    elif render_mode == "side_by_side":
                        # Concatenate original and rendered side by side
                        result_img = np.hstack([img_bgr, rendered])
                    else:  # overlay
                        result_img = rendered

                    # Convert back to ComfyUI format
                    result_comfy = numpy_to_comfy_image(result_img)
                    print(f"[SAM3DBody] [OK] Visualization complete")
                    return (result_comfy,)

            except Exception as e:
                print(f"[SAM3DBody] [WARNING] Could not use visualization tools: {e}")
                print(f"[SAM3DBody] Returning original image")
                return (image,)

        except Exception as e:
            print(f"[SAM3DBody] [ERROR] Visualization failed: {str(e)}")
            import traceback
            traceback.print_exc()
            # Return original image on error
            return (image,)


class SAM3DBodyRenderOffsetView:
    """
    Render a new calibrated view from SAM3D mesh/camera output.

    Uses the recovered SAM3D camera translation and focal length as the base
    view, then applies an orbit and local camera offset on top.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT", {
                    "tooltip": "Mesh data from SAM3DBodyProcess node"
                }),
                "reference_image": ("IMAGE", {
                    "tooltip": "Reference image used to define the original camera intrinsics"
                }),
                "render_width": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                }),
                "render_height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 8,
                }),
                "enable_viewer": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Mount the interactive preview viewer for this node."
                }),
                "use_interactive_view": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If enabled, the browser viewer camera overrides the parameter-based camera when available."
                }),
                "show_viewer_hud": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Show the camera HUD overlay in the interactive viewer."
                }),
                "pivot_mode": (["mesh_center", "mesh_bottom", "root_joint", "origin"], {
                    "default": "mesh_center",
                    "tooltip": "Point to orbit the camera around"
                }),
                "orbit_x": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 0.5,
                }),
                "orbit_y": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 0.5,
                }),
                "orbit_z": ("FLOAT", {
                    "default": 0.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 0.5,
                    "tooltip": "Camera roll after orbit"
                }),
                "offset_x": ("FLOAT", {
                    "default": 0.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "Camera-local X translation after orbit"
                }),
                "offset_y": ("FLOAT", {
                    "default": 0.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "Camera-local Y translation after orbit"
                }),
                "offset_z": ("FLOAT", {
                    "default": 0.0,
                    "min": -5.0,
                    "max": 5.0,
                    "step": 0.01,
                    "tooltip": "Camera-local Z translation after orbit"
                }),
                "focal_scale": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.1,
                    "max": 4.0,
                    "step": 0.01,
                }),
                "principal_offset_x": ("FLOAT", {
                    "default": 0.0,
                    "min": -2048.0,
                    "max": 2048.0,
                    "step": 1.0,
                }),
                "principal_offset_y": ("FLOAT", {
                    "default": 0.0,
                    "min": -2048.0,
                    "max": 2048.0,
                    "step": 1.0,
                }),
                "lighting_preset": (["studio", "flat", "dramatic"], {
                    "default": "studio",
                    "tooltip": "Quick lighting profile"
                }),
                "ambient_intensity": ("FLOAT", {
                    "default": 0.35,
                    "min": 0.0,
                    "max": 3.0,
                    "step": 0.01,
                }),
                "key_intensity": ("FLOAT", {
                    "default": 14.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                }),
                "key_yaw": ("FLOAT", {
                    "default": 35.0,
                    "min": -180.0,
                    "max": 180.0,
                    "step": 0.5,
                }),
                "key_pitch": ("FLOAT", {
                    "default": 35.0,
                    "min": -89.0,
                    "max": 89.0,
                    "step": 0.5,
                }),
                "fill_intensity": ("FLOAT", {
                    "default": 6.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                }),
                "rim_intensity": ("FLOAT", {
                    "default": 8.0,
                    "min": 0.0,
                    "max": 100.0,
                    "step": 0.1,
                }),
                "mesh_r": ("INT", {"default": 235, "min": 0, "max": 255, "step": 1}),
                "mesh_g": ("INT", {"default": 235, "min": 0, "max": 255, "step": 1}),
                "mesh_b": ("INT", {"default": 235, "min": 0, "max": 255, "step": 1}),
                "bg_preset": (["mid_gray", "black", "white", "custom"], {
                    "default": "mid_gray",
                    "tooltip": "Quick background color preset"
                }),
                "bg_r": ("INT", {"default": 38, "min": 0, "max": 255, "step": 1}),
                "bg_g": ("INT", {"default": 38, "min": 0, "max": 255, "step": 1}),
                "bg_b": ("INT", {"default": 38, "min": 0, "max": 255, "step": 1}),
                "interactive_state": ("STRING", {
                    "default": "{\"pivot_x\":0,\"pivot_y\":0,\"pivot_z\":0,\"yaw_deg\":0,\"pitch_deg\":0,\"roll_deg\":0,\"distance\":0}",
                    "multiline": False,
                    "tooltip": "Hidden interactive camera state managed by the viewer."
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("rendered_image", "camera_info")
    FUNCTION = "render"
    CATEGORY = "CamShotToolkit/visualization"

    def render(
        self,
        mesh_data,
        reference_image,
        render_width=512,
        render_height=512,
        enable_viewer=True,
        use_interactive_view=True,
        show_viewer_hud=True,
        pivot_mode="mesh_center",
        orbit_x=0.0,
        orbit_y=0.0,
        orbit_z=0.0,
        offset_x=0.0,
        offset_y=0.0,
        offset_z=0.0,
        focal_scale=1.0,
        principal_offset_x=0.0,
        principal_offset_y=0.0,
        lighting_preset="studio",
        ambient_intensity=0.35,
        key_intensity=14.0,
        key_yaw=35.0,
        key_pitch=35.0,
        fill_intensity=6.0,
        rim_intensity=8.0,
        mesh_r=235,
        mesh_g=235,
        mesh_b=235,
        bg_preset="mid_gray",
        bg_r=38,
        bg_g=38,
        bg_b=38,
        interactive_state="",
    ):
        ensure_runtime_dependencies("Cam Shot Toolkit: Render Offset View")
        _prepare_pyrender_backend()
        try:
            import pyrender
            import trimesh
        except Exception as exc:
            raise RuntimeError(f"SAM3DBody offset renderer requires pyrender + trimesh: {exc}")

        vertices = _to_numpy(mesh_data.get("vertices"))
        faces = _to_numpy(mesh_data.get("faces"))
        camera = _to_numpy(mesh_data.get("camera"))
        focal_length = mesh_data.get("focal_length")
        joints = _to_numpy(mesh_data.get("joint_coords"))

        if vertices is None or faces is None:
            raise RuntimeError("Mesh vertices/faces not found in mesh_data")
        if camera is None:
            raise RuntimeError("Recovered SAM3D camera not found in mesh_data")
        if focal_length is None:
            raise RuntimeError("Recovered SAM3D focal_length not found in mesh_data")

        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)
        camera = np.asarray(camera, dtype=np.float32).reshape(-1)[:3]
        focal_length = float(np.asarray(focal_length).reshape(-1)[0])

        ref_img = reference_image[0].cpu().numpy()
        ref_h, ref_w = ref_img.shape[:2]
        scale_x = float(render_width) / float(ref_w)
        scale_y = float(render_height) / float(ref_h)
        fx = focal_length * scale_x * focal_scale
        fy = focal_length * scale_y * focal_scale
        cx = (ref_w * 0.5 * scale_x) + float(principal_offset_x)
        cy = (ref_h * 0.5 * scale_y) + float(principal_offset_y)

        verts_render = _transform_vertices_to_render_space(vertices)
        if joints is not None:
            joints = _transform_points_to_render_space(np.asarray(joints, dtype=np.float32))

        base_cam_pos = camera.copy()
        base_cam_pos[0] *= -1.0

        if pivot_mode == "mesh_bottom":
            bounds_min = verts_render.min(axis=0)
            bounds_max = verts_render.max(axis=0)
            pivot = np.array([
                0.5 * (bounds_min[0] + bounds_max[0]),
                bounds_min[1],
                0.5 * (bounds_min[2] + bounds_max[2]),
            ], dtype=np.float32)
        elif pivot_mode == "root_joint" and joints is not None and len(joints) > 0:
            pivot = joints[0].astype(np.float32)
        elif pivot_mode == "origin":
            pivot = np.zeros(3, dtype=np.float32)
        else:
            pivot = verts_render.mean(axis=0).astype(np.float32)

        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        base_forward = _normalize(pivot - base_cam_pos, fallback=np.array([0.0, 0.0, -1.0], dtype=np.float32))
        base_right = _normalize(np.cross(base_forward, world_up), fallback=np.array([1.0, 0.0, 0.0], dtype=np.float32))
        base_up = _normalize(np.cross(base_right, base_forward), fallback=np.array([0.0, 1.0, 0.0], dtype=np.float32))

        pre_orbit_cam_pos = (
            base_cam_pos
            + (base_right * float(offset_x))
            + (base_up * float(offset_y))
            + (base_forward * float(offset_z))
        )

        cam_pos, base_yaw, base_pitch, final_yaw, final_pitch = _upright_orbit_position(
            pre_orbit_cam_pos,
            pivot,
            orbit_pitch_deg=orbit_x,
            orbit_yaw_deg=orbit_y,
            world_up=world_up,
        )

        parameter_state = _camera_state_from_parameters(
            pivot=pivot,
            cam_pos=cam_pos,
            yaw_deg=final_yaw,
            pitch_deg=final_pitch,
            roll_deg=orbit_z,
        )
        parsed_interactive_state = _parse_interactive_state(interactive_state)
        active_state = (
            parsed_interactive_state
            if use_interactive_view and _state_has_interactive_camera(parsed_interactive_state)
            else parameter_state
        )
        cam_pos, pivot, camera_pose = _state_to_camera_pose(active_state)

        if bg_preset == "black":
            bg_r, bg_g, bg_b = 0, 0, 0
        elif bg_preset == "white":
            bg_r, bg_g, bg_b = 255, 255, 255
        elif bg_preset == "mid_gray":
            bg_r, bg_g, bg_b = 38, 38, 38

        mesh = trimesh.Trimesh(verts_render.copy(), faces.copy(), process=False)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            roughnessFactor=0.8,
            alphaMode="OPAQUE",
            baseColorFactor=(
                float(mesh_r) / 255.0,
                float(mesh_g) / 255.0,
                float(mesh_b) / 255.0,
                1.0,
            ),
        )

        renderer = pyrender.OffscreenRenderer(
            viewport_width=int(render_width),
            viewport_height=int(render_height),
        )

        ambient_intensity, key_intensity, fill_intensity, rim_intensity = _resolve_lighting(
            lighting_preset,
            float(ambient_intensity),
            float(key_intensity),
            float(fill_intensity),
            float(rim_intensity),
        )

        scene = pyrender.Scene(
            bg_color=[
                float(bg_r) / 255.0,
                float(bg_g) / 255.0,
                float(bg_b) / 255.0,
                0.0,
            ],
            ambient_light=(ambient_intensity, ambient_intensity, ambient_intensity),
        )
        scene.add(pyrender.Mesh.from_trimesh(mesh, material=material), "mesh")

        camera_node = pyrender.IntrinsicsCamera(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            zfar=1e12,
        )
        scene.add(camera_node, pose=camera_pose)

        mesh_extent = float(np.max(np.ptp(verts_render, axis=0)))
        light_radius = max(mesh_extent * 2.5, 2.5)

        if key_intensity > 0.0:
            key_pos = pivot + _spherical_offset(key_yaw, key_pitch, light_radius)
            scene.add(
                pyrender.PointLight(color=np.ones(3), intensity=key_intensity),
                pose=_camera_pose_look_at(key_pos, pivot, world_up=world_up),
            )
        if fill_intensity > 0.0:
            fill_pos = pivot + _spherical_offset(key_yaw - 55.0, max(10.0, key_pitch * 0.45), light_radius * 0.92)
            scene.add(
                pyrender.PointLight(color=np.ones(3), intensity=fill_intensity),
                pose=_camera_pose_look_at(fill_pos, pivot, world_up=world_up),
            )
        if rim_intensity > 0.0:
            rim_pos = pivot + _spherical_offset(key_yaw + 180.0, max(15.0, key_pitch * 0.7), light_radius * 1.08)
            scene.add(
                pyrender.PointLight(color=np.ones(3), intensity=rim_intensity),
                pose=_camera_pose_look_at(rim_pos, pivot, world_up=world_up),
            )

        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        renderer.delete()

        rendered_rgb = color[:, :, :3].astype(np.uint8)
        rendered_bgr = rendered_rgb[:, :, ::-1].copy()
        result = numpy_to_comfy_image(rendered_bgr)

        preview_points = _sample_preview_points(verts_render, max_points=6000)

        camera_info = {
            "render_width": int(render_width),
            "render_height": int(render_height),
            "enable_viewer": bool(enable_viewer),
            "use_interactive_view": bool(use_interactive_view),
            "show_viewer_hud": bool(show_viewer_hud),
            "pivot_mode": pivot_mode,
            "pivot": [float(x) for x in pivot],
            "camera_position": [float(x) for x in cam_pos],
            "camera_target": [float(x) for x in pivot],
            "base_orbit_yaw": float(base_yaw),
            "base_orbit_pitch": float(base_pitch),
            "final_orbit_yaw": float(final_yaw),
            "final_orbit_pitch": float(final_pitch),
            "focal_x": float(fx),
            "focal_y": float(fy),
            "principal_x": float(cx),
            "principal_y": float(cy),
            "lighting_preset": lighting_preset,
            "ambient_intensity": float(ambient_intensity),
            "key_intensity": float(key_intensity),
            "key_yaw": float(key_yaw),
            "key_pitch": float(key_pitch),
            "fill_intensity": float(fill_intensity),
            "rim_intensity": float(rim_intensity),
            "orbit_x": float(orbit_x),
            "orbit_y": float(orbit_y),
            "orbit_z": float(orbit_z),
            "offset_x": float(offset_x),
            "offset_y": float(offset_y),
            "offset_z": float(offset_z),
            "bg_preset": bg_preset,
            "bg_color": [int(bg_r), int(bg_g), int(bg_b)],
        }
        ui_data = {
            "preview_points": [json.dumps(preview_points.tolist())],
            "parameter_camera_state": [json.dumps(parameter_state)],
            "active_camera_state": [json.dumps(active_state)],
            "render_size": [json.dumps({
                "width": int(render_width),
                "height": int(render_height),
                "fx": float(fx),
                "fy": float(fy),
                "cx": float(cx),
                "cy": float(cy),
            })],
            "bg_color": [json.dumps([int(bg_r), int(bg_g), int(bg_b)])],
        }

        return {"ui": ui_data, "result": (result, json.dumps(camera_info))}


class SAM3DBodyExportMesh:
    """
    Exports SAM 3D Body mesh to STL format.

    Saves the reconstructed 3D mesh as ASCII STL for use in 3D viewers and editors.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT", {
                    "tooltip": "Mesh data from SAM3DBodyProcess node"
                }),
                "filename": ("STRING", {
                    "default": "output_mesh.stl",
                    "tooltip": "Output filename (exports as ASCII STL)"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "export_mesh"
    CATEGORY = "CamShotToolkit/io"

    def export_mesh(self, mesh_data, filename="output_mesh.stl"):
        """Export mesh to file."""

        print(f"[SAM3DBody] Exporting mesh to {filename}")

        try:
            import os

            # Use ComfyUI's output directory
            output_dir = folder_paths.get_output_directory()
            full_path = os.path.join(output_dir, filename)

            # Extract mesh data
            vertices = mesh_data.get("vertices", None)
            faces = mesh_data.get("faces", None)

            if vertices is None or faces is None:
                raise ValueError("No mesh data available to export")

            # Convert to numpy if needed
            if isinstance(vertices, torch.Tensor):
                vertices = vertices.cpu().numpy()
            if isinstance(faces, torch.Tensor):
                faces = faces.cpu().numpy()

            # Export to STL format
            self._export_stl(vertices, faces, full_path)

            print(f"[SAM3DBody] [OK] Mesh exported to {full_path}")
            return (filename,)

        except Exception as e:
            print(f"[SAM3DBody] [ERROR] Export failed: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _export_obj(self, vertices, faces, filepath):
        """Export mesh to OBJ format."""
        with open(filepath, 'w') as f:
            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write faces (OBJ uses 1-based indexing)
            for face in faces:
                f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

    def _export_ply(self, vertices, faces, filepath):
        """Export mesh to PLY format."""
        with open(filepath, 'w') as f:
            # Write PLY header
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(vertices)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write(f"element face {len(faces)}\n")
            f.write("property list uchar int vertex_indices\n")
            f.write("end_header\n")

            # Write vertices
            for v in vertices:
                f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write faces
            for face in faces:
                f.write(f"3 {face[0]} {face[1]} {face[2]}\n")

    def _export_stl(self, vertices, faces, filepath):
        """Export mesh to ASCII STL format."""
        import numpy as np

        # Apply 180° X-rotation to undo MHR coordinate transform (flip both Y and Z)
        # This matches what the renderer does for visualization
        vertices_flipped = vertices.copy()
        vertices_flipped[:, 1] = -vertices_flipped[:, 1]  # Flip Y
        vertices_flipped[:, 2] = -vertices_flipped[:, 2]  # Flip Z

        with open(filepath, 'w') as f:
            # Write STL header
            f.write("solid mesh\n")

            # Write each triangle face
            for face in faces:
                # Get the three vertices of the triangle
                v0 = vertices_flipped[int(face[0])]
                v1 = vertices_flipped[int(face[1])]
                v2 = vertices_flipped[int(face[2])]

                # Calculate face normal using cross product
                edge1 = v1 - v0
                edge2 = v2 - v0
                normal = np.cross(edge1, edge2)

                # Normalize the normal vector
                norm_length = np.linalg.norm(normal)
                if norm_length > 0:
                    normal = normal / norm_length
                else:
                    normal = np.array([0.0, 0.0, 1.0])  # Default normal if degenerate

                # Write facet
                f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
                f.write("    outer loop\n")
                f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
                f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
                f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
                f.write("    endloop\n")
                f.write("  endfacet\n")

            # Write STL footer
            f.write("endsolid mesh\n")


class SAM3DBodyGetVertices:
    """
    Extracts vertex data from SAM 3D Body output.

    Useful for custom processing or analysis of the reconstructed mesh.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT", {
                    "tooltip": "Mesh data from SAM3DBodyProcess node"
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("info",)
    FUNCTION = "get_vertices"
    CATEGORY = "CamShotToolkit/utilities"

    def get_vertices(self, mesh_data):
        """Extract and display vertex information."""

        try:
            vertices = mesh_data.get("vertices", None)
            faces = mesh_data.get("faces", None)
            joints = mesh_data.get("joints", None)

            info_lines = ["[SAM3DBody] Mesh Information:"]

            if vertices is not None:
                if isinstance(vertices, torch.Tensor):
                    vertices = vertices.cpu().numpy()
                info_lines.append(f"Vertices: {len(vertices)} points")
                info_lines.append(f"Vertex shape: {vertices.shape}")

            if faces is not None:
                if isinstance(faces, torch.Tensor):
                    faces = faces.cpu().numpy()
                info_lines.append(f"Faces: {len(faces)} triangles")

            if joints is not None:
                if isinstance(joints, torch.Tensor):
                    joints = joints.cpu().numpy()
                info_lines.append(f"Joints: {len(joints)} keypoints")

            info = "\n".join(info_lines)
            print(info)

            return (info,)

        except Exception as e:
            error_msg = f"[SAM3DBody] [ERROR] Failed to get mesh info: {str(e)}"
            print(error_msg)
            return (error_msg,)


# Register nodes
NODE_CLASS_MAPPINGS = {
    "CamShotToolkitRenderOffsetView": SAM3DBodyRenderOffsetView,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CamShotToolkitRenderOffsetView": "Cam Shot Toolkit: Render Offset View",
}
