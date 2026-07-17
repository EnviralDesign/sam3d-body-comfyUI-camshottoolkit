# Copyright (c) 2025 Cam Shot Toolkit contributors
# SPDX-License-Identifier: MIT
"""
Save reconstructed SAM3D bodies to disk as a rigged GLB.

Sits after the SAM3D process/detector node. Writes a single .glb containing one
named mesh object per detected person, each bound to its own armature so the
skeleton travels with the mesh into Blender (or any glTF-aware app).

Rigging fidelity, best effort:
  1. Try to pull the real linear-blend-skinning weights + joint hierarchy out of
     the loaded MHR model (requires the optional ``model`` input).
  2. Otherwise fall back to auto skin weights (nearest-bone / distance falloff)
     plus whatever joint hierarchy we can find, so the mesh still deforms.
  3. ``skeleton_only`` embeds the armature without binding the mesh; ``none``
     writes plain meshes.
"""

import os

from ..lazy_import import LazyModule
from ..runtime_deps import ensure_runtime_dependencies
from . import glb_export

np = LazyModule("numpy")
torch = LazyModule("torch")
folder_paths = LazyModule("folder_paths")

# Momentum Human Rig has 127 skeleton joints.
_NUM_JOINTS = 127


def _to_numpy(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _mesh_people(mesh_data):
    people = mesh_data.get("people")
    if isinstance(people, list) and people:
        return people
    return [mesh_data]


def _extract_people(mesh_data):
    """Pull posed verts/faces/joints/rotations/camera for each selected person."""
    people = []
    for fallback_index, person in enumerate(_mesh_people(mesh_data)):
        vertices = _to_numpy(person.get("vertices"))
        faces = _to_numpy(person.get("faces"))
        if vertices is None or faces is None:
            continue
        focal = person.get("focal_length")
        focal_val = None
        if focal is not None:
            try:
                focal_val = float(np.asarray(focal).reshape(-1)[0])
            except Exception:
                focal_val = None
        people.append({
            "person_index": int(person.get("person_index", fallback_index)),
            "vertices": np.asarray(vertices, dtype=np.float32).reshape(-1, 3),
            "faces": np.asarray(faces, dtype=np.int64).reshape(-1, 3),
            "joint_coords": _numpy_or_none(person.get("joint_coords")),
            "joint_rotations": _numpy_or_none(person.get("joint_rotations")),
            "camera": _numpy_or_none(person.get("camera")),
            "focal_length": focal_val,
        })
    return people


def _numpy_or_none(value):
    arr = _to_numpy(value)
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float32)


# 180-degree flip about X (negate Y and Z): takes SAM3D's CV camera frame
# (X-right, Y-down, +Z-forward, camera at origin) into the glTF convention
# (X-right, Y-up, -Z-forward). det(F)=+1 so winding/handedness hold. Same
# transform the STL exporter applies.
_FLIP = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def _flip_points(points):
    return (np.asarray(points, dtype=np.float32) * _FLIP).astype(np.float32)


def _flip_rotations(rotations):
    # R' = F R F  (F = diag(1,-1,-1), F == F^-1)
    f = np.diag(_FLIP).astype(np.float32)
    return np.einsum("ij,njk,kl->nil", f, np.asarray(rotations, dtype=np.float32), f).astype(np.float32)


def _place_and_anchor(people):
    """Position people and produce flipped (export-space) geometry.

    When every person has a camera translation we keep the true camera-relative
    layout (camera at the origin, people at verts + cam_t) so the framing matches
    the capture. We then rigidly translate the WHOLE scene (people + camera) so
    the group's bounding box is centered on the floor-plane axes (X=0, Z=0 in the
    Y-up export frame) and the lowest foot rests on the floor (Y=0). This keeps
    coordinates numerically sane (group sits over the origin) while preserving
    each person's real relative height and the camera's relative framing.

    Returns whether a real camera layout is available and the vec3 the camera
    ends up at (it started at the origin, so it equals the applied translation).
    """
    has_camera = all(p["camera"] is not None for p in people)

    for person in people:
        verts = person["vertices"]
        joints = person["joint_coords"]
        if has_camera:
            offset = person["camera"].reshape(-1)[:3].reshape(1, 3)
            world_v = verts + offset
            world_j = (joints + offset) if joints is not None else None
        else:
            world_v = verts
            world_j = joints
        person["fverts"] = _flip_points(world_v)
        person["fjoints"] = _flip_points(world_j) if world_j is not None else None

    combined = np.concatenate([p["fverts"] for p in people], axis=0)
    mn = combined.min(axis=0)
    mx = combined.max(axis=0)
    # center the two floor-plane axes (X, Z); drop feet (min Y) to the floor.
    translation = np.array([
        -(mn[0] + mx[0]) * 0.5,
        -mn[1],
        -(mn[2] + mx[2]) * 0.5,
    ], dtype=np.float32)

    for person in people:
        person["fverts"] = person["fverts"] + translation
        if person["fjoints"] is not None:
            person["fjoints"] = person["fjoints"] + translation

    # Camera started at the origin, so it moves by the same translation.
    camera_translation = translation
    return has_camera, camera_translation


def _build_camera(people, camera_translation, reference_image):
    """Build a glTF perspective camera matching the SAM3D capture, or None."""
    if reference_image is None:
        return None
    focal = next((p["focal_length"] for p in people if p.get("focal_length")), None)
    if not focal:
        return None
    try:
        ref = reference_image[0]
        ref_h = int(ref.shape[0])
        ref_w = int(ref.shape[1])
    except Exception:
        return None
    if ref_h <= 0 or ref_w <= 0:
        return None
    import math
    yfov = 2.0 * math.atan((ref_h * 0.5) / float(focal))
    return {
        "name": "capture_camera",
        "yfov": float(yfov),
        "aspect": float(ref_w) / float(ref_h),
        # camera at origin, identity rotation (looks -Z, +Y up), moved with the group
        "translation": tuple(float(x) for x in camera_translation),
        "rotation": (0.0, 0.0, 0.0, 1.0),
        "znear": 0.05,
        "zfar": 1000.0,
    }


# --------------------------------------------------------------------------- #
# Best-effort extraction of real rig data from the loaded MHR model.
# --------------------------------------------------------------------------- #
def _load_model_object(model_config):
    """Return the loaded SAM3D model object from a SAM3D_MODEL config dict."""
    try:
        from .process import _load_sam3d_model
        return _load_sam3d_model(model_config).get("model")
    except Exception as exc:
        print(f"[SAM3DBody] Save Meshes: could not load model for rig extraction: {exc}")
        return None


def _iter_state_dict(model_obj):
    try:
        return list(model_obj.state_dict().items())
    except Exception:
        return []


def _get_by_path(obj, path):
    for attr in path:
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj


# The MHR (Momentum) rig stores linear-blend-skinning as a flattened sparse
# triplet list rather than a dense matrix, which is why a plain 2D scan misses
# it. These are the module paths / buffer name suffixes it lives under.
_LBS_MODULE_PATHS = [
    ("head_pose", "mhr", "character_torch", "linear_blend_skinning"),
    ("mhr_head", "mhr", "character_torch", "linear_blend_skinning"),
]
_LBS_FIELDS = ("vert_indices_flattened", "skin_indices_flattened", "skin_weights_flattened")


def _try_reconstruct_sparse_lbs(model_obj, num_verts, num_joints=_NUM_JOINTS):
    """Rebuild a dense [num_verts, num_joints] weight matrix from the MHR
    flattened sparse skinning buffers (vert idx / joint idx / weight)."""
    if model_obj is None:
        return None, None

    triplet = None
    source = None
    for path in _LBS_MODULE_PATHS:
        module = _get_by_path(model_obj, path)
        if module is None:
            continue
        parts = [_to_numpy(getattr(module, field, None)) for field in _LBS_FIELDS]
        if all(part is not None for part in parts):
            triplet = parts
            source = ".".join(path)
            break

    if triplet is None:  # fall back to matching flattened buffers by name suffix
        by_suffix = {}
        for name, tensor in _iter_state_dict(model_obj):
            for field in _LBS_FIELDS:
                if name.endswith("linear_blend_skinning." + field):
                    by_suffix[field] = _to_numpy(tensor)
        if all(field in by_suffix for field in _LBS_FIELDS):
            triplet = [by_suffix[field] for field in _LBS_FIELDS]
            source = "state_dict:linear_blend_skinning"

    if triplet is None:
        return None, None

    vert_idx, joint_idx, weights = (arr.reshape(-1) for arr in triplet)
    if not (len(vert_idx) == len(joint_idx) == len(weights)) or len(vert_idx) == 0:
        return None, None

    max_joint = int(joint_idx.max())
    n_joints = max(num_joints, max_joint + 1)
    max_vert = int(vert_idx.max())
    n_verts = max(num_verts, max_vert + 1)

    dense = np.zeros((n_verts, n_joints), dtype=np.float32)
    dense[vert_idx.astype(np.int64), joint_idx.astype(np.int64)] = weights.astype(np.float32)
    dense = dense[:num_verts, :num_joints]
    dense = np.clip(dense, 0.0, None)
    dense = dense / np.maximum(dense.sum(axis=1, keepdims=True), 1e-8)
    return dense, source


def _try_extract_skin_weights(model_obj, num_verts, num_joints=_NUM_JOINTS):
    """Get real skin weights: first the MHR sparse LBS buffers, then any dense
    [num_verts, num_joints] matrix as a secondary heuristic."""
    if model_obj is None:
        return None, None

    dense, source = _try_reconstruct_sparse_lbs(model_obj, num_verts, num_joints)
    if dense is not None:
        print(f"[SAM3DBody] Save Meshes: using real skin weights from {source}")
        return dense, source

    for name, tensor in _iter_state_dict(model_obj):
        try:
            shape = tuple(int(s) for s in tensor.shape)
        except Exception:
            continue
        if len(shape) != 2:
            continue
        arr = None
        if shape == (num_verts, num_joints):
            arr = tensor.detach().cpu().float().numpy()
        elif shape == (num_joints, num_verts):
            arr = tensor.detach().cpu().float().numpy().T
        if arr is None:
            continue
        # skinning weights are non-negative and rows sum to ~1
        if arr.min() < -1e-3:
            continue
        row_sums = arr.sum(axis=1)
        finite = np.isfinite(row_sums)
        if finite.mean() < 0.99:
            continue
        if np.abs(row_sums[finite] - 1.0).mean() > 0.25:
            continue
        arr = np.clip(arr, 0.0, None)
        arr = arr / np.maximum(arr.sum(axis=1, keepdims=True), 1e-8)
        print(f"[SAM3DBody] Save Meshes: using real skin weights from '{name}' {shape}")
        return arr.astype(np.float32), name
    return None, None


def _try_extract_joint_parents(model_obj, num_joints=_NUM_JOINTS):
    """Find the [num_joints] parent-index array in the model, by attribute path
    or by scanning integer buffers named like a parent table."""
    if model_obj is None:
        return None
    attr_paths = [
        ("mhr_head", "mhr", "character_torch", "skeleton", "joint_parents"),
        ("head_pose", "mhr", "character_torch", "skeleton", "joint_parents"),
        ("mhr_head", "mhr", "skeleton", "joint_parents"),
        ("head_pose", "mhr", "skeleton", "joint_parents"),
    ]
    for path in attr_paths:
        obj = model_obj
        ok = True
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                ok = False
                break
        if ok:
            arr = _to_numpy(obj)
            if arr is not None and arr.reshape(-1).shape[0] == num_joints:
                print(f"[SAM3DBody] Save Meshes: joint hierarchy from attr {'.'.join(path)}")
                return arr.reshape(-1).astype(np.int64)

    for name, tensor in _iter_state_dict(model_obj):
        try:
            shape = tuple(int(s) for s in tensor.shape)
        except Exception:
            continue
        if shape != (num_joints,) or "parent" not in name.lower():
            continue
        arr = tensor.detach().cpu().numpy().reshape(-1)
        if np.issubdtype(arr.dtype, np.integer) and arr.min() >= -1 and arr.max() < num_joints:
            print(f"[SAM3DBody] Save Meshes: joint hierarchy from buffer '{name}'")
            return arr.astype(np.int64)
    return None


def _try_extract_joint_names(model_obj, num_joints=_NUM_JOINTS):
    """Get the MHR skeleton's anatomical joint names (l_uparm, c_spine0, ...)
    so exported bones are pickable in a DCC instead of joint_0..joint_126."""
    if model_obj is None:
        return None
    name_paths = [
        ("head_pose", "mhr", "character_torch", "skeleton", "joint_names"),
        ("mhr_head", "mhr", "character_torch", "skeleton", "joint_names"),
        ("head_pose", "mhr", "skeleton", "joint_names"),
        ("mhr_head", "mhr", "skeleton", "joint_names"),
    ]
    for path in name_paths:
        obj = _get_by_path(model_obj, path)
        if obj is None:
            continue
        try:
            names = [str(n) for n in obj]
        except Exception:
            continue
        if len(names) == num_joints:
            print(f"[SAM3DBody] Save Meshes: joint names from attr {'.'.join(path)}")
            return names
    return None


def _parents_from_skeleton(skeleton):
    if not isinstance(skeleton, dict):
        return None
    arr = _to_numpy(skeleton.get("joint_parents"))
    if arr is None:
        return None
    return arr.reshape(-1).astype(np.int64)


class SAM3DBodySaveMeshesGLB:
    """Save detected SAM3D bodies to a single rigged GLB on disk."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("SAM3D_OUTPUT", {
                    "tooltip": "Mesh data from the SAM3D process node."
                }),
                "filename_prefix": ("STRING", {
                    "default": "sam3d/body",
                    "tooltip": "Output name under the ComfyUI output folder. Subfolders allowed."
                }),
                "rigging": (["auto", "skeleton_only", "none"], {
                    "default": "auto",
                    "tooltip": "auto: armature + skin binding (real weights if available, else auto weights). "
                               "skeleton_only: armature present but mesh not bound. none: plain meshes."
                }),
            },
            "optional": {
                "model": ("SAM3D_MODEL", {
                    "tooltip": "Optional. Enables extracting the real skin weights / joint hierarchy from the MHR model."
                }),
                "skeleton": ("SKELETON", {
                    "tooltip": "Optional. Provides the joint parent hierarchy for the armature."
                }),
                "reference_image": ("IMAGE", {
                    "tooltip": "Optional. The image the reconstruction came from; used to emit a "
                               "matching capture camera (FOV from focal length + aspect)."
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "CamShotToolkit/io"

    def save(self, mesh_data, filename_prefix="sam3d/body", rigging="auto",
             model=None, skeleton=None, reference_image=None):
        ensure_runtime_dependencies("Cam Shot Toolkit: Save Meshes (GLB)")

        people = _extract_people(mesh_data)
        if not people:
            raise RuntimeError("No mesh vertices/faces found in mesh_data")

        has_camera, camera_translation = _place_and_anchor(people)
        camera = _build_camera(people, camera_translation, reference_image) if has_camera else None

        want_rig = rigging in ("auto", "skeleton_only")
        bind_skin = rigging == "auto"

        # Resolve rig data sources once (shared across people: same topology).
        model_obj = _load_model_object(model) if (want_rig and model is not None) else None
        joint_parents = _parents_from_skeleton(skeleton)
        if joint_parents is None and model_obj is not None:
            joint_parents = _try_extract_joint_parents(model_obj)

        joint_names_all = _try_extract_joint_names(model_obj) if model_obj is not None else None

        real_weights = None
        if bind_skin and model_obj is not None:
            num_verts = people[0]["vertices"].shape[0]
            real_weights, _ = _try_extract_skin_weights(model_obj, num_verts)

        payload = []
        for person in people:
            name = f"person_{person['person_index']:03d}"
            vertices = person["fverts"]
            faces = person["faces"]

            entry = {"name": name, "vertices": vertices, "faces": faces}

            joints = person.get("fjoints")
            if want_rig and joints is not None and len(joints) > 0:
                rotations = person.get("joint_rotations")
                rotations = _flip_rotations(rotations) if rotations is not None else None
                num_joints = len(joints)

                parents = joint_parents
                if parents is None or len(parents) != num_joints:
                    # flat armature: every joint a root (no reposing hierarchy)
                    parents = np.full((num_joints,), -1, dtype=np.int64)

                if joint_names_all is not None and len(joint_names_all) == num_joints:
                    joint_names = list(joint_names_all)
                else:
                    joint_names = [f"joint_{i}" for i in range(num_joints)]

                if bind_skin:
                    if real_weights is not None and real_weights.shape == (len(vertices), num_joints):
                        v_joints, v_weights = glb_export._top_k_from_weight_matrix(real_weights, k=4)
                    else:
                        v_joints, v_weights = glb_export.auto_skin_weights(vertices, joints, k=4)
                else:
                    # skeleton_only: pin every vertex to nearest joint (visual only,
                    # keeps the armature bound so bones show, minimal deformation).
                    v_joints, v_weights = glb_export.auto_skin_weights(vertices, joints, k=1)

                entry["rig"] = glb_export.build_rig(
                    joint_positions=joints,
                    joint_rotations=rotations,
                    joint_parents=parents,
                    vertex_joints=v_joints,
                    vertex_weights=v_weights,
                    joint_names=joint_names,
                )
            payload.append(entry)

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        os.makedirs(full_output_folder, exist_ok=True)
        out_name = f"{filename}_{counter:05d}.glb"
        out_path = os.path.join(full_output_folder, out_name)

        glb_export.write_glb(payload, out_path, camera=camera)
        rigged = sum(1 for e in payload if "rig" in e)
        cam_note = "with camera" if camera is not None else "no camera"
        print(f"[SAM3DBody] Save Meshes: wrote {len(payload)} mesh(es) "
              f"({rigged} rigged, {cam_note}) -> {out_path}")

        return {"ui": {"text": [out_path]}, "result": (out_path,)}


NODE_CLASS_MAPPINGS = {
    "CamShotToolkitSaveMeshesGLB": SAM3DBodySaveMeshesGLB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CamShotToolkitSaveMeshesGLB": "Cam Shot Toolkit: Save Meshes (GLB)",
}
