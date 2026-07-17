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
        people.append({
            "person_index": int(person.get("person_index", fallback_index)),
            "vertices": np.asarray(vertices, dtype=np.float32).reshape(-1, 3),
            "faces": np.asarray(faces, dtype=np.int64).reshape(-1, 3),
            "joint_coords": _numpy_or_none(person.get("joint_coords")),
            "joint_rotations": _numpy_or_none(person.get("joint_rotations")),
            "camera": _numpy_or_none(person.get("camera")),
        })
    return people


def _numpy_or_none(value):
    arr = _to_numpy(value)
    if arr is None:
        return None
    return np.asarray(arr, dtype=np.float32)


def _position_people(people):
    """Offset each person by its camera translation so groups stay in relative
    world positions, then re-center the group around the origin. Mirrors the
    positioning the render/export nodes already use."""
    cameras = [p["camera"].reshape(-1)[:3] if p["camera"] is not None else None for p in people]

    if len(people) == 1 or all(c is None for c in cameras):
        for person in people:
            person["vertices_world"] = person["vertices"]
            person["joints_world"] = person["joint_coords"]
        return people

    world_verts = []
    for person, camera in zip(people, cameras):
        offset = camera.reshape(1, 3) if camera is not None else np.zeros((1, 3), np.float32)
        person["_offset"] = offset
        world_verts.append(person["vertices"] + offset)

    combined = np.concatenate(world_verts, axis=0)
    center = ((combined.max(axis=0) + combined.min(axis=0)) * 0.5).astype(np.float32).reshape(1, 3)
    for person in people:
        person["vertices_world"] = person["vertices"] + person["_offset"] - center
        if person["joint_coords"] is not None:
            person["joints_world"] = person["joint_coords"] + person["_offset"] - center
        else:
            person["joints_world"] = None
    return people


# 180-degree flip about X (negate Y and Z): MHR camera space -> glTF Y-up upright.
# Same transform the STL exporter applies. det(F)=+1 so winding/handedness hold.
_FLIP = np.array([1.0, -1.0, -1.0], dtype=np.float32)


def _flip_points(points):
    return (np.asarray(points, dtype=np.float32) * _FLIP).astype(np.float32)


def _flip_rotations(rotations):
    # R' = F R F  (F = diag(1,-1,-1), F == F^-1)
    f = np.diag(_FLIP).astype(np.float32)
    return np.einsum("ij,njk,kl->nil", f, np.asarray(rotations, dtype=np.float32), f).astype(np.float32)


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
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "CamShotToolkit/io"

    def save(self, mesh_data, filename_prefix="sam3d/body", rigging="auto", model=None, skeleton=None):
        ensure_runtime_dependencies("Cam Shot Toolkit: Save Meshes (GLB)")

        people = _extract_people(mesh_data)
        if not people:
            raise RuntimeError("No mesh vertices/faces found in mesh_data")

        people = _position_people(people)

        want_rig = rigging in ("auto", "skeleton_only")
        bind_skin = rigging == "auto"

        # Resolve rig data sources once (shared across people: same topology).
        model_obj = _load_model_object(model) if (want_rig and model is not None) else None
        joint_parents = _parents_from_skeleton(skeleton)
        if joint_parents is None and model_obj is not None:
            joint_parents = _try_extract_joint_parents(model_obj)

        real_weights = None
        if bind_skin and model_obj is not None:
            num_verts = people[0]["vertices"].shape[0]
            real_weights, _ = _try_extract_skin_weights(model_obj, num_verts)

        payload = []
        for person in people:
            name = f"person_{person['person_index']:03d}"
            vertices = _flip_points(person["vertices_world"])
            faces = person["faces"]

            entry = {"name": name, "vertices": vertices, "faces": faces}

            joints_world = person.get("joints_world")
            if want_rig and joints_world is not None and len(joints_world) > 0:
                joints = _flip_points(joints_world)
                rotations = person.get("joint_rotations")
                rotations = _flip_rotations(rotations) if rotations is not None else None
                num_joints = len(joints)

                parents = joint_parents
                if parents is None or len(parents) != num_joints:
                    # flat armature: every joint a root (no reposing hierarchy)
                    parents = np.full((num_joints,), -1, dtype=np.int64)

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
                    joint_names=[f"joint_{i}" for i in range(num_joints)],
                )
            payload.append(entry)

        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, folder_paths.get_output_directory()
        )
        os.makedirs(full_output_folder, exist_ok=True)
        out_name = f"{filename}_{counter:05d}.glb"
        out_path = os.path.join(full_output_folder, out_name)

        glb_export.write_glb(payload, out_path)
        rigged = sum(1 for e in payload if "rig" in e)
        print(f"[SAM3DBody] Save Meshes: wrote {len(payload)} mesh(es) "
              f"({rigged} rigged) -> {out_path}")

        return {"ui": {"text": [out_path]}, "result": (out_path,)}


NODE_CLASS_MAPPINGS = {
    "CamShotToolkitSaveMeshesGLB": SAM3DBodySaveMeshesGLB,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CamShotToolkitSaveMeshesGLB": "Cam Shot Toolkit: Save Meshes (GLB)",
}
