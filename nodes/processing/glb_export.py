# Copyright (c) 2025 Cam Shot Toolkit contributors
# SPDX-License-Identifier: MIT
"""
Dependency-free GLB (binary glTF 2.0) writer with skinning support.

Writes a single .glb containing one or more named mesh objects, each optionally
bound to its own armature (skeleton). We hand-roll the glTF because trimesh /
the other libs bundled with ComfyUI do not export skins, joints, inverse-bind
matrices, or per-vertex skin weights.

glTF conventions worth remembering while reading this file:
  * Y-up, right-handed. SAM3D/MHR meshes come out in a camera space that is
    effectively Y-down, so callers flip Y and Z before handing data here (the
    same 180-deg-about-X flip the STL exporter applies).
  * Node matrices and inverse-bind matrices are stored COLUMN-major.
  * Node rotations are quaternions ordered [x, y, z, w].
  * A skinned mesh node's own transform is ignored by the runtime; vertices are
    placed purely by the joint transforms, so we keep mesh nodes at identity.
"""

from __future__ import annotations

import json
import struct

import numpy as np

# glTF component types
_FLOAT = 5126
_UNSIGNED_INT = 5125
_UNSIGNED_SHORT = 5123

# glTF bufferView targets
_ARRAY_BUFFER = 34962
_ELEMENT_ARRAY_BUFFER = 34963

_GLB_MAGIC = 0x46546C67  # "glTF"
_CHUNK_JSON = 0x4E4F534A  # "JSON"
_CHUNK_BIN = 0x004E4942   # "BIN\0"


def mat3_to_quat(matrix):
    """Convert a 3x3 rotation matrix to a glTF quaternion [x, y, z, w]."""
    m = np.asarray(matrix, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (m[2, 1] - m[1, 2]) * s
        y = (m[0, 2] - m[2, 0]) * s
        z = (m[1, 0] - m[0, 1]) * s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    quat = np.array([x, y, z, w], dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


class _GLBBuilder:
    """Accumulates buffer data + glTF JSON, then serializes to a GLB blob."""

    def __init__(self):
        self._bin = bytearray()
        self.buffer_views = []
        self.accessors = []
        self.meshes = []
        self.nodes = []
        self.skins = []
        self.cameras = []
        self.scene_nodes = []

    # -- low level -----------------------------------------------------------
    def _add_view(self, data, target=None):
        offset = len(self._bin)
        self._bin.extend(data)
        while len(self._bin) % 4 != 0:  # keep 4-byte alignment
            self._bin.append(0)
        view = {"buffer": 0, "byteOffset": offset, "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        self.buffer_views.append(view)
        return len(self.buffer_views) - 1

    def _add_accessor(self, array, component_type, type_str, target=None, with_minmax=False):
        array = np.ascontiguousarray(array)
        view = self._add_view(array.tobytes(), target)
        accessor = {
            "bufferView": view,
            "componentType": component_type,
            "count": int(array.shape[0]),
            "type": type_str,
        }
        if with_minmax:
            accessor["min"] = [float(v) for v in array.min(axis=0)]
            accessor["max"] = [float(v) for v in array.max(axis=0)]
        self.accessors.append(accessor)
        return len(self.accessors) - 1

    def _add_node(self, node):
        self.nodes.append(node)
        return len(self.nodes) - 1

    # -- high level ----------------------------------------------------------
    def add_person(self, name, vertices, normals, faces, rig=None):
        """Add one mesh object (with optional armature) to the scene."""
        vertices = np.ascontiguousarray(vertices, dtype=np.float32)
        normals = np.ascontiguousarray(normals, dtype=np.float32)
        faces = np.ascontiguousarray(faces, dtype=np.uint32).reshape(-1)

        pos_acc = self._add_accessor(vertices, _FLOAT, "VEC3", _ARRAY_BUFFER, with_minmax=True)
        nrm_acc = self._add_accessor(normals, _FLOAT, "VEC3", _ARRAY_BUFFER)
        idx_acc = self._add_accessor(faces, _UNSIGNED_INT, "SCALAR", _ELEMENT_ARRAY_BUFFER)

        attributes = {"POSITION": pos_acc, "NORMAL": nrm_acc}

        skin_index = None
        if rig is not None:
            joints = np.ascontiguousarray(rig["vertex_joints"], dtype=np.uint16)
            weights = np.ascontiguousarray(rig["vertex_weights"], dtype=np.float32)
            attributes["JOINTS_0"] = self._add_accessor(joints, _UNSIGNED_SHORT, "VEC4", _ARRAY_BUFFER)
            attributes["WEIGHTS_0"] = self._add_accessor(weights, _FLOAT, "VEC4", _ARRAY_BUFFER)
            skin_index = self._build_skin(name, rig)

        mesh_index = len(self.meshes)
        self.meshes.append({
            "name": name,
            "primitives": [{"attributes": attributes, "indices": idx_acc, "mode": 4}],
        })

        mesh_node = {"name": name, "mesh": mesh_index}
        if skin_index is not None:
            mesh_node["skin"] = skin_index
        mesh_node_index = self._add_node(mesh_node)
        self.scene_nodes.append(mesh_node_index)

    def _build_skin(self, name, rig):
        local_t = np.asarray(rig["joint_local_t"], dtype=np.float32)
        local_q = np.asarray(rig["joint_local_q"], dtype=np.float32)
        parents = np.asarray(rig["joint_parents"], dtype=np.int64).reshape(-1)
        names = rig.get("joint_names")
        num_joints = len(parents)

        # Create one node per joint first so we can reference them by index.
        joint_node_ids = []
        for j in range(num_joints):
            node = {
                "name": names[j] if names is not None else f"{name}_joint_{j}",
                "translation": [float(v) for v in local_t[j]],
                "rotation": [float(v) for v in local_q[j]],
            }
            joint_node_ids.append(self._add_node(node))

        # Wire up parent -> children relationships and collect roots.
        roots = []
        children_map = {}
        for j in range(num_joints):
            p = int(parents[j])
            if 0 <= p < num_joints and p != j:
                children_map.setdefault(p, []).append(joint_node_ids[j])
            else:
                roots.append(joint_node_ids[j])
        for parent_local, child_node_ids in children_map.items():
            self.nodes[joint_node_ids[parent_local]]["children"] = child_node_ids

        # A wrapper node gives Blender a clean single armature root.
        skeleton_root = self._add_node({"name": f"{name}_armature", "children": roots})
        self.scene_nodes.append(skeleton_root)

        # inverse-bind matrices: [K,4,4] row-major -> column-major flatten.
        ibm = np.asarray(rig["ibm"], dtype=np.float32).reshape(num_joints, 4, 4)
        ibm_col = np.transpose(ibm, (0, 2, 1)).reshape(num_joints, 16)
        ibm_acc = self._add_accessor(np.ascontiguousarray(ibm_col, dtype=np.float32), _FLOAT, "MAT4")

        self.skins.append({
            "name": f"{name}_skin",
            "joints": joint_node_ids,
            "inverseBindMatrices": ibm_acc,
            "skeleton": skeleton_root,
        })
        return len(self.skins) - 1

    def add_camera(self, name, yfov, aspect=None, translation=(0.0, 0.0, 0.0),
                   rotation=(0.0, 0.0, 0.0, 1.0), znear=0.05, zfar=1000.0):
        """Add a perspective camera. A node with identity rotation looks down
        local -Z with +Y up (glTF convention)."""
        perspective = {"yfov": float(yfov), "znear": float(znear), "zfar": float(zfar)}
        if aspect:
            perspective["aspectRatio"] = float(aspect)
        self.cameras.append({"type": "perspective", "perspective": perspective, "name": name})
        cam_index = len(self.cameras) - 1
        node_index = self._add_node({
            "name": name,
            "camera": cam_index,
            "translation": [float(x) for x in translation],
            "rotation": [float(x) for x in rotation],
        })
        self.scene_nodes.append(node_index)
        return cam_index

    def serialize(self):
        gltf = {
            "asset": {"version": "2.0", "generator": "sam3d-camshot-toolkit"},
            "scene": 0,
            "scenes": [{"nodes": self.scene_nodes}],
            "nodes": self.nodes,
            "meshes": self.meshes,
            "accessors": self.accessors,
            "bufferViews": self.buffer_views,
            "buffers": [{"byteLength": len(self._bin)}],
        }
        if self.skins:
            gltf["skins"] = self.skins
        if self.cameras:
            gltf["cameras"] = self.cameras

        json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
        while len(json_bytes) % 4 != 0:  # JSON chunk padded with spaces
            json_bytes += b" "
        bin_bytes = bytes(self._bin)
        while len(bin_bytes) % 4 != 0:   # BIN chunk padded with zeros
            bin_bytes += b"\x00"

        total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)
        out = bytearray()
        out += struct.pack("<III", _GLB_MAGIC, 2, total)
        out += struct.pack("<II", len(json_bytes), _CHUNK_JSON)
        out += json_bytes
        out += struct.pack("<II", len(bin_bytes), _CHUNK_BIN)
        out += bin_bytes
        return bytes(out)


def compute_vertex_normals(vertices, faces):
    """Area-weighted vertex normals."""
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(vertices)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    np.add.at(normals, faces[:, 0], face_normals)
    np.add.at(normals, faces[:, 1], face_normals)
    np.add.at(normals, faces[:, 2], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-8)
    return normals.astype(np.float32)


def _top_k_from_weight_matrix(weight_matrix, k=4):
    """Select up to k largest weights per vertex -> (indices[N,k], weights[N,k])."""
    weight_matrix = np.asarray(weight_matrix, dtype=np.float32)
    num_verts, num_joints = weight_matrix.shape
    k = min(k, num_joints)

    top_idx = np.argpartition(-weight_matrix, k - 1, axis=1)[:, :k]
    rows = np.arange(num_verts)[:, None]
    top_w = weight_matrix[rows, top_idx]

    # sort each row by weight descending (nice-to-have, keeps dominant bone first)
    order = np.argsort(-top_w, axis=1)
    top_idx = np.take_along_axis(top_idx, order, axis=1)
    top_w = np.take_along_axis(top_w, order, axis=1)

    sums = top_w.sum(axis=1, keepdims=True)
    zero_rows = sums[:, 0] <= 1e-12
    top_w = np.where(sums > 1e-12, top_w / np.maximum(sums, 1e-12), top_w)
    # vertices with no influence at all -> pin to joint 0 fully
    if np.any(zero_rows):
        top_idx[zero_rows, 0] = 0
        top_w[zero_rows] = 0.0
        top_w[zero_rows, 0] = 1.0

    if k < 4:  # pad to the vec4 glTF expects
        pad = 4 - k
        top_idx = np.concatenate([top_idx, np.zeros((num_verts, pad), dtype=top_idx.dtype)], axis=1)
        top_w = np.concatenate([top_w, np.zeros((num_verts, pad), dtype=top_w.dtype)], axis=1)
    return top_idx.astype(np.uint16), top_w.astype(np.float32)


def auto_skin_weights(vertices, joint_positions, k=4, falloff=8.0):
    """
    Heuristic linear-blend-skinning weights when real weights are unavailable.

    Each vertex is bound to its k nearest joints with inverse-distance falloff.
    Approximate, but produces a mesh that actually deforms when bones move.
    """
    vertices = np.asarray(vertices, dtype=np.float32)
    joints = np.asarray(joint_positions, dtype=np.float32)
    num_verts = len(vertices)
    num_joints = len(joints)
    k = min(k, num_joints)

    # squared distances, computed in chunks to bound memory on big meshes
    idx_out = np.zeros((num_verts, k), dtype=np.int64)
    w_out = np.zeros((num_verts, k), dtype=np.float32)
    chunk = 4096
    for start in range(0, num_verts, chunk):
        end = min(start + chunk, num_verts)
        diff = vertices[start:end, None, :] - joints[None, :, :]
        d2 = np.einsum("vjk,vjk->vj", diff, diff)
        nearest = np.argpartition(d2, k - 1, axis=1)[:, :k]
        rows = np.arange(end - start)[:, None]
        near_d2 = d2[rows, nearest]
        # inverse-distance^falloff, normalized against the closest joint for stability
        min_d2 = near_d2.min(axis=1, keepdims=True)
        w = np.power(min_d2 / np.maximum(near_d2, 1e-12), falloff * 0.5)
        idx_out[start:end] = nearest
        w_out[start:end] = w

    full = np.zeros((num_verts, num_joints), dtype=np.float32)
    np.put_along_axis(full, idx_out, w_out, axis=1)
    return _top_k_from_weight_matrix(full, k=4)


def build_rig(joint_positions, joint_rotations, joint_parents, vertex_joints, vertex_weights, joint_names=None):
    """
    Assemble the per-joint local transforms + inverse-bind matrices a skin needs.

    joint_positions:  [K,3] joint world positions in the SAME (flipped) space as
                      the exported vertices.
    joint_rotations:  [K,3,3] joint world rotation matrices, same space.
    joint_parents:    [K] parent index per joint (<0 for roots).

    The bind pose IS the exported (posed) pose: with local transforms derived
    from the world transforms and inverse-bind = inverse(world), each joint's
    skinning matrix is identity at rest, so the mesh renders exactly as exported
    and deforms correctly once bones are re-posed.
    """
    joint_positions = np.asarray(joint_positions, dtype=np.float64)
    num_joints = len(joint_positions)
    parents = np.asarray(joint_parents, dtype=np.int64).reshape(-1)

    if joint_rotations is None:
        rotations = np.tile(np.eye(3), (num_joints, 1, 1))
    else:
        rotations = np.asarray(joint_rotations, dtype=np.float64).reshape(num_joints, 3, 3)

    # world (global) transform per joint
    world = np.tile(np.eye(4), (num_joints, 1, 1))
    world[:, :3, :3] = rotations
    world[:, :3, 3] = joint_positions

    inverse_world = np.linalg.inv(world)

    local_t = np.zeros((num_joints, 3), dtype=np.float32)
    local_q = np.zeros((num_joints, 4), dtype=np.float32)
    for j in range(num_joints):
        p = int(parents[j])
        if 0 <= p < num_joints and p != j:
            local = inverse_world[p] @ world[j]
        else:
            local = world[j]
        local_t[j] = local[:3, 3].astype(np.float32)
        local_q[j] = mat3_to_quat(local[:3, :3])

    return {
        "joint_local_t": local_t,
        "joint_local_q": local_q,
        "joint_parents": parents,
        "joint_names": joint_names,
        "ibm": inverse_world.astype(np.float32),
        "vertex_joints": vertex_joints,
        "vertex_weights": vertex_weights,
    }


def write_glb(people, filepath, camera=None):
    """
    people: list of dicts, each:
        name:     str
        vertices: [N,3] float (already in export/flipped space)
        faces:    [F,3] int
        normals:  optional [N,3] float (computed if absent)
        rig:      optional rig dict from build_rig()
    camera: optional dict {name, yfov, aspect, translation, rotation, znear, zfar}
    """
    builder = _GLBBuilder()
    for person in people:
        vertices = np.asarray(person["vertices"], dtype=np.float32)
        faces = np.asarray(person["faces"], dtype=np.uint32).reshape(-1, 3)
        normals = person.get("normals")
        if normals is None:
            normals = compute_vertex_normals(vertices, faces)
        builder.add_person(person["name"], vertices, normals, faces, rig=person.get("rig"))

    if camera is not None:
        builder.add_camera(**camera)

    blob = builder.serialize()
    with open(filepath, "wb") as handle:
        handle.write(blob)
    return filepath
