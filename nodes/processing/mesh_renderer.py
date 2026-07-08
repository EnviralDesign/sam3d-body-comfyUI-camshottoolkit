from __future__ import annotations

import numpy as np


def _project_vertices(vertices, camera_pose, fx, fy, cx, cy):
    pose = np.asarray(camera_pose, dtype=np.float32).reshape(4, 4)
    rotation = pose[:3, :3]
    origin = pose[:3, 3]

    camera_space = (np.asarray(vertices, dtype=np.float32) - origin) @ rotation
    depth = -camera_space[:, 2]
    valid = depth > 1e-5

    projected = np.empty((len(vertices), 3), dtype=np.float32)
    safe_depth = np.where(valid, depth, 1.0)
    projected[:, 0] = (camera_space[:, 0] * float(fx) / safe_depth) + float(cx)
    projected[:, 1] = float(cy) - (camera_space[:, 1] * float(fy) / safe_depth)
    projected[:, 2] = depth
    return projected, valid


def _shade_faces(vertices, faces, camera_position, base_color, lights):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    normals = np.cross(v1 - v0, v2 - v0)
    normal_len = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(normal_len, 1e-8)

    centers = (v0 + v1 + v2) / 3.0
    view_dirs = camera_position.reshape(1, 3) - centers
    view_dirs = view_dirs / np.maximum(np.linalg.norm(view_dirs, axis=1, keepdims=True), 1e-8)
    away = np.sum(normals * view_dirs, axis=1) < 0.0
    normals[away] *= -1.0

    shade = np.full(len(faces), float(lights["ambient_intensity"]), dtype=np.float32)
    for key in ("key", "fill", "rim"):
        position = lights.get(f"{key}_position")
        intensity = float(lights.get(f"{key}_intensity", 0.0))
        if position is None or intensity <= 0.0:
            continue
        light_dirs = np.asarray(position, dtype=np.float32).reshape(1, 3) - centers
        light_dirs = light_dirs / np.maximum(np.linalg.norm(light_dirs, axis=1, keepdims=True), 1e-8)
        diffuse = np.maximum(np.sum(normals * light_dirs, axis=1), 0.0)
        shade += diffuse.astype(np.float32) * intensity * 0.04

    shade = np.clip(shade, 0.04, 1.35).reshape(-1, 1)
    return np.clip(np.asarray(base_color, dtype=np.float32).reshape(1, 3) * shade, 0, 255).astype(np.uint8)


def _rasterize_triangle(image, z_buffer, points, color):
    x0, y0, z0 = points[0]
    x1, y1, z1 = points[1]
    x2, y2, z2 = points[2]

    min_x = max(int(np.floor(min(x0, x1, x2))), 0)
    max_x = min(int(np.ceil(max(x0, x1, x2))), image.shape[1] - 1)
    min_y = max(int(np.floor(min(y0, y1, y2))), 0)
    max_y = min(int(np.ceil(max(y0, y1, y2))), image.shape[0] - 1)
    if min_x > max_x or min_y > max_y:
        return

    denom = ((y1 - y2) * (x0 - x2)) + ((x2 - x1) * (y0 - y2))
    if abs(float(denom)) < 1e-8:
        return

    xs = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
    ys = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)

    w0 = (((y1 - y2) * (grid_x - x2)) + ((x2 - x1) * (grid_y - y2))) / denom
    w1 = (((y2 - y0) * (grid_x - x2)) + ((x0 - x2) * (grid_y - y2))) / denom
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-5) & (w1 >= -1e-5) & (w2 >= -1e-5)
    if not np.any(inside):
        return

    depth = (w0 * z0) + (w1 * z1) + (w2 * z2)
    region_z = z_buffer[min_y:max_y + 1, min_x:max_x + 1]
    visible = inside & (depth < region_z)
    if not np.any(visible):
        return

    region_image = image[min_y:max_y + 1, min_x:max_x + 1]
    region_z[visible] = depth[visible]
    region_image[visible] = color


def render_mesh(
    vertices,
    faces,
    camera_pose,
    fx,
    fy,
    cx,
    cy,
    width,
    height,
    bg_color,
    mesh_color,
    lighting,
):
    width = int(width)
    height = int(height)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:, :] = np.asarray(bg_color, dtype=np.uint8).reshape(1, 1, 3)
    if len(vertices) == 0 or len(faces) == 0:
        return image

    projected, valid_vertices = _project_vertices(vertices, camera_pose, fx, fy, cx, cy)
    valid_faces = valid_vertices[faces].all(axis=1)
    if not np.any(valid_faces):
        return image

    faces = faces[valid_faces]
    camera_position = np.asarray(camera_pose, dtype=np.float32).reshape(4, 4)[:3, 3]
    face_colors = _shade_faces(vertices, faces, camera_position, mesh_color, lighting)
    face_depth = projected[faces, 2].mean(axis=1)
    draw_order = np.argsort(face_depth)[::-1]

    z_buffer = np.full((height, width), np.inf, dtype=np.float32)
    for face_index in draw_order:
        face = faces[face_index]
        points = projected[face]
        if not np.all(np.isfinite(points)):
            continue
        _rasterize_triangle(image, z_buffer, points, face_colors[face_index])

    return image
