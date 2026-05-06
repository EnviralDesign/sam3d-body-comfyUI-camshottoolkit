import os
import tempfile
import torch
import numpy as np
import cv2
from ..runtime_deps import ensure_runtime_dependencies

# =============================================================================
# Helper functions (inlined to avoid relative import issues in worker)
# =============================================================================

def comfy_image_to_numpy(image):
    """Convert ComfyUI image tensor [B,H,W,C] to numpy BGR [H,W,C] for OpenCV."""
    img_np = image[0].cpu().numpy()
    img_np = (img_np * 255).astype(np.uint8)
    return img_np[..., ::-1].copy()  # RGB -> BGR


def comfy_mask_to_numpy(mask):
    """Convert ComfyUI mask tensor [N,H,W] to numpy [N,H,W]."""
    return mask.cpu().numpy()


def numpy_to_comfy_image(np_image):
    """Convert numpy BGR [H,W,C] to ComfyUI image tensor [1,H,W,C]."""
    img_rgb = np_image[..., ::-1].copy()  # BGR -> RGB
    img_rgb = img_rgb.astype(np.float32) / 255.0
    return torch.from_numpy(img_rgb).unsqueeze(0)


def _bbox_from_binary_mask(mask):
    rows = np.any(mask > 0.5, axis=1)
    cols = np.any(mask > 0.5, axis=0)

    if not rows.any() or not cols.any():
        return None

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [cmin, rmin, cmax, rmax]


def _split_mask_components(mask):
    binary = (mask > 0.5).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    components = []
    image_area = max(mask.shape[0] * mask.shape[1], 1)
    min_area = max(64, int(image_area * 0.0005))

    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        component = (labels == label).astype(np.float32)
        bbox = _bbox_from_binary_mask(component)
        if bbox is not None:
            components.append((component, bbox))

    return components


def _prepare_masks_and_bboxes(mask_np):
    masks = np.asarray(mask_np, dtype=np.float32)
    if masks.ndim == 2:
        components = _split_mask_components(masks)
        if not components:
            bbox = _bbox_from_binary_mask(masks)
            if bbox is None:
                return None, None
            components = [(masks, bbox)]
        out_masks = np.stack([component[0] for component in components], axis=0)
        bboxes = np.asarray([component[1] for component in components], dtype=np.float32)
        return out_masks, bboxes

    if masks.ndim == 3:
        out_masks = []
        bboxes = []
        for item in masks:
            bbox = _bbox_from_binary_mask(item)
            if bbox is None:
                continue
            out_masks.append(item)
            bboxes.append(bbox)
        if not out_masks:
            return None, None
        return np.stack(out_masks, axis=0), np.asarray(bboxes, dtype=np.float32)

    return None, None

# Module-level cache for loaded model (persists across calls in worker)
_MODEL_CACHE = {}
_DETECTOR_CACHE = {}
_PERSON_DETECTOR_CACHE = {}


def _load_sam3d_model(model_config: dict):
    """
    Load SAM 3D Body model from config paths.

    Uses module-level caching to avoid reloading on every call.
    This runs inside the isolated worker subprocess.
    """
    cache_key = model_config["ckpt_path"]

    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    ensure_runtime_dependencies("Cam Shot Toolkit: Process Image")

    # Import heavy dependencies only inside worker
    from ..sam_3d_body import load_sam_3d_body

    ckpt_path = model_config["ckpt_path"]
    device = model_config["device"]
    mhr_path = model_config.get("mhr_path", "")

    # Load model using the library's built-in function
    print(f"[SAM3DBody] Loading model from {ckpt_path}...")
    sam_3d_model, model_cfg, _ = load_sam_3d_body(
        checkpoint_path=ckpt_path,
        device=device,
        mhr_path=mhr_path,
    )

    print(f"[SAM3DBody] Model loaded successfully on {device}")

    # Cache for reuse
    result = {
        "model": sam_3d_model,
        "model_cfg": model_cfg,
        "device": device,
        "mhr_path": mhr_path,
    }
    _MODEL_CACHE[cache_key] = result

    return result


def _load_torchvision_person_detector(device):
    cache_key = str(device)
    if cache_key in _DETECTOR_CACHE:
        return _DETECTOR_CACHE[cache_key]

    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_320_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_320_fpn,
    )

    weights = FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
    detector = fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
    detector.eval().to(device)
    _DETECTOR_CACHE[cache_key] = detector
    return detector


def _detect_people_with_torchvision(img_bgr, bbox_threshold, device):
    detector = _load_torchvision_person_detector(device)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        result = detector([image_tensor])[0]

    labels = result["labels"].detach().cpu().numpy()
    scores = result["scores"].detach().cpu().numpy()
    boxes = result["boxes"].detach().cpu().numpy()
    keep = (labels == 1) & (scores >= float(bbox_threshold))
    boxes = boxes[keep]
    scores = scores[keep]

    if len(boxes) == 0:
        return None

    order = np.argsort(boxes[:, 0])
    boxes = boxes[order].astype(np.float32)
    scores = scores[order].astype(np.float32)
    print(
        "[SAM3DBody] Torchvision person detector found "
        f"{len(boxes)} person box(es): "
        + ", ".join(f"{score:.2f}" for score in scores)
    )
    return boxes


def _pad_boxes(boxes, width, height, scale):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scale = max(float(scale), 1.0)
    if scale <= 1.0001:
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, width)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, height)
        return boxes.astype(np.float32)

    padded = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w = (x2 - x1) * scale
        h = (y2 - y1) * scale
        padded.append([
            max(cx - w * 0.5, 0.0),
            max(cy - h * 0.5, 0.0),
            min(cx + w * 0.5, float(width)),
            min(cy + h * 0.5, float(height)),
        ])
    return np.asarray(padded, dtype=np.float32)


def _sort_boxes_left_to_right(boxes, scores=None):
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if len(boxes) == 0:
        return boxes, scores
    order = np.lexsort((boxes[:, 1], boxes[:, 0]))
    boxes = boxes[order]
    if scores is not None:
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)[order]
    return boxes, scores


def _load_transformers_sam3_detector(config, device):
    key = ("transformers_sam3", config.get("model_path"), str(device))
    if key in _PERSON_DETECTOR_CACHE:
        return _PERSON_DETECTOR_CACHE[key]

    try:
        from transformers import Sam3Model, Sam3Processor
    except Exception as exc:
        raise RuntimeError(
            "Transformers SAM3 support is not available in this environment. "
            "Install/upgrade transformers and restart ComfyUI, or use the torchvision fallback detector. "
            f"Import error: {exc}"
        ) from exc

    model_path = config.get("model_path") or config.get("repo_id")
    if not model_path:
        raise RuntimeError("SAM3 detector config is missing model_path/repo_id")

    processor = Sam3Processor.from_pretrained(model_path)
    model = Sam3Model.from_pretrained(model_path).to(device)
    model.eval()
    loaded = {"model": model, "processor": processor}
    _PERSON_DETECTOR_CACHE[key] = loaded
    return loaded


def _load_native_sam3_detector(config, device):
    key = ("native_sam3", config.get("checkpoint_path"), str(device))
    if key in _PERSON_DETECTOR_CACHE:
        return _PERSON_DETECTOR_CACHE[key]

    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor
    except Exception as exc:
        raise RuntimeError(
            "native_sam3 detector was selected, but Meta's sam3 package is not installed. "
            "Use the default transformers_sam3 detector unless you intentionally installed sam3."
        ) from exc

    checkpoint_path = config.get("checkpoint_path")
    if not checkpoint_path:
        raise RuntimeError("native_sam3 detector config is missing checkpoint_path")

    model = build_sam3_image_model(
        checkpoint_path=checkpoint_path,
        load_from_HF=False,
        device=str(device),
    )
    processor = Sam3Processor(model)
    loaded = {"model": model, "processor": processor}
    _PERSON_DETECTOR_CACHE[key] = loaded
    return loaded


def _detect_people_with_person_detector(img_bgr, detector_config, bbox_threshold, device):
    if not detector_config:
        return None

    implementation = detector_config.get("implementation", "torchvision")
    height, width = img_bgr.shape[:2]

    if implementation == "torchvision":
        boxes = _detect_people_with_torchvision(img_bgr, bbox_threshold, device)
        if boxes is None:
            return None
        boxes = _pad_boxes(boxes, width, height, detector_config.get("bbox_padding", 1.0))
        return boxes

    from PIL import Image

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_rgb.astype(np.uint8), "RGB")
    prompt = detector_config.get("prompt") or "person"
    mask_threshold = float(detector_config.get("mask_threshold", 0.5))
    padding = float(detector_config.get("bbox_padding", 1.2))

    if implementation == "native_sam3":
        loaded = _load_native_sam3_detector(detector_config, device)
        processor = loaded["processor"]
        with torch.no_grad():
            inference_state = processor.set_image(image)
            output = processor.set_text_prompt(state=inference_state, prompt=prompt)
        scores = output["scores"].detach().cpu().numpy()
        boxes = output["boxes"].detach().cpu().numpy()
    elif implementation == "transformers_sam3":
        loaded = _load_transformers_sam3_detector(detector_config, device)
        model = loaded["model"]
        processor = loaded["processor"]
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        processed = processor.post_process_instance_segmentation(
            outputs,
            threshold=float(bbox_threshold),
            mask_threshold=mask_threshold,
            target_sizes=(
                inputs.get("original_sizes").tolist()
                if inputs.get("original_sizes") is not None
                else [[height, width]]
            ),
        )[0]
        boxes = processed.get("boxes")
        scores = processed.get("scores")
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.detach().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.detach().cpu().numpy()
    else:
        raise RuntimeError(f"Unsupported person detector implementation: {implementation}")

    if boxes is None:
        return None
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    if scores is None:
        scores = np.ones((len(boxes),), dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    keep = scores >= float(bbox_threshold)
    boxes = boxes[keep]
    scores = scores[keep]
    if len(boxes) == 0:
        return None

    boxes = _pad_boxes(boxes, width, height, padding)
    boxes, scores = _sort_boxes_left_to_right(boxes, scores)
    print(
        f"[SAM3DBody] {implementation} detector found {len(boxes)} person box(es): "
        + ", ".join(f"{score:.2f}" for score in scores)
    )
    return boxes


def _coerce_person_selection(person_index, people_count):
    try:
        requested_index = int(person_index)
    except Exception:
        requested_index = 0

    if requested_index == -1:
        return requested_index, list(range(people_count))

    selected_index = max(0, min(requested_index, people_count - 1))
    return requested_index, [selected_index]


def _build_person_record(output, faces, mhr_path, source_index):
    return {
        "person_index": int(source_index),
        "vertices": output.get("pred_vertices", None),
        "faces": faces,
        "joints": output.get("pred_keypoints_3d", None),
        "joint_coords": output.get("pred_joint_coords", None),
        "joint_rotations": output.get("pred_global_rots", None),
        "camera": output.get("pred_cam_t", None),
        "focal_length": output.get("focal_length", None),
        "bbox": output.get("bbox", None),
        "pose_params": {
            "body_pose": output.get("body_pose_params", None),
            "hand_pose": output.get("hand_pose_params", None),
            "global_rot": output.get("global_rot", None),
            "shape": output.get("shape_params", None),
            "scale": output.get("scale_params", None),
            "expr": output.get("expr_params", None),
        },
        "raw_output": output,
        "mhr_path": mhr_path,
    }


def _build_skeleton(output):
    return {
        "joint_positions": output.get("pred_joint_coords", None),
        "joint_rotations": output.get("pred_global_rots", None),
        "pose_params": output.get("body_pose_params", None),
        "shape_params": output.get("shape_params", None),
        "scale_params": output.get("scale_params", None),
        "hand_pose": output.get("hand_pose_params", None),
        "global_rot": output.get("global_rot", None),
        "expr_params": output.get("expr_params", None),
        "camera": output.get("pred_cam_t", None),
        "focal_length": output.get("focal_length", None),
    }


def _add_joint_parent_hierarchy(skeleton, sam_3d_model):
    try:
        if hasattr(sam_3d_model, 'mhr_head') and hasattr(sam_3d_model.mhr_head, 'mhr'):
            mhr = sam_3d_model.mhr_head.mhr
            if hasattr(mhr, 'character_torch') and hasattr(mhr.character_torch, 'skeleton'):
                skeleton_obj = mhr.character_torch.skeleton
                if hasattr(skeleton_obj, 'joint_parents'):
                    parent_tensor = skeleton_obj.joint_parents
                    if isinstance(parent_tensor, torch.Tensor):
                        skeleton["joint_parents"] = parent_tensor.cpu().numpy()
    except Exception:
        pass


def _build_mesh_and_skeleton(outputs, faces, loaded, sam_3d_model, person_index):
    requested_index, selected_indices = _coerce_person_selection(person_index, len(outputs))
    mhr_path = loaded.get("mhr_path", None)
    people = [
        _build_person_record(outputs[source_index], faces, mhr_path, source_index)
        for source_index in selected_indices
    ]
    primary = dict(people[0])
    primary.update({
        "people": people,
        "people_count": len(outputs),
        "selected_people_count": len(people),
        "selected_person_indices": selected_indices,
        "selected_primary_index": selected_indices[0],
        "requested_person_index": requested_index,
        "person_index": requested_index if requested_index == -1 else selected_indices[0],
        "selection_mode": "all" if requested_index == -1 else "single",
        "all_people": outputs,
    })

    skeleton = _build_skeleton(outputs[selected_indices[0]])
    skeleton.update({
        "people_count": len(outputs),
        "selected_people_count": len(people),
        "selected_person_indices": selected_indices,
        "selected_primary_index": selected_indices[0],
        "requested_person_index": requested_index,
        "person_index": requested_index if requested_index == -1 else selected_indices[0],
        "selection_mode": "all" if requested_index == -1 else "single",
    })
    _add_joint_parent_hierarchy(skeleton, sam_3d_model)
    return primary, skeleton


def _requested_person_index(person_index):
    try:
        return int(person_index)
    except Exception:
        return 0

class SAM3DBodyProcess:
    """
    Performs 3D human mesh reconstruction from a single image.

    Takes an input image and outputs 3D mesh data including vertices, faces,
    pose parameters, and camera parameters.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAM3D_MODEL", {
                    "tooltip": "Loaded SAM 3D Body model from Load node"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image containing human subject"
                }),
                "bbox_threshold": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Confidence threshold for human detection bounding boxes"
                }),
                "inference_type": (["full", "body", "hand"], {
                    "default": "full",
                    "tooltip": "full: body+hand decoders, body: body decoder only, hand: hand decoder only"
                }),
                "person_index": ("INT", {
                    "default": 0,
                    "min": -1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "-1 selects all detected people. 0..N selects one detected person by index."
                }),
            },
            "optional": {
                "mask": ("MASK", {
                    "tooltip": "Optional segmentation mask to guide reconstruction"
                }),
                "person_detector": ("SAM3D_PERSON_DETECTOR", {
                    "tooltip": "Optional SAM3 person detector for multi-person images"
                }),
            }
        }

    RETURN_TYPES = ("SAM3D_OUTPUT", "SKELETON", "IMAGE")
    RETURN_NAMES = ("mesh_data", "skeleton", "debug_image")
    FUNCTION = "process"
    CATEGORY = "CamShotToolkit/processing"

    def _compute_bbox_from_mask(self, mask):
        """Compute bounding box from binary mask."""
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)

        if not rows.any() or not cols.any():
            return None

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return np.array([[cmin, rmin, cmax, rmax]], dtype=np.float32)

    def process(self, model, image, bbox_threshold=0.8, inference_type="full", person_index=0, mask=None, person_detector=None):
        """Process image and reconstruct 3D human mesh.

        Args:
            model: Config dict from LoadSAM3DBodyModel with paths (model loaded lazily here)
            image: Input image tensor
            bbox_threshold: Detection confidence threshold
            inference_type: "full", "body", or "hand"
            person_index: -1 for all detected people, otherwise a single 0-based person index
            mask: Optional segmentation mask
            person_detector: Optional detector config from Load SAM3 Person Detector
        """
        ensure_runtime_dependencies("Cam Shot Toolkit: Process Image")
        from ..sam_3d_body import SAM3DBodyEstimator

        # Lazy load model (cached after first call)
        loaded = _load_sam3d_model(model)
        sam_3d_model = loaded["model"]
        model_cfg = loaded["model_cfg"]

        # Create estimator
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=sam_3d_model,
            model_cfg=model_cfg,
            human_detector=None,
            human_segmentor=None,
            fov_estimator=None,
        )

        # Convert ComfyUI image to numpy (BGR format for OpenCV)
        img_bgr = comfy_image_to_numpy(image)

        # Convert mask if provided and compute bounding box
        mask_np = None
        bboxes = None
        detector_error = None
        requested_index = _requested_person_index(person_index)
        if person_detector is not None:
            try:
                bboxes = _detect_people_with_person_detector(
                    img_bgr,
                    person_detector,
                    bbox_threshold,
                    torch.device(loaded["device"]),
                )
            except Exception as exc:
                detector_error = exc
                print(f"[SAM3DBody] [ERROR] Person detector failed: {exc}")
                bboxes = None
            if bboxes is None and requested_index != 0:
                if detector_error is not None:
                    raise RuntimeError(
                        "Person detector failed before returning person boxes. "
                        f"{detector_error}"
                    ) from detector_error
                raise RuntimeError(
                    "Person detector did not return any person boxes. "
                    "Try lowering bbox_threshold or disconnect the detector to use the full-image fallback."
                )

        if bboxes is None and mask is not None:
            mask_np = comfy_mask_to_numpy(mask)
            mask_np, bboxes = _prepare_masks_and_bboxes(mask_np)

        if bboxes is None and requested_index != 0:
            try:
                bboxes = _detect_people_with_torchvision(
                    img_bgr,
                    bbox_threshold,
                    torch.device(loaded["device"]),
                )
            except Exception as exc:
                print(f"[SAM3DBody] [WARNING] Automatic person detection failed: {exc}")
                bboxes = None

        # Save image to temporary file (required by SAM3DBodyEstimator)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, img_bgr)
            tmp_path = tmp.name

        try:
            outputs = estimator.process_one_image(
                tmp_path,
                bboxes=bboxes,
                masks=mask_np,
                bbox_thr=bbox_threshold,
                use_mask=(mask is not None),
                inference_type=inference_type,
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if not outputs or len(outputs) == 0:
            raise RuntimeError("No people detected in image")

        mesh_data, skeleton = _build_mesh_and_skeleton(
            outputs,
            estimator.faces,
            loaded,
            sam_3d_model,
            person_index,
        )

        # Create debug visualization
        debug_img = self._create_debug_visualization(img_bgr, outputs, estimator.faces)
        debug_img_comfy = numpy_to_comfy_image(debug_img)

        return (mesh_data, skeleton, debug_img_comfy)

    def _create_debug_visualization(self, img_bgr, outputs, faces):
        """Create a debug visualization of the results."""
        debug = img_bgr.copy()
        for index, output in enumerate(outputs):
            bbox = output.get("bbox", None)
            if bbox is None:
                continue
            box = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cv2.putText(
                debug,
                str(index),
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
        return debug

class SAM3DBodyProcessAdvanced:
    """
    Advanced processing node with full control over detection, segmentation, and FOV estimation.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SAM3D_MODEL", {
                    "tooltip": "Loaded SAM 3D Body model from Load node"
                }),
                "image": ("IMAGE", {
                    "tooltip": "Input image containing human subject"
                }),
                "bbox_threshold": ("FLOAT", {
                    "default": 0.8,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Confidence threshold for human detection"
                }),
                "nms_threshold": ("FLOAT", {
                    "default": 0.3,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Non-maximum suppression threshold for detection"
                }),
                "inference_type": (["full", "body", "hand"], {
                    "default": "full",
                    "tooltip": "Inference mode: full (body+hand), body only, or hand only"
                }),
                "person_index": ("INT", {
                    "default": 0,
                    "min": -1,
                    "max": 100,
                    "step": 1,
                    "tooltip": "-1 selects all detected people. 0..N selects one detected person by index."
                }),
                "detector_name": (["none", "vitdet"], {
                    "default": "none",
                    "tooltip": "Human detector to use (requires detector_path)"
                }),
                "segmentor_name": (["none", "sam2"], {
                    "default": "none",
                    "tooltip": "Segmentation model to use (requires segmentor_path)"
                }),
                "fov_name": (["none", "moge2"], {
                    "default": "none",
                    "tooltip": "FOV estimator to use (requires fov_path)"
                }),
            },
            "optional": {
                "detector_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to detector model or set SAM3D_DETECTOR_PATH env var"
                }),
                "segmentor_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to segmentor model or set SAM3D_SEGMENTOR_PATH env var"
                }),
                "fov_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to FOV model or set SAM3D_FOV_PATH env var"
                }),
                "mask": ("MASK", {
                    "tooltip": "Optional pre-computed segmentation mask"
                }),
                "person_detector": ("SAM3D_PERSON_DETECTOR", {
                    "tooltip": "Optional SAM3 person detector for multi-person images"
                }),
            }
        }

    RETURN_TYPES = ("SAM3D_OUTPUT", "SKELETON", "IMAGE")
    RETURN_NAMES = ("mesh_data", "skeleton", "debug_image")
    FUNCTION = "process_advanced"
    CATEGORY = "SAM3DBody/advanced"

    def _compute_bbox_from_mask(self, mask):
        """Compute bounding box from binary mask."""
        rows = np.any(mask > 0.5, axis=1)
        cols = np.any(mask > 0.5, axis=0)

        if not rows.any() or not cols.any():
            return None

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return np.array([[cmin, rmin, cmax, rmax]], dtype=np.float32)

    def process_advanced(self, model, image, bbox_threshold=0.8, nms_threshold=0.3,
                        inference_type="full", person_index=0, detector_name="none", segmentor_name="none",
                        fov_name="none", detector_path="", segmentor_path="", fov_path="", mask=None, person_detector=None):
        """Process image with advanced options.

        Args:
            model: Config dict from LoadSAM3DBodyModel with paths (model loaded lazily here)
            image: Input image tensor
            bbox_threshold: Detection confidence threshold
            nms_threshold: Non-maximum suppression threshold
            inference_type: "full", "body", or "hand"
            person_index: -1 for all detected people, otherwise a single 0-based person index
            detector_name: Human detector to use
            segmentor_name: Segmentation model to use
            fov_name: FOV estimator to use
            detector_path: Path to detector model
            segmentor_path: Path to segmentor model
            fov_path: Path to FOV model
            mask: Optional pre-computed segmentation mask
            person_detector: Optional detector config from Load SAM3 Person Detector
        """
        from ..sam_3d_body import SAM3DBodyEstimator

        # Lazy load model (cached after first call)
        loaded = _load_sam3d_model(model)
        sam_3d_model = loaded["model"]
        model_cfg = loaded["model_cfg"]
        device = torch.device(loaded["device"])

        # Initialize optional components
        detector = None
        segmentor = None
        fov_estimator = None

        # Load detector if specified
        if detector_name != "none":
            detector_path = detector_path or os.environ.get("SAM3D_DETECTOR_PATH", "")
            if detector_path:
                from tools.build_detector import HumanDetector
                detector = HumanDetector(name=detector_name, device=device, path=detector_path)

        # Load segmentor if specified
        if segmentor_name != "none":
            segmentor_path = segmentor_path or os.environ.get("SAM3D_SEGMENTOR_PATH", "")
            if segmentor_path:
                from tools.build_sam import HumanSegmentor
                segmentor = HumanSegmentor(name=segmentor_name, device=device, path=segmentor_path)

        # Load FOV estimator if specified
        if fov_name != "none":
            fov_path = fov_path or os.environ.get("SAM3D_FOV_PATH", "")
            if fov_path:
                from tools.build_fov_estimator import FOVEstimator
                fov_estimator = FOVEstimator(name=fov_name, device=device, path=fov_path)

        # Create estimator with optional components
        estimator = SAM3DBodyEstimator(
            sam_3d_body_model=sam_3d_model,
            model_cfg=model_cfg,
            human_detector=detector,
            human_segmentor=segmentor,
            fov_estimator=fov_estimator,
        )

        # Convert image and mask
        img_bgr = comfy_image_to_numpy(image)
        mask_np = None
        bboxes = None
        detector_error = None
        requested_index = _requested_person_index(person_index)
        if person_detector is not None:
            try:
                bboxes = _detect_people_with_person_detector(
                    img_bgr,
                    person_detector,
                    bbox_threshold,
                    device,
                )
            except Exception as exc:
                detector_error = exc
                print(f"[SAM3DBody] [ERROR] Person detector failed: {exc}")
                bboxes = None
            if bboxes is None and requested_index != 0:
                if detector_error is not None:
                    raise RuntimeError(
                        "Person detector failed before returning person boxes. "
                        f"{detector_error}"
                    ) from detector_error
                raise RuntimeError(
                    "Person detector did not return any person boxes. "
                    "Try lowering bbox_threshold or disconnect the detector to use the full-image fallback."
                )

        if bboxes is None and mask is not None:
            mask_np = comfy_mask_to_numpy(mask)
            mask_np, bboxes = _prepare_masks_and_bboxes(mask_np)

        if bboxes is None and requested_index != 0:
            try:
                bboxes = _detect_people_with_torchvision(
                    img_bgr,
                    bbox_threshold,
                    device,
                )
            except Exception as exc:
                print(f"[SAM3DBody] [WARNING] Automatic person detection failed: {exc}")
                bboxes = None

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            cv2.imwrite(tmp.name, img_bgr)
            tmp_path = tmp.name

        try:
            outputs = estimator.process_one_image(
                tmp_path,
                bboxes=bboxes,
                masks=mask_np,
                bbox_thr=bbox_threshold,
                nms_thr=nms_threshold,
                use_mask=(mask is not None or segmentor is not None),
                inference_type=inference_type,
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

        if not outputs or len(outputs) == 0:
            raise RuntimeError("No people detected in image")

        mesh_data, skeleton = _build_mesh_and_skeleton(
            outputs,
            estimator.faces,
            loaded,
            sam_3d_model,
            person_index,
        )

        # Create debug visualization
        debug_img = self._create_debug_visualization(img_bgr, outputs, estimator.faces)
        debug_img_comfy = numpy_to_comfy_image(debug_img)

        return (mesh_data, skeleton, debug_img_comfy)

    def _create_debug_visualization(self, img_bgr, outputs, faces):
        """Create debug visualization."""
        debug = img_bgr.copy()
        for index, output in enumerate(outputs):
            bbox = output.get("bbox", None)
            if bbox is None:
                continue
            box = np.asarray(bbox, dtype=np.float32).reshape(-1)[:4]
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 220, 255), 2)
            cv2.putText(
                debug,
                str(index),
                (x1, max(0, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )
        return debug


# Register nodes
NODE_CLASS_MAPPINGS = {
    "CamShotToolkitSAM3DBodyProcess": SAM3DBodyProcess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CamShotToolkitSAM3DBodyProcess": "Cam Shot Toolkit: Process Image",
}
