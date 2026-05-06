import os
import folder_paths
from ..runtime_deps import ensure_runtime_dependencies

# Default model path in ComfyUI models folder
DEFAULT_MODEL_PATH = os.path.join(folder_paths.models_dir, "sam3dbody")
DEFAULT_DETECTOR_PATH = os.path.join(folder_paths.models_dir, "sam3_person_detector")


def _safe_repo_dir(repo_id):
    return repo_id.replace("/", "--").replace("\\", "--").replace(":", "_")


def _download_detector_artifacts(
    repo_id,
    implementation,
    checkpoint_filename,
    local_base_dir,
):
    from huggingface_hub import hf_hub_download, snapshot_download

    local_dir = os.path.join(local_base_dir, _safe_repo_dir(repo_id))
    os.makedirs(local_dir, exist_ok=True)

    if implementation == "native_sam3":
        checkpoint_path = hf_hub_download(
            repo_id=repo_id,
            filename=checkpoint_filename,
            local_dir=local_dir,
        )
        return local_dir, checkpoint_path

    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        allow_patterns=[
            "config.json",
            "model.safetensors",
            "processor_config.json",
            "special_tokens_map.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
        ],
    )
    return local_dir, None


class LoadSAM3DBodyModel:
    """
    Prepares SAM 3D Body model configuration.

    Returns a config dict with model paths. The actual model is loaded
    lazily inside the isolated worker when inference runs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ("SAM3D_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "CamShotToolkit"

    def load_model(self):
        """Prepare model config (actual loading happens in inference nodes)."""
        ensure_runtime_dependencies("Cam Shot Toolkit: Load SAM3D Model")
        import torch

        # Auto-detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Always derive the model path from the local ComfyUI install to keep
        # workflows portable across machines.
        model_path = os.path.abspath(DEFAULT_MODEL_PATH)
        print(f"[SAM3DBody] Using model path: {model_path}")

        # Expected file paths
        ckpt_path = os.path.join(model_path, "model.ckpt")
        mhr_path = os.path.join(model_path, "assets", "mhr_model.pt")

        # Check if model exists locally, download if not
        model_exists = os.path.exists(ckpt_path) and os.path.exists(mhr_path)

        if not model_exists:
            try:
                from huggingface_hub import snapshot_download

                print(f"[SAM3DBody] Model not found locally. Downloading from HuggingFace...")
                os.makedirs(model_path, exist_ok=True)
                snapshot_download(
                    repo_id="jetjodh/sam-3d-body-dinov3",
                    local_dir=model_path
                )
                print(f"[SAM3DBody] Download complete.")

            except Exception as e:
                raise RuntimeError(
                    f"\n[SAM3DBody] Download failed.\n\n"
                    f"Please manually download from:\n"
                    f"  https://huggingface.co/jetjodh/sam-3d-body-dinov3\n\n"
                    f"And place the model files at:\n"
                    f"  {DEFAULT_MODEL_PATH}/\n"
                    f"    +-- model.ckpt          (SAM 3D Body checkpoint)\n"
                    f"    +-- model_config.yaml   (model configuration)\n"
                    f"    \\-- assets/\n"
                    f"        \\-- mhr_model.pt    (Momentum Human Rig model)\n\n"
                    f"Download error: {e}"
                ) from e

        # Return config dict (not the actual model)
        model_config = {
            "model_path": model_path,
            "ckpt_path": ckpt_path,
            "mhr_path": mhr_path,
            "device": device,
        }

        return (model_config,)


class LoadSAM3PersonDetector:
    """
    Prepares a person detector for multi-person SAM3D Body reconstruction.

    The detector is loaded lazily during processing. The loader resolves and
    downloads weights so failures are visible before the expensive body pass.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "implementation": (["transformers_sam3", "native_sam3", "torchvision"], {
                    "default": "transformers_sam3",
                    "tooltip": "Use Transformers SAM3 by default. native_sam3 requires Meta's sam3 package to be installed separately."
                }),
                "repo_mode": (["auto", "official", "mirror"], {
                    "default": "auto",
                    "tooltip": "auto tries the official gated repo first, then an ungated mirror if access is unavailable."
                }),
                "prompt": ("STRING", {
                    "default": "person",
                    "tooltip": "Text concept prompt used by SAM3."
                }),
                "fallback_repo_id": ("STRING", {
                    "default": "jetjodh/sam3",
                    "tooltip": "Ungated fallback mirror used when facebook/sam3 is gated or inaccessible."
                }),
                "bbox_padding": ("FLOAT", {
                    "default": 1.2,
                    "min": 1.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Scale detected boxes around their center before SAM3D Body crops them."
                }),
                "mask_threshold": ("FLOAT", {
                    "default": 0.5,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Mask threshold used by Transformers SAM3 post-processing."
                }),
            },
            "optional": {
                "official_repo_id": ("STRING", {
                    "default": "facebook/sam3",
                    "tooltip": "Official SAM3 repo. Uses your Hugging Face login/cache or HF_TOKEN if required."
                }),
                "checkpoint_filename": ("STRING", {
                    "default": "sam3.pt",
                    "tooltip": "Checkpoint file used only by native_sam3."
                }),
            }
        }

    RETURN_TYPES = ("SAM3D_PERSON_DETECTOR",)
    RETURN_NAMES = ("person_detector",)
    FUNCTION = "load_detector"
    CATEGORY = "CamShotToolkit"

    def load_detector(
        self,
        implementation="transformers_sam3",
        repo_mode="auto",
        prompt="person",
        fallback_repo_id="jetjodh/sam3",
        bbox_padding=1.2,
        mask_threshold=0.5,
        official_repo_id="facebook/sam3",
        checkpoint_filename="sam3.pt",
    ):
        ensure_runtime_dependencies("Cam Shot Toolkit: Load SAM3 Person Detector")

        if implementation == "torchvision":
            return ({
                "implementation": "torchvision",
                "prompt": prompt,
                "bbox_padding": float(bbox_padding),
                "mask_threshold": float(mask_threshold),
            },)

        if implementation == "native_sam3":
            try:
                import sam3  # noqa: F401
            except Exception as exc:
                raise RuntimeError(
                    "[SAM3DBody] native_sam3 was selected, but Meta's sam3 package is not installed.\n"
                    "Install it manually if you specifically need the native implementation, or use transformers_sam3.\n"
                    "The Transformers implementation is preferred for Comfy because Meta's native package currently pins numpy<2.\n\n"
                    f"Import error: {exc}"
                ) from exc

        repo_candidates = []
        if repo_mode in ("auto", "official"):
            repo_candidates.append(official_repo_id.strip() or "facebook/sam3")
        if repo_mode in ("auto", "mirror"):
            repo_candidates.append(fallback_repo_id.strip() or "jetjodh/sam3")

        last_error = None
        for repo_id in repo_candidates:
            try:
                print(f"[SAM3DBody] Resolving SAM3 person detector from {repo_id}...")
                model_path, checkpoint_path = _download_detector_artifacts(
                    repo_id=repo_id,
                    implementation=implementation,
                    checkpoint_filename=checkpoint_filename,
                    local_base_dir=os.path.abspath(DEFAULT_DETECTOR_PATH),
                )
                if repo_id != official_repo_id:
                    print(f"[SAM3DBody] Using SAM3 fallback mirror: {repo_id}")
                return ({
                    "implementation": implementation,
                    "repo_id": repo_id,
                    "model_path": model_path,
                    "checkpoint_path": checkpoint_path,
                    "checkpoint_filename": checkpoint_filename,
                    "prompt": prompt,
                    "bbox_padding": float(bbox_padding),
                    "mask_threshold": float(mask_threshold),
                },)
            except Exception as exc:
                last_error = exc
                print(f"[SAM3DBody] Could not use SAM3 detector repo {repo_id}: {exc}")
                if repo_mode != "auto":
                    break

        raise RuntimeError(
            "[SAM3DBody] Could not resolve SAM3 person detector weights.\n\n"
            "If using the official repo, accept the model terms on Hugging Face and make sure HF_TOKEN or huggingface-cli login is available.\n"
            "Otherwise set repo_mode=mirror or choose a known ungated mirror such as jetjodh/sam3.\n\n"
            f"Last error: {last_error}"
        )


# Register node
NODE_CLASS_MAPPINGS = {
    "CamShotToolkitLoadSAM3DBodyModel": LoadSAM3DBodyModel,
    "CamShotToolkitLoadSAM3PersonDetector": LoadSAM3PersonDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CamShotToolkitLoadSAM3DBodyModel": "Cam Shot Toolkit: Load SAM3D Model",
    "CamShotToolkitLoadSAM3PersonDetector": "Cam Shot Toolkit: Load SAM3 Person Detector",
}
