"""
Deploy GLM-5.2-NVFP4 (model name "glm-5.2-marathon") onto Modal 4x B200s.

Flow
----
1. The serve image is the prebuilt sm_100 (Blackwell) sglang image at
   REGISTRY_IMAGE (subconsciouslabs/sglang-baseten:sm_100-v0.10 on Docker Hub).
   It must contain the sglang fork with the --subconscious-* / DFLASH / fa4
   features and the /sgl-workspace checkout (incl. the chat template at
   CHAT_TEMPLATE below). The image is built externally and pushed to Hub.
2. Populate the weights Volume once from Hugging Face:
   `uv run modal run scripts/download_weights.py`
3. `uv run modal deploy deploy.py` -> Modal pulls the serve image (auth via the
   Docker Hub secret), mounts the weights + warm-cache volumes, and runs
   the sglang launch server on 4x B200.
   `uv run modal serve deploy.py` does the same but hot-reloads for dev.
"""
import modal

# ---------------------------------------------------------------------------
# Configuration — edit these to match your Docker Hub image + model.
# ---------------------------------------------------------------------------
REGISTRY_IMAGE = "subconsciouslabs/sglang-baseten:sm_100-v0.10"
REGISTRY_SECRET = "SUBCONSCIOUS_DOCKERHUB"
# ^ Modal Secret (name: SUBCONSCIOUS_DOCKERHUB) with keys REGISTRY_USERNAME
#   (Docker Hub username) + REGISTRY_PASSWORD (Docker Hub access token).
#   Upsert it (idempotently) from .env via:
#   `uv run python scripts/write_secrets.py`

HF_SECRET = "SUBCONSCIOUS_HF_TOKEN"
# ^ Modal Secret (name: SUBCONSCIOUS_HF_TOKEN) with key HF_TOKEN, needed to
#   download the gated DFLASH draft model SubconsciousDev/glm-5.2-fp8-dflash-v2
#   at startup (and by scripts/download_weights.py for the main nvidia/GLM-5.2-NVFP4
#   weights). Upsert from .env via: `uv run python scripts/write_secrets.py`

# Volumes
WEIGHTS_VOLUME = "glm_weights_vol"      # holds the main model at MODEL_PATH
WEIGHTS_MOUNT = "/models"
CACHE_VOLUME = "tim_cache_vol"          # persists compiled kernels / draft model
CACHE_MOUNT = "/opt/sgl-warm-cache"

# Model settings
MODEL_PATH = "/models/glm-5.2-nvfp4"                       # main model on the weights volume
DRAFT_MODEL_PATH = "SubconsciousDev/glm-5.2-fp8-dflash-v2"  # downloaded from HF at startup
CHAT_TEMPLATE = "/sgl-workspace/sglang/deploy/chat_templates/glm5.2.jinja"
APP_NAME = "glm-5-2-marathon"

# Compute — 4x B200 node, max host RAM for the hierarchical KV-cache offload tier.
GPU = "B200:4"                     # 4x B200 = 768 GB VRAM. VRAM is fixed by type+count, not tunable.
CPU = 64.0                         # Modal hard cap: 64 physical cores
MEMORY_MIB = 1_650_688            # Modal hard cap (~1.575 TiB); flat, does NOT scale with GPU count
PORT = 8000
TP = 4                             # tensor-parallel == GPU count
STARTUP_TIMEOUT = 3000            # GLM-5.2 nvfp4 load + cuda-graph(bs=96) build is slow
RUN_TIMEOUT = 86400               # max container lifetime per cold cycle; watch for a Modal cap at deploy

# ---------------------------------------------------------------------------
# Image — pull from the private registry and bake in sglang env vars.
# The base image has python3/pip3 but may not alias them to `python`/`pip`,
# which Modal's runtime needs; fix that with setup_dockerfile_commands.
# ---------------------------------------------------------------------------
image = (
    modal.Image.from_registry(
        REGISTRY_IMAGE,
        secret=modal.Secret.from_name(
            REGISTRY_SECRET,
            required_keys=["REGISTRY_USERNAME", "REGISTRY_PASSWORD"],
        ),
        setup_dockerfile_commands=[
            'RUN ln -sf "$(which python3)" /usr/local/bin/python 2>/dev/null || true',
            'RUN ln -sf "$(which pip3)" /usr/local/bin/pip 2>/dev/null || true',
        ],
    )
    .env(
        {
            # Warm-cache dirs -> persisted on the cache volume for faster cold starts.
            # Baked into the image env so the web_server subprocess inherits them.
            "SGLANG_CACHE_DIR": f"{CACHE_MOUNT}/sglang",
            "SGLANG_DG_CACHE_DIR": f"{CACHE_MOUNT}/deep_gemm",
            "FLASHINFER_WORKSPACE_BASE": f"{CACHE_MOUNT}/flashinfer",
            # HF cache on the volume so the draft model isn't re-downloaded each cold
            # start. /tmp is ephemeral on Modal, so keep HF_HOME on the persistent volume.
            "HF_HOME": f"{CACHE_MOUNT}/hf",
            # sglang fork behavior flags.
            "SGLANG_PP_SPIKE_MODE": "on_drop",
            "SGLANG_SUBCONSCIOUS_TRANSPLANT": "1",
            "SGLANG_SUBCONSCIOUS_BACKTRACK_MAX_LEVELS": "1",
        }
    )
)

app = modal.App(APP_NAME)
weights_vol = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)
cache_vol = modal.Volume.from_name(CACHE_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    cpu=CPU,
    memory=MEMORY_MIB,
    volumes={WEIGHTS_MOUNT: weights_vol, CACHE_MOUNT: cache_vol},
    secrets=[modal.Secret.from_name(HF_SECRET, required_keys=["HF_TOKEN"])],
    scaledown_window=600,
    min_containers=0,
    max_containers=1,
    timeout=RUN_TIMEOUT,
)
@modal.web_server(port=PORT, startup_timeout=STARTUP_TIMEOUT, label=APP_NAME)
def serve():
    """Run the container's native sglang server.

    `@modal.web_server` runs this command as a subprocess, waits for the port
    to come up, and proxies HTTP traffic to it. The endpoint is
    `/v1/chat/completions` (OpenAI-compatible).
    """
    return (
        "python3 -m sglang.launch_server "
        f"--port {PORT} --host 0.0.0.0 --tp {TP} --model-path {MODEL_PATH} "
        f"--chat-template {CHAT_TEMPLATE} "
        "--tool-call-parser glm47 "
        "--subconscious-x-mode "
        "--mem-fraction-static 0.82 "
        "--enable-hierarchical-cache --hicache-ratio 8 --hicache-io-backend direct "
        "--hicache-mem-layout page_first_direct --hicache-write-policy write_back "
        "--subconscious-x-st-buffer-size 5 --subconscious-x-min-span-length 3 "
        f"--speculative-algorithm DFLASH --speculative-draft-model-path {DRAFT_MODEL_PATH} "
        "--speculative-num-draft-tokens 12 --speculative-draft-kv-cache-dtype bfloat16 "
        "--speculative-draft-attention-backend fa4 "
        "--subconscious-leaf-only --stream-response-default-include-usage "
        "--trust-remote-code --enable-cache-report --cuda-graph-max-bs 96 "
        "--cuda-graph-backend-prefill disabled "
        "--max-running-requests 96"
    )
