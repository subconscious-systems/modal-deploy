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
   Serve reads local paths `/models/glm-5.2-nvfp4` and
   `/models/glm-5.2-fp8-dflash-v2` — no Hub fetch at GPU startup.
3. `uv run modal deploy deploy.py` -> Modal pulls the serve image (auth via the
   Docker Hub secret), mounts the weights + warm-cache volumes, and runs
   the sglang launch server on 4x B200.
   `uv run modal serve deploy.py` does the same but hot-reloads for dev.
"""
import subprocess
import sys

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
# ^ Modal Secret (name: SUBCONSCIOUS_HF_TOKEN) with key HF_TOKEN. Weights are
#   pre-downloaded onto the Volume by scripts/download_weights.py; the serve
#   container reads them from local paths under /models.

# Volumes
WEIGHTS_VOLUME = "glm_weights_vol"      # main + draft checkpoints under /models
WEIGHTS_MOUNT = "/models"
CACHE_VOLUME = "tim_cache_vol"          # runtime caches (not model weights)
CACHE_MOUNT = "/mnt/sgl-warm-cache"     # empty path; Modal cannot overlay a non-empty dir
IMAGE_WARM_CACHE = "/opt/sgl-warm-cache"  # baked kernel cache in the Hub image; leave it in place

# Model settings — local paths on WEIGHTS_VOLUME (not Hugging Face repo ids).
MODEL_PATH = "/models/glm-5.2-nvfp4"
DRAFT_MODEL_PATH = "/models/glm-5.2-fp8-dflash-v2"
CHAT_TEMPLATE = "/sgl-workspace/sglang/deploy/chat_templates/glm5.2.jinja"
APP_NAME = "glm-5-2-marathon"

# Compute — 4x B200 node, host RAM for the hierarchical KV-cache offload tier.
GPU = "B200:4"                     # 4x B200 = 768 GB VRAM. VRAM is fixed by type+count, not tunable.
CPU = 64.0                         # Modal hard cap: 64 physical cores
MEMORY_MIB = 1_572_864            # 1.5 TiB; a bit under Modal's 1,650,688 MiB (~1.575 TiB) hard cap
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
            # Use the image's baked kernel cache in place. Do not copy it onto the
            # Volume (that is a silent multi-GB network copy and looks like a hang).
            "SGLANG_CACHE_DIR": f"{IMAGE_WARM_CACHE}/sglang",
            "SGLANG_DG_CACHE_DIR": f"{IMAGE_WARM_CACHE}/deep_gemm",
            "FLASHINFER_WORKSPACE_BASE": f"{IMAGE_WARM_CACHE}/flashinfer",
            # Draft model download persists on the volume; /tmp is ephemeral on Modal.
            "HF_HOME": f"{CACHE_MOUNT}/hf",
            # Unbuffered so sglang logs show up during the long weight-load.
            "PYTHONUNBUFFERED": "1",
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
@modal.concurrent(max_inputs=96)
@modal.web_server(port=PORT, startup_timeout=STARTUP_TIMEOUT, label=APP_NAME)
def serve():
    """Start sglang and leave it listening on PORT.

    `@modal.web_server` runs this function at container start, then proxies
    HTTP to `0.0.0.0:PORT`. The process must be spawned here (returning a
    command string does nothing). Endpoint: `/v1/chat/completions`.
    """
    cmd = [
        "python3", "-u", "-m", "sglang.launch_server",
        "--port", str(PORT), "--host", "0.0.0.0", "--tp", str(TP),
        "--model-path", MODEL_PATH,
        "--chat-template", CHAT_TEMPLATE,
        "--tool-call-parser", "glm47",
        "--subconscious-x-mode",
        "--mem-fraction-static", "0.82",
        "--enable-hierarchical-cache", "--hicache-ratio", "8",
        "--hicache-io-backend", "direct",
        "--hicache-mem-layout", "page_first_direct",
        "--hicache-write-policy", "write_back",
        "--subconscious-x-st-buffer-size", "5",
        "--subconscious-x-min-span-length", "3",
        "--speculative-algorithm", "DFLASH",
        "--speculative-draft-model-path", DRAFT_MODEL_PATH,
        "--speculative-num-draft-tokens", "12",
        "--speculative-draft-kv-cache-dtype", "bfloat16",
        "--speculative-draft-attention-backend", "fa4",
        "--subconscious-leaf-only",
        "--stream-response-default-include-usage",
        "--trust-remote-code", "--enable-cache-report",
        "--cuda-graph-max-bs", "96",
        "--cuda-graph-backend-prefill", "disabled",
        "--max-running-requests", "96",
    ]
    print("[serve] launching:", " ".join(cmd), flush=True)
    subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr)
