# modal-exp

Experiment harness for deploying the **GLM-5.2-NVFP4** sglang
container onto Modal **4× B200** GPUs, pulling the image from a private
`distr` registry.

## Layout

```
modal-exp/
├── pyproject.toml          # uv env (Python 3.12, modal CLI)
├── deploy.py               # Modal app: pulls serve image, runs sglang on 4x B200
├── Dockerfile              # (optional/legacy) thin serving layer — not used when pulling the prebuilt distr image
├── .env.example            # template for sensitive values (HF token, distr PAT)
├── scripts/
│   ├── write_secrets.py    # idempotent upsert of .env secrets into Modal
│   ├── download_weights.py # one-time job: pull gated nvidia/GLM-5.2-NVFP4 into a Volume
│   └── test_endpoint.py    # send a sample chat-completion request to the deployed endpoint
└── README.md
```

## One-time setup

```bash
# 1. Sync the uv env (already done once; re-run after pulling)
uv sync

# 2. Authenticate Modal (browser flow). Already authed as dana-12991 in the
#    subconscious-systems workspace; re-run only if you need to redo it.
uv run modal setup            # or: uv run modal token new

# 3. Put your sensitive values in .env (copy from .env.example), then upsert
#    them idempotently into Modal. This creates two Modal secrets:
#      SUBCONSCIOUS_HF_TOKEN   (key: HF_TOKEN)
#      SUBCONSCIOUS_DISTR_PAT  (keys: REGISTRY_USERNAME=- , REGISTRY_PASSWORD=<PAT>)
cp .env.example .env
$EDITOR .env                 # fill in SUBCONSCIOUS_HF_TOKEN and SUBCONSCIOUS_DISTR_PAT
uv run python scripts/write_secrets.py

# 4. Populate the weights Volume once (downloads nvidia/GLM-5.2-NVFP4 at the
#    pinned commit aec724e8...):
uv run modal run scripts/download_weights.py
```

## Deploy
Edit the config block at the top of `deploy.py`:
- `REGISTRY_IMAGE` → your pushed distr image (default `registry.distr.sh/subconscious/timrun:sm_100-v0.10`)
- `REGISTRY_SECRET`, `HF_SECRET` → `SUBCONSCIOUS_DISTR_PAT` / `SUBCONSCIOUS_HF_TOKEN` (upserted by `scripts/write_secrets.py`)
- `WEIGHTS_VOLUME`, `MODEL_PATH` → where the GLM weights live in the Volume
- `DRAFT_MODEL_PATH` → HF repo id of the DFLASH draft model (downloaded at startup)
- `GPU`, `TP`, `CPU`, `MEMORY_MIB`, `PORT`, `STARTUP_TIMEOUT` → compute/serve config

Then:

```bash
uv run modal deploy deploy.py      # production deploy
uv run modal serve  deploy.py      # dev: hot-reload on file change
```

## Test the endpoint

`scripts/test_endpoint.py` sends a basic streaming chat-completion request to
the deployed endpoint and prints the streamed tokens. No extra deps — it uses
`aiohttp` (already pulled in by `modal`).

```bash
# After deploy, copy the URL it prints (or `uv run modal app list`):
uv run python scripts/test_endpoint.py https://<workspace>--glm-5.2-marathon.modal.run
```

## Lifecycle: deploy / serve / stop / etc.

The Modal app name is `APP_NAME` (currently `glm-5.2-marathon`). Most lifecycle
commands take the app id/name shown by `modal app list`.

```bash
# Deploy / update (production). Creates a new deployment; containers persist
# and autoscale min_containers(0) -> max_containers(1) on traffic.
uv run modal deploy deploy.py
uv run modal deploy deploy.py --tag v1     # tag this deployment version
uv run modal deploy deploy.py --strategy rolling   # rolling (default) or recreate

# Serve (dev). Hot-reloads on file change; containers stop when you Ctrl-C.
# Nothing is permanently deployed — it's ephemeral.
uv run modal serve deploy.py

# Inspect
uv run modal app list                     # running, deployed, recently stopped apps
uv run modal app logs glm-5.2-marathon    # stream logs (the app name from deploy.py)
uv run modal app history glm-5.2-marathon # deployment versions
uv run modal app dashboard glm-5.2-marathon  # open the dashboard page

# Stop / roll
uv run modal app stop glm-5.2-marathon     # permanently stop + terminate containers
uv run modal app rollover glm-5.2-marathon # redeploy to get fresh containers, no code change
uv run modal app rollback glm-5.2-marathon # go back to the previous deployment

# Shell into a running container (debug the image / env)
uv run modal shell deploy.py

# Manage the backing storage
uv run modal volume list                  # glm_weights_vol, tim_cache_vol
uv run modal secret list                  # SUBCONSCIOUS_HF_TOKEN, SUBCONSCIOUS_DISTR_PAT
```

Notes:
- `deploy` is the persistent path; `serve` is for local dev and tears down on exit.
- `app stop` terminates running containers but the deployment record remains
  (so it still appears in `app list` as stopped). Re-running `deploy` revives it.
- The endpoint URL stays the same across redeploys as long as `APP_NAME` is unchanged.

## Notes

- **Host RAM cap:** Modal limits a container to **1,650,688 MiB (~1.575 TiB)**
  of host RAM — this cap does NOT increase with GPU count. `MEMORY_MIB` is set to
  that max. The hierarchical KV-cache offload tier (`--hicache-ratio 8`) is bounded by it.
- **CPU cap:** 64 physical cores max (`CPU = 64.0`).
- **Cold start:** GLM-5.2 nvfp4 load + cuda-graph(bs=96) build is slow;
  `STARTUP_TIMEOUT = 3000`
- **Warm cache:** compiled kernels (sglang/deep_gemm/flashinfer) and the HF
  draft-model cache are persisted on Volume `tim_cache_vol` at
  `/opt/sgl-warm-cache` to speed subsequent cold starts.
- **Timeout:** `RUN_TIMEOUT = 86400` (container lifetime per cold cycle). If
  Modal rejects this at deploy (server-side cap), lower it and redeploy.

