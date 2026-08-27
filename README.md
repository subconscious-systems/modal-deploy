# modal-deploy

Experiment harness for deploying the **GLM-5.2-NVFP4** sglang
container onto Modal **4× B200** GPUs. The serve image is pulled from
either a private Docker Hub repository or a private `distr`
(`registry.distr.sh`) registry — selected by `.env` at deploy time.

## Layout

```
modal-deploy/
├── pyproject.toml          # uv env (Python 3.12, modal CLI)
├── deploy.py               # Modal app: pulls serve image, runs sglang on 4x B200
├── Dockerfile              # (optional/legacy) thin serving layer — not used when pulling the prebuilt serve image
├── .env.example            # template: HF + Hub/distr creds, plus deploy image/secret
├── scripts/
│   ├── write_secrets.py    # idempotent upsert of .env secrets into Modal
│   ├── download_weights.py # one-time job: pull GLM-5.2 + DFLASH draft into a Volume
│   ├── test_endpoint.py    # send a sample chat-completion request to the deployed endpoint
│   ├── run_claude.sh       # launch Claude Code against the Modal endpoint
│   └── run_opencode.sh     # launch OpenCode against the Modal endpoint
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
#    them idempotently into Modal:
#      SUBCONSCIOUS_HF_TOKEN    (key: HF_TOKEN)
#      SUBCONSCIOUS_DOCKERHUB   (keys: REGISTRY_USERNAME, REGISTRY_PASSWORD)
#      SUBCONSCIOUS_DISTR_HUB   (same keys; optional, if DISTR_* is set)
#    deploy.py reads DOCKER_TOKEN_SECRET (which of those registry secrets
#    to use) and ORANGELINE_IMAGE_NAME (which image to pull).
cp .env.example .env
$EDITOR .env                 # HF token + Hub and/or distr username+token
uv run python scripts/write_secrets.py

# 4. Populate the weights Volume once (GLM-5.2-NVFP4 + DFLASH draft). Both
#    are served from local paths under /models — the GPU container does not
#    pull from Hugging Face at startup.
uv run modal run scripts/download_weights.py
```

## Deploy

`write_secrets.py` combines username+token into a Modal secret. `deploy.py`
then picks which secret and which image to use:

- `DOCKERHUB_USERNAME` + `DOCKERHUB_TOKEN` → `SUBCONSCIOUS_DOCKERHUB`
- `DISTR_USERNAME` + `DISTR_TOKEN` → `SUBCONSCIOUS_DISTR_HUB`
- `DOCKER_TOKEN_SECRET` → `SUBCONSCIOUS_DOCKERHUB` or `SUBCONSCIOUS_DISTR_HUB`
- `ORANGELINE_IMAGE_NAME` → e.g. `subconsciouslabs/sglang-baseten:sm_100-v0.10`
  or `registry.distr.sh/subconscious/timrun:sm_100-v0.10`

Keep both secrets in Modal if you want; switch registries by changing
`DOCKER_TOKEN_SECRET` and `ORANGELINE_IMAGE_NAME`.

Other knobs live in the config block at the top of `deploy.py`:
- `HF_SECRET` → `SUBCONSCIOUS_HF_TOKEN` (upserted by `scripts/write_secrets.py`)
- `WEIGHTS_VOLUME`, `MODEL_PATH` → GLM weights on the Volume (`/models/glm-5.2-nvfp4`)
- `DRAFT_MODEL_PATH` → DFLASH draft on the same Volume (`/models/glm-5.2-fp8-dflash-v2`)
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
uv run python scripts/test_endpoint.py https://<workspace>--glm-5-2-marathon.modal.run
```

## Run Claude Code against it

`scripts/run_claude.sh` points Claude Code at the deployed endpoint for this
process only (env vars, same idea as `subc claude`). An `http(s)://` arg sets
the endpoint; other args go to `claude`.

```bash
./scripts/run_claude.sh
./scripts/run_claude.sh --continue
./scripts/run_claude.sh https://<workspace>--glm-5-2-marathon.modal.run
```

## Run OpenCode against it

`scripts/run_opencode.sh` does the same for OpenCode: ephemeral
`OPENCODE_CONFIG_CONTENT` (like `subc opencode`), nothing written to
`~/.opencode/`. An `http(s)://` arg sets the endpoint; other args go to
`opencode`.

```bash
./scripts/run_opencode.sh
./scripts/run_opencode.sh --continue
./scripts/run_opencode.sh https://<workspace>--glm-5-2-marathon.modal.run
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
uv run modal secret list                  # SUBCONSCIOUS_HF_TOKEN, SUBCONSCIOUS_DOCKERHUB, SUBCONSCIOUS_DISTR_HUB
```

Notes:
- `deploy` is the persistent path; `serve` is for local dev and tears down on exit.
- `app stop` terminates running containers but the deployment record remains
  (so it still appears in `app list` as stopped). Re-running `deploy` revives it.
- The endpoint URL stays the same across redeploys as long as `APP_NAME` is unchanged.

## Notes

- **Host RAM cap:** Modal limits a container to **1,650,688 MiB (~1.575 TiB)**
  of host RAM — this cap does NOT increase with GPU count. `MEMORY_MIB` is
  **1,363,149 MiB (1.3 TiB)** to make 4× B200 scheduling easier. The hierarchical
  KV-cache offload tier (`--hicache-ratio 8`) is bounded by it.
- **CPU cap:** 64 physical cores max (`CPU = 64.0`).
- **Cold start:** GLM-5.2 nvfp4 load + cuda-graph(bs=96) build is slow;
  `STARTUP_TIMEOUT = 3000`
- **Warm cache:** compiled kernels live in the image at `/opt/sgl-warm-cache`.
  The HF draft-model cache is persisted on Volume `tim_cache_vol` at
  `/mnt/sgl-warm-cache` (Modal cannot mount over the image's non-empty
  `/opt/sgl-warm-cache`).
- **Timeout:** `RUN_TIMEOUT = 86400` (container lifetime per cold cycle). If
  Modal rejects this at deploy (server-side cap), lower it and redeploy.

