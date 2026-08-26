"""One-time job to populate the GLM-5.2-NVFP4 weights Volume from Hugging Face.

The main model is gated on HF, so this needs the `SUBCONSCIOUS_HF_TOKEN` Modal
Secret (key: HF_TOKEN), upserted by write_secrets.py. The draft model
(SubconsciousDev/glm-5.2-fp8-dflash-v2) is NOT downloaded here — sglang fetches
it at server startup into the HF cache on the warm-cache volume (see deploy.py).

Run:
    uv run python write_secrets.py            # upsert SUBCONSCIOUS_HF_TOKEN
    uv run modal run download_weights.py
"""
import modal

WEIGHTS_VOLUME = "glm_weights_vol"
WEIGHTS_MOUNT = "/models"
MODEL_REPO = "nvidia/GLM-5.2-NVFP4"
MODEL_REVISION = "aec724e8c7b8ee9db3b48c01c320f63f9cdaf8aa"
MODEL_DIR = "/models/glm-5.2-nvfp4"
HF_SECRET = "SUBCONSCIOUS_HF_TOKEN"   # Modal secret (key: HF_TOKEN) upserted by write_secrets.py

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App("glm-weights-download")
vol = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    volumes={WEIGHTS_MOUNT: vol},
    secrets=[modal.Secret.from_name(HF_SECRET, required_keys=["HF_TOKEN"])],
    timeout=7200,  # large model download can take a while
    cpu=8.0,
    memory=64 * 1024,  # HF_XET_HIGH_PERFORMANCE wants >= 64 GB for buffering
)
def download():
    from huggingface_hub import snapshot_download

    print(f"Downloading {MODEL_REPO}@{MODEL_REVISION} -> {MODEL_DIR} ...")
    snapshot_download(
        repo_id=MODEL_REPO,
        revision=MODEL_REVISION,
        local_dir=MODEL_DIR,
        repo_type="model",
    )
    vol.commit()
    print(f"Done. Weights committed to volume '{WEIGHTS_VOLUME}' at {MODEL_DIR}.")


if __name__ == "__main__":
    with app.run():
        download()
