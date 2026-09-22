"""One-time job to populate the GLM-5.3 weights Volume from Hugging Face.

Downloads the checkpoint used by deploy_glm5_3.py:

  - zai-org/GLM-5.3-FP8 -> /models/glm-5.3-fp8

The EAGLE deployment uses this local model path without a separate draft
checkpoint. Authentication uses the `SUBCONSCIOUS_HF_TOKEN` Modal Secret
(key: HF_TOKEN), upserted by write_secrets.py.

Run:
    uv run python scripts/write_secrets.py
    uv run modal run scripts/download_weights_glm5_3.py
"""
import modal

WEIGHTS_VOLUME = "glm_weights_vol"
WEIGHTS_MOUNT = "/models"
HF_SECRET = "SUBCONSCIOUS_HF_TOKEN"

MODELS = [
    {
        "repo_id": "zai-org/GLM-5.3-FP8",
        "revision": None,
        "local_dir": "/models/glm-5.3-fp8",
    },
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("huggingface_hub")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App("glm-5-3-weights-download")
vol = modal.Volume.from_name(WEIGHTS_VOLUME, create_if_missing=True)


@app.function(
    image=image,
    volumes={WEIGHTS_MOUNT: vol},
    secrets=[modal.Secret.from_name(HF_SECRET, required_keys=["HF_TOKEN"])],
    timeout=14400,
    cpu=8.0,
    memory=64 * 1024,
)
def download():
    from huggingface_hub import snapshot_download

    for spec in MODELS:
        rev = spec["revision"]
        label = f"{spec['repo_id']}" + (f"@{rev}" if rev else "")
        print(f"Downloading {label} -> {spec['local_dir']} ...")
        kwargs = {
            "repo_id": spec["repo_id"],
            "local_dir": spec["local_dir"],
            "repo_type": "model",
        }
        if rev:
            kwargs["revision"] = rev
        snapshot_download(**kwargs)
        vol.commit()
        print(f"  committed {spec['local_dir']}")
    print(f"Done. Weights on volume '{WEIGHTS_VOLUME}'.")


if __name__ == "__main__":
    with app.run():
        download.remote()
