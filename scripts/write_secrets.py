#!/usr/bin/env python3
"""Idempotent upsert of local .env secrets into Modal.

Reads the repo-root .env (gitignored) and creates/overwrites two Modal secrets:

  - SUBCONSCIOUS_HF_TOKEN   -> { HF_TOKEN: <value> }
  - SUBCONSCIOUS_DISTR_PAT  -> { REGISTRY_USERNAME: "-", REGISTRY_PASSWORD: <value> }

The Modal secret *names* use the SUBCONSCIOUS_* namespace, but the *keys*
inside are the names consumers expect:
  - huggingface_hub / sglang reads `HF_TOKEN`
  - modal.Image.from_registry reads `REGISTRY_USERNAME` / `REGISTRY_PASSWORD`

`modal secret create ... --force` is an upsert (create-or-overwrite), so this
is safe to re-run whenever values change. Secret values are passed to the
Modal CLI via a chmod-0600 temp JSON file (`--from-json`), never via argv
or shell history.

Usage:
    uv run python scripts/write_secrets.py
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = REPO_ROOT / ".env"

# Modal secret name -> { internal_key: (env_var | None, literal_value | None) }
SECRET_SPECS: dict[str, dict[str, tuple[str | None, str | None]]] = {
    "SUBCONSCIOUS_HF_TOKEN": {
        "HF_TOKEN": ("SUBCONSCIOUS_HF_TOKEN", None),
    },
    "SUBCONSCIOUS_DISTR_PAT": {
        "REGISTRY_USERNAME": (None, "-"),                  # distr login convention
        "REGISTRY_PASSWORD": ("SUBCONSCIOUS_DISTR_PAT", None),
    },
}

REQUIRED_ENV = ["SUBCONSCIOUS_HF_TOKEN", "SUBCONSCIOUS_DISTR_PAT"]


def load_dotenv(path: Path) -> dict[str, str]:
    if not path.exists():
        sys.exit(
            f"error: {path} not found. Copy .env.example to .env at the repo"
            " root and fill in the values first."
        )
    out: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def modal_cmd() -> list[str]:
    """Resolve the modal CLI (venv bin under `uv run`, else python -m modal)."""
    bin_path = shutil.which("modal")
    return [bin_path] if bin_path else [sys.executable, "-m", "modal"]


def upsert_secret(name: str, kv: dict[str, str]) -> None:
    # Temp JSON so values aren't exposed in argv / shell history.
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix=f"{name}-", text=True)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(kv, f)
        os.chmod(tmp, 0o600)
        cmd = modal_cmd() + ["secret", "create", "--from-json", tmp, "--force", name]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(
                f"error: failed to upsert secret '{name}'\n"
                f"stdout: {r.stdout}\nstderr: {r.stderr}"
            )
        print(f"  ✓ upserted secret '{name}' (keys: {', '.join(kv.keys())})")
    finally:
        os.unlink(tmp)


def main() -> None:
    env = load_dotenv(ENV_PATH)
    missing = [k for k in REQUIRED_ENV if not env.get(k)]
    if missing:
        sys.exit(f"error: missing/empty values in .env: {', '.join(missing)}")
    print(f"Loaded {len(REQUIRED_ENV)} value(s) from {ENV_PATH}; upserting into Modal...")
    for secret_name, spec in SECRET_SPECS.items():
        kv = {key: (literal if literal is not None else env[env_var]) for key, (env_var, literal) in spec.items()}
        upsert_secret(secret_name, kv)
    print("Done. Verify with:  uv run modal secret list")


if __name__ == "__main__":
    main()
