#!/usr/bin/env python3
"""Idempotent upsert of local .env secrets into Modal.

Reads the repo-root .env (gitignored) and creates/overwrites Modal secrets:

  - SUBCONSCIOUS_HF_TOKEN    -> { HF_TOKEN }                                 (required)
  - SUBCONSCIOUS_DOCKERHUB   -> { REGISTRY_USERNAME, REGISTRY_PASSWORD }     (if Hub pair set)
  - SUBCONSCIOUS_DISTR_HUB   -> { REGISTRY_USERNAME, REGISTRY_PASSWORD }     (if distr pair set)

Username + token are combined into one Modal secret per registry so
`modal.Image.from_registry` can read `REGISTRY_USERNAME` / `REGISTRY_PASSWORD`.

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

HF_SECRET_NAME = "SUBCONSCIOUS_HF_TOKEN"
DOCKERHUB_SECRET_NAME = "SUBCONSCIOUS_DOCKERHUB"
DISTR_SECRET_NAME = "SUBCONSCIOUS_DISTR_HUB"

# Optional registry secret -> (username env var, token env var)
REGISTRY_SPECS: tuple[tuple[str, str, str], ...] = (
    (DOCKERHUB_SECRET_NAME, "DOCKERHUB_USERNAME", "DOCKERHUB_TOKEN"),
    (DISTR_SECRET_NAME, "DISTR_USERNAME", "DISTR_TOKEN"),
)


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

    hf_token = env.get("SUBCONSCIOUS_HF_TOKEN")
    if not hf_token:
        sys.exit("error: missing/empty SUBCONSCIOUS_HF_TOKEN in .env")

    pairs: list[tuple[str, str, str]] = []
    skipped: list[str] = []
    for secret_name, user_var, token_var in REGISTRY_SPECS:
        user, token = env.get(user_var), env.get(token_var)
        if user and token:
            pairs.append((secret_name, user, token))
        elif user or token:
            sys.exit(
                f"error: {user_var} and {token_var} must both be set to upsert "
                f"'{secret_name}' (got "
                f"{user_var}={'set' if user else 'empty'}, "
                f"{token_var}={'set' if token else 'empty'})"
            )
        else:
            skipped.append(f"  · skipping '{secret_name}' (no {user_var}/{token_var})")

    if not pairs:
        sys.exit(
            "error: provide at least one registry pair in .env: "
            "DOCKERHUB_USERNAME+DOCKERHUB_TOKEN and/or DISTR_USERNAME+DISTR_TOKEN"
        )

    print(
        f"Loaded HF token + {len(pairs)} registry secret(s) from {ENV_PATH};"
        " upserting into Modal..."
    )
    upsert_secret(HF_SECRET_NAME, {"HF_TOKEN": hf_token})
    for secret_name, user, token in pairs:
        upsert_secret(
            secret_name,
            {"REGISTRY_USERNAME": user, "REGISTRY_PASSWORD": token},
        )
    for line in skipped:
        print(line)
    print("Done. Verify with:  uv run modal secret list")


if __name__ == "__main__":
    main()
