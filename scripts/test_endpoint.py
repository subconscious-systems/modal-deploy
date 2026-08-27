#!/usr/bin/env python
"""Send a sample streaming chat-completion request to the deployed endpoint.

Usage:
    uv run python scripts/test_endpoint.py https://<workspace>--glm-5.2-marathon.modal.run
"""
import asyncio
import json
import sys

MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What does Subconscious.dev do?"},
]


async def main(url: str) -> None:
    import aiohttp

    url = url.rstrip("/") + "/v1/chat/completions"
    payload = {"messages": MESSAGES, "model": "glm-5.2-nvfp4", "stream": True}
    async with aiohttp.ClientSession() as session:
        async with session.post(
            url, json=payload, headers={"Accept": "text/event-stream"}
        ) as resp:
            if resp.status != 200:
                print(f"HTTP {resp.status}: {await resp.text()}", file=sys.stderr)
                sys.exit(1)
            async for raw in resp.content:
                line = raw.decode().strip()
                if not line or line == "data: [DONE]" or not line.startswith("data: "):
                    continue
                chunk = json.loads(line[len("data: "):])
                if "choices" not in chunk:
                    continue
                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content") or delta.get("reasoning_content")
                if content:
                    print(content, end="", flush=True)
    print()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: test_endpoint.py <endpoint-url>", file=sys.stderr)
        sys.exit(1)
    asyncio.run(main(sys.argv[1]))
