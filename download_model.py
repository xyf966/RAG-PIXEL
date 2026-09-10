from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
import truststore


URL = "https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B/resolve/main/model.safetensors"
EXPECTED_SIZE = 4_255_140_312


def main() -> None:
    truststore.inject_into_ssl()
    target_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "model-cache/Qwen3-VL-Embedding-2B")
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "model.safetensors"
    partial = target.with_suffix(".safetensors.part")

    if target.exists() and target.stat().st_size == EXPECTED_SIZE:
        print(f"Model already complete: {target}", flush=True)
        return

    redirect = requests.head(URL, allow_redirects=False, timeout=30)
    redirect.raise_for_status()
    signed_url = redirect.headers["location"]
    total = EXPECTED_SIZE
    block_size = 8 * 1024 * 1024
    blocks = [(start, min(start + block_size, total) - 1) for start in range(0, total, block_size)]

    with partial.open("wb") as handle:
        handle.truncate(total)

    def fetch(block: tuple[int, int]) -> int:
        start, end = block
        expected = end - start + 1
        last_error = None
        for attempt in range(1, 5):
            try:
                with requests.get(
                    signed_url,
                    headers={"Range": f"bytes={start}-{end}"},
                    stream=True,
                    timeout=(30, 180),
                ) as response:
                    if response.status_code != 206:
                        raise RuntimeError(f"range {start}-{end}: HTTP {response.status_code}")
                    written = 0
                    with partial.open("r+b", buffering=0) as handle:
                        handle.seek(start)
                        for chunk in response.iter_content(chunk_size=4 * 1024 * 1024):
                            if chunk:
                                handle.write(chunk)
                                written += len(chunk)
                    if written != expected:
                        raise RuntimeError(f"range {start}-{end}: got {written}, expected {expected}")
                    return written
            except Exception as exc:
                last_error = exc
                time.sleep(attempt * 2)
        raise RuntimeError(str(last_error))

    started = time.time()
    downloaded = 0
    next_report = 64 * 1024 * 1024
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, block) for block in blocks]
        for future in as_completed(futures):
            downloaded += future.result()
            if downloaded >= next_report or downloaded == total:
                elapsed = max(time.time() - started, 0.001)
                print(
                    f"{downloaded / total:6.1%}  "
                    f"{downloaded / 1024**3:5.2f}/{total / 1024**3:5.2f} GiB  "
                    f"{downloaded / 1024**2 / elapsed:5.1f} MiB/s",
                    flush=True,
                )
                next_report += 64 * 1024 * 1024

    if downloaded != total or partial.stat().st_size != total:
        raise RuntimeError(f"Size mismatch: got {downloaded}, expected {total}")
    os.replace(partial, target)
    print(f"Download complete: {target} ({downloaded} bytes)", flush=True)


if __name__ == "__main__":
    main()
