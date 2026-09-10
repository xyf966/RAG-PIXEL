from __future__ import annotations

import hashlib
import time
from pathlib import Path

from .contracts import Artifact
from .interfaces import ImageProcessor


def _normalized_rgb(image):
    from PIL import Image

    if "A" in image.getbands():
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, "white")
        background.alpha_composite(rgba)
        return background.convert("RGB")
    return image.convert("RGB")


def _rgb_content_sha256(image) -> str:
    value = f"{image.mode}:{image.width}x{image.height}\0".encode("ascii") + image.tobytes()
    return hashlib.sha256(value).hexdigest()


def image_content_sha256(path: Path) -> str:
    from PIL import Image

    with Image.open(path) as opened:
        opened.load()
        return _rgb_content_sha256(_normalized_rgb(opened))


class PillowImageProcessor(ImageProcessor):
    name = "pillow"

    def __init__(
        self,
        min_width: int = 32,
        min_height: int = 32,
        retain_duplicates: bool = False,
    ) -> None:
        self.min_width = min_width
        self.min_height = min_height
        self.retain_duplicates = retain_duplicates

    def process(self, artifacts: list[Artifact], output_dir: Path) -> list[Artifact]:
        from PIL import Image

        target_dir = output_dir / "images"
        target_dir.mkdir(parents=True, exist_ok=True)
        seen: dict[str, tuple[str, str]] = {}
        processed: list[Artifact] = []
        for artifact in artifacts:
            if artifact.kind != "image":
                processed.append(artifact)
                continue
            if not artifact.asset_path:
                artifact.metadata["filtered"] = "missing-asset-path"
                continue
            source = Path(artifact.asset_path)
            opened = None
            last_error: OSError | None = None
            # On Windows, freshly created files can briefly be unavailable to
            # a second decoder (antivirus/indexer/concurrent Studio worker).
            for attempt in range(6):
                try:
                    opened = Image.open(source)
                    opened.load()
                    break
                except OSError as exc:
                    last_error = exc
                    if attempt < 5:
                        time.sleep(0.1 * (attempt + 1))
            if opened is None:
                artifact.metadata["filtered"] = "missing-or-unreadable-asset"
                artifact.metadata["asset_error"] = str(last_error or "unknown image error")
                print(
                    f"Skipping unavailable visual asset {artifact.block_id}: "
                    f"{artifact.metadata['asset_error']}",
                    flush=True,
                )
                continue
            with opened:
                image = _normalized_rgb(opened)
                if image.width < self.min_width or image.height < self.min_height:
                    artifact.metadata["filtered"] = "below-minimum-size"
                    continue
                extrema = image.getextrema()
                if all(high <= 1 for _low, high in extrema):
                    artifact.metadata["filtered"] = "black-placeholder"
                    continue
                if all(low >= 254 for low, _high in extrema):
                    artifact.metadata["filtered"] = "white-placeholder"
                    continue
                raw_hash = _rgb_content_sha256(image)
                artifact.sha256 = raw_hash
                if raw_hash in seen:
                    duplicate_id, duplicate_path = seen[raw_hash]
                    artifact.metadata["duplicate_of"] = duplicate_id
                    if self.retain_duplicates:
                        artifact.asset_path = duplicate_path
                        processed.append(artifact)
                    continue
                destination = target_dir / f"{artifact.block_id.replace(':', '-')}.png"
                image.save(destination, format="PNG", optimize=True)
            seen[raw_hash] = (artifact.block_id, str(destination.resolve()))
            artifact.asset_path = str(destination.resolve())
            processed.append(artifact)
        return processed
