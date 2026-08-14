from __future__ import annotations

import hashlib
from pathlib import Path

from .contracts import Artifact
from .interfaces import ImageProcessor


class PillowImageProcessor(ImageProcessor):
    name = "pillow"

    def __init__(self, min_width: int = 32, min_height: int = 32) -> None:
        self.min_width = min_width
        self.min_height = min_height

    def process(self, artifacts: list[Artifact], output_dir: Path) -> list[Artifact]:
        from PIL import Image

        target_dir = output_dir / "images"
        target_dir.mkdir(parents=True, exist_ok=True)
        seen: dict[str, str] = {}
        processed: list[Artifact] = []
        for artifact in artifacts:
            if artifact.kind != "image" or not artifact.asset_path:
                processed.append(artifact)
                continue
            source = Path(artifact.asset_path)
            with Image.open(source) as opened:
                image = opened.convert("RGB")
                if image.width < self.min_width or image.height < self.min_height:
                    artifact.metadata["filtered"] = "below-minimum-size"
                    continue
                raw_hash = hashlib.sha256(image.tobytes()).hexdigest()
                artifact.sha256 = raw_hash
                if raw_hash in seen:
                    artifact.metadata["duplicate_of"] = seen[raw_hash]
                    continue
                destination = target_dir / f"{artifact.block_id.replace(':', '-')}.png"
                image.save(destination, format="PNG", optimize=True)
            seen[raw_hash] = artifact.block_id
            artifact.asset_path = str(destination.resolve())
            processed.append(artifact)
        return processed
