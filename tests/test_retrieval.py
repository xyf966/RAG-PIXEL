from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from hybrid_input.processors import image_content_sha256
from hybrid_input.retrieval import (
    QUERY_INSTRUCTION,
    CandidateRanker,
    EvidenceAssembler,
    EvidenceExpander,
    HybridSearchEngine,
    ModalitySearcher,
    PixelRAGQueryEmbedder,
    QueryAwareTextReranker,
    SearchCandidate,
    SnapshotStore,
    VectorRetriever,
    VectorSearchResult,
    _lexical_relevance_score,
)
from hybrid_input.retrieval_contracts import RetrievalFilter, RetrievalRequest


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metadata_row(modality: str, build_id: str) -> dict:
    return {
        "record_id": f"{build_id}-{modality}",
        "document_id": "document-1",
        "modality": modality,
        "source_block_ids": [f"{modality}-block"],
        "embedding_text": f"{modality} content",
        "original_content": {"text": f"{modality} content"},
        "context": "",
        "structure": {},
        "provenance": {"source_file": "sample.pdf", "page": 1},
        "content_hash": "hash",
        "overlap_source_block_ids": [],
        "asset_path": None,
        "visual_type": None,
        "source_path": "sample.pdf",
        "document_type": "pdf",
    }


def _publish_snapshot(
    index_dir: Path,
    build_id: str,
    *,
    dimension: int = 4,
    counts: dict[str, int] | None = None,
    vectors_by_modality: dict[str, np.ndarray] | None = None,
    metadata_overrides: dict[str, list[dict]] | None = None,
    manifest_warnings: list[str] | None = None,
) -> Path:
    import faiss

    counts = counts or {"text": 1, "table": 1, "visual": 1}
    snapshot = index_dir / "snapshots" / build_id
    snapshot.mkdir(parents=True)
    files: dict[str, str] = {}
    for modality in ("text", "table", "visual"):
        supplied = (vectors_by_modality or {}).get(modality)
        if supplied is not None:
            vectors = np.asarray(supplied, dtype=np.float32)
            if vectors.ndim != 2 or vectors.shape[1] != dimension:
                raise ValueError("Synthetic vectors have an invalid shape")
            norms = np.linalg.norm(vectors, axis=1)
            if len(vectors) and np.any(norms == 0):
                raise ValueError("Synthetic vectors must be non-zero")
            if len(vectors):
                vectors = vectors / norms[:, None]
            count = len(vectors)
        else:
            count = counts.get(modality, 0)
            vectors = np.zeros((count, dimension), dtype=np.float32)
            if count:
                vectors[:, 0] = 1.0
        index = faiss.IndexFlatIP(dimension)
        if count:
            index.add(vectors)
        index_path = snapshot / f"{modality}.faiss"
        faiss.write_index(index, str(index_path))
        metadata_path = snapshot / f"{modality}-metadata.jsonl"
        rows = [
            _metadata_row(modality, f"{build_id}-{number}")
            for number in range(count)
        ]
        overrides = (metadata_overrides or {}).get(modality, [])
        if len(overrides) > count:
            raise ValueError("Too many synthetic metadata overrides")
        for row, override in zip(rows, overrides):
            row.update(override)
        if modality == "visual":
            asset_dir = snapshot / "visual-assets" / "document-1" / "images"
            asset_dir.mkdir(parents=True, exist_ok=True)
            for number, row in enumerate(rows):
                asset = asset_dir / f"visual-{number}.png"
                Image.new("RGB", (32, 32), (20 + number, 40, 60)).save(asset)
                row["asset_path"] = asset.relative_to(snapshot).as_posix()
                row["visual_type"] = "image"
                row["original_content"] = {
                    "context": "",
                    "sha256": image_content_sha256(asset),
                    "visual_type": "image",
                }
        metadata_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        files[index_path.name] = _sha256(index_path)
        files[metadata_path.name] = _sha256(metadata_path)

    report_path = snapshot / "build-report.json"
    report_path.write_text("{}", encoding="utf-8")
    files[report_path.name] = _sha256(report_path)
    manifest = {
        "schema_version": "2.0",
        "build_id": build_id,
        "status": "complete",
        "model": "fake-unified-model",
        "vector_dimension": dimension,
        "index_type": "faiss.IndexFlatIP",
        "normalized_vectors": True,
        "sources": [],
        "counts": {},
        "files": files,
        "warnings": manifest_warnings or [],
    }
    (snapshot / "manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    index_dir.mkdir(parents=True, exist_ok=True)
    (index_dir / "CURRENT").write_text(build_id + "\n", encoding="utf-8")
    return snapshot


class SnapshotStoreTests(unittest.TestCase):
    def test_lexical_relevance_tolerates_spelling_error_and_prioritizes_identifier(self) -> None:
        related = _lexical_relevance_score(
            "steering whell stw195",
            "Steering Wheel STW195 Formula E",
        )
        unrelated = _lexical_relevance_score(
            "steering whell stw195",
            "Bosch motorsport racing series logos",
        )
        self.assertGreaterEqual(related, 4.5)
        self.assertEqual(unrelated, 0.0)

    def test_page_sized_pdf_visual_is_excluded_from_final_retrieval(self) -> None:
        metadata = _metadata_row("visual", "page")
        metadata.update(
            {
                "document_type": "pdf",
                "provenance": {"page": 3, "bbox": [0.0137, 0.0, 0.9859, 1.001]},
            }
        )
        self.assertFalse(ModalitySearcher._is_final_evidence_eligible("visual", metadata))
        metadata["provenance"]["bbox"] = [0.2, 0.2, 0.7, 0.7]
        self.assertTrue(ModalitySearcher._is_final_evidence_eligible("visual", metadata))

    def test_loads_all_modalities_and_reuses_cached_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            _publish_snapshot(index_dir, "build-a")
            store = SnapshotStore(index_dir)

            with patch(
                "hybrid_input.retrieval.current_snapshot",
                wraps=__import__(
                    "hybrid_input.indexing",
                    fromlist=["current_snapshot"],
                ).current_snapshot,
            ) as validate:
                first = store.get_snapshot()
                second = store.get_snapshot()

            self.assertIs(first, second)
            self.assertEqual(validate.call_count, 1)
            self.assertEqual(first.build_id, "build-a")
            self.assertEqual(first.dimension, 4)
            self.assertEqual(tuple(first.indexes), ("text", "table", "visual"))
            self.assertTrue(all(index.ntotal == 1 for index in first.indexes.values()))
            self.assertEqual(first.metadata["table"][0]["modality"], "table")

    def test_loaded_mappings_and_metadata_rows_are_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            _publish_snapshot(index_dir, "build-a")
            loaded = SnapshotStore(index_dir).get_snapshot()

            with self.assertRaises(TypeError):
                loaded.indexes["text"] = None
            with self.assertRaises(TypeError):
                loaded.metadata["text"][0]["record_id"] = "changed"
            with self.assertRaises(TypeError):
                loaded.manifest["build_id"] = "changed"

    def test_switches_to_a_new_current_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            _publish_snapshot(index_dir, "build-a")
            store = SnapshotStore(index_dir)
            first = store.get_snapshot()

            _publish_snapshot(
                index_dir,
                "build-b",
                counts={"text": 2, "table": 0, "visual": 1},
            )
            second = store.get_snapshot()

            self.assertEqual(first.build_id, "build-a")
            self.assertEqual(second.build_id, "build-b")
            self.assertIsNot(first, second)
            self.assertEqual(second.indexes["text"].ntotal, 2)
            self.assertEqual(second.metadata["table"], ())
            self.assertEqual(store.cached_build_id, "build-b")

    def test_failed_new_snapshot_does_not_replace_cached_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            _publish_snapshot(index_dir, "build-a")
            store = SnapshotStore(index_dir)
            first = store.get_snapshot()
            broken = _publish_snapshot(index_dir, "build-b")
            (broken / "text-metadata.jsonl").write_text("{broken", encoding="utf-8")

            with self.assertRaises(RuntimeError):
                store.get_snapshot()

            self.assertEqual(store.cached_build_id, "build-a")
            (index_dir / "CURRENT").write_text("build-a\n", encoding="utf-8")
            self.assertIs(store.get_snapshot(), first)

    def test_rejects_count_mismatch_even_after_snapshot_selection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            snapshot = _publish_snapshot(index_dir, "build-a")
            (snapshot / "text-metadata.jsonl").write_text("", encoding="utf-8")
            store = SnapshotStore(index_dir)

            with patch("hybrid_input.retrieval.current_snapshot", return_value=snapshot):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "text index count 1 != metadata count 0",
                ):
                    store.get_snapshot()

    def test_rejects_manifest_directory_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            snapshot = _publish_snapshot(index_dir, "build-a")
            manifest_path = snapshot / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["build_id"] = "different-build"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with patch("hybrid_input.retrieval.current_snapshot", return_value=snapshot):
                with self.assertRaisesRegex(RuntimeError, "does not match"):
                    SnapshotStore(index_dir).get_snapshot()

    def test_missing_current_pointer_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError, "No published index snapshot"):
                SnapshotStore(Path(temporary) / "index").get_snapshot()


class FixedQueryEmbedder:
    def __init__(self, vector, model_name: str = "fake-unified-model") -> None:
        self.vector = vector
        self.model_name = model_name
        self.calls: list[str] = []

    def embed_query(self, query_text: str):
        self.calls.append(query_text)
        return self.vector


class VectorRetrieverTests(unittest.TestCase):
    def test_exact_keyword_recall_adds_record_outside_vector_top_k(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            _publish_snapshot(
                index_dir,
                "build-a",
                vectors_by_modality={
                    "text": np.asarray(
                        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]],
                        dtype=np.float32,
                    ),
                    "table": np.empty((0, 4), dtype=np.float32),
                    "visual": np.empty((0, 4), dtype=np.float32),
                },
                metadata_overrides={
                    "text": [
                        {
                            "record_id": "unrelated-1983",
                            "embedding_text": "1983",
                            "original_content": {"text": "1983"},
                        },
                        {
                            "record_id": "television-page-7",
                            "embedding_text": "Display Resolution: 3840 x 2160",
                            "original_content": {
                                "text": "Display Resolution: 3840 x 2160"
                            },
                            "provenance": {"source_file": "电视.pdf", "page": 7},
                        },
                    ]
                },
            )
            result = VectorRetriever(
                SnapshotStore(index_dir),
                FixedQueryEmbedder([1.0, 0.0, 0.0, 0.0]),
            ).search(
                RetrievalRequest(
                    "3840",
                    top_k=1,
                    candidate_k=1,
                    modalities=("text",),
                )
            )

            by_id = {
                candidate.metadata["record_id"]: candidate
                for candidate in result.candidates["text"]
            }
            self.assertEqual(set(by_id), {"unrelated-1983", "television-page-7"})
            exact = by_id["television-page-7"]
            self.assertEqual(exact.modality_rank, 0)
            self.assertEqual(exact.keyword_rank, 1)
            self.assertTrue(exact.exact_match)
            ranked = CandidateRanker().rank(result, top_k=1)
            self.assertEqual(
                ranked.candidates[0].candidate.metadata["record_id"],
                "television-page-7",
            )

    def test_evidence_expansion_returns_bounded_neighbor_window_across_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            _publish_snapshot(
                index_dir,
                "build-a",
                vectors_by_modality={
                    "text": np.asarray(
                        [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.9, 0.1, 0.0, 0.0],
                            [0.8, 0.2, 0.0, 0.0],
                            [0.7, 0.3, 0.0, 0.0],
                        ],
                        dtype=np.float32,
                    ),
                    "table": np.empty((0, 4), dtype=np.float32),
                    "visual": np.empty((0, 4), dtype=np.float32),
                },
                metadata_overrides={
                    "text": [
                        {
                            "record_id": "power-page-4-a",
                            "embedding_text": "POWER safety instructions",
                            "original_content": {"text": "POWER safety instructions"},
                            "provenance": {"source_file": "电视.pdf", "page": 4},
                            "next_record_id": "power-page-4-b",
                        },
                        {
                            "record_id": "position-decoy",
                            "embedding_text": "unrelated position decoy",
                            "original_content": {"text": "unrelated position decoy"},
                            "provenance": {"source_file": "电视.pdf", "page": 4},
                        },
                        {
                            "record_id": "power-page-4-b",
                            "embedding_text": "Pay particular attention to cords at the",
                            "original_content": {
                                "text": "Pay particular attention to cords at the"
                            },
                            "overlap_source_block_ids": ["text-block"],
                            "provenance": {"source_file": "电视.pdf", "page": 4},
                            "previous_record_id": "power-page-4-a",
                            "next_record_id": "power-page-5",
                        },
                        {
                            "record_id": "power-page-5",
                            "embedding_text": "plug end, at wall outlets",
                            "original_content": {"text": "plug end, at wall outlets"},
                            "provenance": {"source_file": "电视.pdf", "page": 5},
                            "previous_record_id": "power-page-4-b",
                        },
                    ]
                },
            )
            snapshot = SnapshotStore(index_dir).get_snapshot()
            seed = SearchCandidate(
                modality="text",
                modality_rank=1,
                index_position=0,
                raw_score=0.8,
                metadata=snapshot.metadata["text"][0],
            )

            adjacent, pages = EvidenceExpander().expand(snapshot, seed, "power")

            self.assertEqual(
                [item["record_id"] for item in adjacent],
                ["power-page-4-b", "power-page-5"],
            )
            self.assertEqual(pages, [4, 5])
            self.assertEqual(adjacent[-1]["provenance"]["page"], 5)

            reverse_seed = SearchCandidate(
                modality="text",
                modality_rank=1,
                index_position=3,
                raw_score=0.8,
                metadata=snapshot.metadata["text"][3],
            )
            previous, reverse_pages = EvidenceExpander().expand(
                snapshot,
                reverse_seed,
                "plug end",
            )
            self.assertEqual(
                [item["record_id"] for item in previous],
                ["power-page-4-b"],
            )
            self.assertEqual(reverse_pages, [4, 5])

    def test_embeds_once_and_searches_only_requested_modalities(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            _publish_snapshot(
                index_dir,
                "build-a",
                vectors_by_modality={
                    "text": np.asarray(
                        [[0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
                        dtype=np.float32,
                    ),
                    "table": np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                    "visual": np.asarray([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32),
                },
            )
            embedder = FixedQueryEmbedder([[2.0, 0.0, 0.0, 0.0]])
            retriever = VectorRetriever(SnapshotStore(index_dir), embedder)

            result = retriever.search(
                RetrievalRequest(
                    "  target query  ",
                    candidate_k=2,
                    top_k=1,
                    modalities=("table", "text"),
                )
            )

            self.assertEqual(embedder.calls, ["target query"])
            self.assertEqual(tuple(result.candidates), ("text", "table"))
            self.assertNotIn("visual", result.candidates)
            self.assertEqual(
                [item.index_position for item in result.candidates["text"]],
                [1, 0],
            )
            self.assertAlmostEqual(result.candidates["text"][0].raw_score, 1.0)
            self.assertEqual(
                [item.modality_rank for item in result.candidates["text"]],
                [1, 2],
            )

    def test_empty_modality_returns_no_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            _publish_snapshot(
                index_dir,
                "build-a",
                counts={"text": 0, "table": 1, "visual": 0},
            )
            result = VectorRetriever(
                SnapshotStore(index_dir),
                FixedQueryEmbedder([1.0, 0.0, 0.0, 0.0]),
            ).search(
                RetrievalRequest(
                    "query",
                    top_k=1,
                    candidate_k=1,
                    modalities=("text",),
                )
            )
            self.assertEqual(result.candidates["text"], ())

    def test_candidate_limit_is_applied_per_modality(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            _publish_snapshot(
                index_dir,
                "build-a",
                counts={"text": 5, "table": 0, "visual": 0},
            )
            result = VectorRetriever(
                SnapshotStore(index_dir),
                FixedQueryEmbedder([1.0, 0.0, 0.0, 0.0]),
            ).search(
                RetrievalRequest(
                    "query",
                    top_k=2,
                    candidate_k=3,
                    modalities=("text",),
                )
            )
            self.assertEqual(len(result.candidates["text"]), 3)

    def test_rejects_invalid_query_vectors(self) -> None:
        invalid = (
            ([1.0, 0.0, 0.0], r"expected \(4,\)"),
            ([0.0, 0.0, 0.0, 0.0], "zero or non-finite norm"),
            ([np.nan, 0.0, 0.0, 0.0], "non-finite values"),
            (np.ones((2, 4), dtype=np.float32), r"expected \(4,\)"),
        )
        for vector, message in invalid:
            with self.subTest(message=message), tempfile.TemporaryDirectory() as temporary:
                index_dir = Path(temporary) / "index"
                _publish_snapshot(index_dir, "build-a")
                retriever = VectorRetriever(
                    SnapshotStore(index_dir),
                    FixedQueryEmbedder(vector),
                )
                with self.assertRaisesRegex(RuntimeError, message):
                    retriever.search(RetrievalRequest("query"))

    def test_model_mismatch_is_rejected_before_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            _publish_snapshot(index_dir, "build-a")
            embedder = FixedQueryEmbedder(
                [1.0, 0.0, 0.0, 0.0],
                model_name="different-model",
            )
            retriever = VectorRetriever(SnapshotStore(index_dir), embedder)

            with self.assertRaisesRegex(RuntimeError, "does not match"):
                retriever.search(RetrievalRequest("query"))
            self.assertEqual(embedder.calls, [])

    def test_pixelrag_adapter_uses_the_query_instruction(self) -> None:
        class Session:
            model_name = "fake-unified-model"

            def __init__(self) -> None:
                self.calls = []

            def embed_texts(self, texts, instruction=""):
                self.calls.append((texts, instruction))
                return np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        session = Session()
        embedder = PixelRAGQueryEmbedder(session)
        vector = embedder.embed_query("query")

        self.assertEqual(vector.shape, (1, 4))
        self.assertEqual(session.calls, [(["query"], QUERY_INSTRUCTION)])

    def test_filters_expand_search_until_enough_candidates_are_found(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            angles = np.asarray([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
            vectors = np.stack(
                [np.cos(angles), np.sin(angles), np.zeros(6), np.zeros(6)],
                axis=1,
            ).astype(np.float32)
            _publish_snapshot(
                index_dir,
                "build-a",
                vectors_by_modality={
                    "text": vectors,
                    "table": np.empty((0, 4), dtype=np.float32),
                    "visual": np.empty((0, 4), dtype=np.float32),
                },
                metadata_overrides={
                    "text": [
                        {"document_id": "other", "document_type": "pdf"},
                        {"document_id": "other", "document_type": "pdf"},
                        {"document_id": "other", "document_type": "pdf"},
                        {"document_id": "other", "document_type": "pdf"},
                        {
                            "document_id": "target",
                            "document_type": "docx",
                            "source_path": "C:/target.docx",
                        },
                        {
                            "document_id": "target",
                            "document_type": "docx",
                            "source_path": "C:/target.docx",
                        },
                    ]
                },
            )
            result = VectorRetriever(
                SnapshotStore(index_dir),
                FixedQueryEmbedder([1.0, 0.0, 0.0, 0.0]),
            ).search(
                RetrievalRequest(
                    "query",
                    top_k=2,
                    candidate_k=2,
                    modalities=("text",),
                    filters=RetrievalFilter(
                        document_ids={"target"},
                        document_types={"DOCX"},
                        source_paths={"C:/target.docx"},
                    ),
                )
            )

            self.assertEqual(
                [candidate.index_position for candidate in result.candidates["text"]],
                [4, 5],
            )
            self.assertEqual(
                [candidate.modality_rank for candidate in result.candidates["text"]],
                [1, 2],
            )


def _candidate(
    modality: str,
    rank: int,
    raw_score: float,
    record_id: str,
    *,
    document_id: str = "document-1",
    content_hash: str = "",
    source_block_ids: list[str] | None = None,
    overlap_source_block_ids: list[str] | None = None,
    keyword_rank: int | None = None,
    keyword_score: float = 0.0,
    exact_match: bool = False,
    text: str = "",
) -> SearchCandidate:
    return SearchCandidate(
        modality=modality,
        modality_rank=rank,
        index_position=rank - 1,
        raw_score=raw_score,
        keyword_rank=keyword_rank,
        keyword_score=keyword_score,
        exact_match=exact_match,
        metadata={
            "record_id": record_id,
            "document_id": document_id,
            "modality": modality,
            "content_hash": content_hash,
            "source_block_ids": source_block_ids or [record_id],
            "overlap_source_block_ids": overlap_source_block_ids or [],
            "embedding_text": text,
            "original_content": {"text": text} if text else {},
        },
    )


def _vector_result(**modalities) -> VectorSearchResult:
    return VectorSearchResult(snapshot=object(), candidates=modalities)


class CandidateRankerTests(unittest.TestCase):
    def test_vector_rrf_is_rank_first_and_deterministic(self) -> None:
        result = _vector_result(
            text=(
                _candidate("text", 1, 0.70, "text-1"),
                _candidate("text", 2, 0.99, "text-2"),
            ),
            table=(_candidate("table", 1, 0.90, "table-1"),),
            visual=(_candidate("visual", 1, 0.80, "visual-1"),),
        )

        first = CandidateRanker().rank(result, top_k=4)
        second = CandidateRanker().rank(result, top_k=4)

        expected = ["visual-1", "table-1", "text-1", "text-2"]
        self.assertEqual(
            [item.candidate.metadata["record_id"] for item in first.candidates],
            expected,
        )
        self.assertEqual(first.candidates, second.candidates)
        self.assertGreater(first.candidates[0].fusion_score, first.candidates[1].fusion_score)

    def test_exact_keyword_candidate_bypasses_vector_threshold_and_wins(self) -> None:
        result = _vector_result(
            text=(
                _candidate("text", 1, 0.60, "unrelated"),
                _candidate(
                    "text",
                    0,
                    0.10,
                    "exact-3840",
                    keyword_rank=1,
                    keyword_score=2.0,
                    exact_match=True,
                ),
            )
        )

        ranked = CandidateRanker().rank(result, top_k=2)

        self.assertEqual(
            [item.candidate.metadata["record_id"] for item in ranked.candidates],
            ["exact-3840", "unrelated"],
        )

    def test_modality_weights_can_adjust_score_aware_rank(self) -> None:
        result = _vector_result(
            text=(_candidate("text", 1, 0.50, "text-1"),),
            table=(_candidate("table", 1, 0.95, "table-1"),),
        )
        ranked = CandidateRanker({"text": 2.0}).rank(result, top_k=2)
        self.assertEqual(ranked.candidates[0].candidate.metadata["record_id"], "text-1")

    def test_low_relevance_channel_leader_is_filtered(self) -> None:
        result = _vector_result(
            text=(_candidate("text", 1, 0.57974, "text-1"),),
            table=(_candidate("table", 1, 0.35941, "table-1"),),
            visual=(_candidate("visual", 1, 0.47387, "visual-1"),),
        )

        ranked = CandidateRanker().rank(result, top_k=3)

        self.assertEqual(
            [item.candidate.metadata["record_id"] for item in ranked.candidates],
            ["visual-1", "text-1"],
        )

    def test_visual_score_has_small_cross_modal_calibration(self) -> None:
        result = _vector_result(
            text=(_candidate("text", 1, 0.48, "text-1"),),
            visual=(_candidate("visual", 1, 0.45, "visual-1"),),
        )

        ranked = CandidateRanker().rank(result, top_k=2)

        self.assertEqual(ranked.candidates[0].candidate.metadata["record_id"], "visual-1")

    def test_thresholds_can_be_overridden_per_modality(self) -> None:
        result = _vector_result(
            table=(_candidate("table", 1, 0.35, "table-1"),),
        )
        ranked = CandidateRanker(min_raw_scores={"table": 0.30}).rank(result, top_k=1)
        self.assertEqual(ranked.candidates[0].candidate.metadata["record_id"], "table-1")

    def test_exact_and_content_hash_duplicates_are_removed_and_top_k_is_filled(self) -> None:
        result = _vector_result(
            text=(
                _candidate("text", 1, 0.90, "record-1", content_hash="same"),
                _candidate("text", 2, 0.85, "record-2", content_hash="same"),
                _candidate("text", 3, 0.80, "record-3", content_hash="different"),
            ),
            table=(_candidate("table", 1, 0.70, "record-1"),),
        )
        ranked = CandidateRanker().rank(result, top_k=2)
        self.assertEqual(
            [item.candidate.metadata["record_id"] for item in ranked.candidates],
            ["record-1", "record-3"],
        )

    def test_highly_overlapping_text_sources_are_deduplicated(self) -> None:
        result = _vector_result(
            text=(
                _candidate(
                    "text", 1, 0.9, "text-1", source_block_ids=["a", "b", "c"]
                ),
                _candidate(
                    "text", 2, 0.8, "text-2", source_block_ids=["a", "b", "c", "d"]
                ),
                _candidate("text", 3, 0.7, "text-3", source_block_ids=["e"]),
            )
        )
        ranked = CandidateRanker().rank(result, top_k=2)
        self.assertEqual(
            [item.candidate.metadata["record_id"] for item in ranked.candidates],
            ["text-1", "text-3"],
        )

    def test_adjacent_overlap_metadata_does_not_delete_new_text(self) -> None:
        result = _vector_result(
            text=(
                _candidate("text", 1, 0.9, "text-1", source_block_ids=["a"]),
                _candidate(
                    "text",
                    2,
                    0.8,
                    "text-2",
                    source_block_ids=["b"],
                    overlap_source_block_ids=["a"],
                ),
            )
        )
        ranked = CandidateRanker().rank(result, top_k=2)
        self.assertEqual(len(ranked.candidates), 2)

    def test_same_content_hash_across_modalities_is_preserved(self) -> None:
        result = _vector_result(
            text=(_candidate("text", 1, 0.9, "text-1", content_hash="same"),),
            visual=(_candidate("visual", 1, 0.8, "visual-1", content_hash="same"),),
        )
        ranked = CandidateRanker().rank(result, top_k=2)
        self.assertEqual(len(ranked.candidates), 2)

    def test_text_reranker_promotes_query_aligned_text_only(self) -> None:
        result = _vector_result(
            text=(
                _candidate(
                    "text",
                    1,
                    0.70,
                    "feed-solution",
                    keyword_rank=1,
                    text="解决送纸问题：调整纸张宽度导板并重新装入纸张。",
                ),
            ),
            table=(_candidate("table", 1, 0.90, "table-1"),),
            visual=(_candidate("visual", 1, 0.80, "visual-1"),),
        )
        coarse = CandidateRanker().rank(result, top_k=3)

        reranked = QueryAwareTextReranker().rerank(
            coarse,
            "怎么解决送纸问题",
            top_k=3,
        )

        self.assertEqual(
            [item.candidate.metadata["record_id"] for item in reranked.candidates],
            ["feed-solution", "visual-1", "table-1"],
        )

    def test_text_reranker_does_not_boost_unrelated_text(self) -> None:
        result = _vector_result(
            text=(_candidate("text", 1, 0.70, "text-1", text="保养周期说明"),),
            visual=(_candidate("visual", 1, 0.80, "visual-1"),),
        )
        coarse = CandidateRanker().rank(result, top_k=2)

        reranked = QueryAwareTextReranker().rerank(coarse, "怎么解决送纸问题", top_k=2)

        self.assertEqual(reranked.candidates, coarse.candidates)


class HybridSearchEngineTests(unittest.TestCase):
    def test_text_anchor_attaches_at_most_one_table_and_visual_per_page(self) -> None:
        def row(record_id: str, modality: str, text: str) -> dict:
            return {
                "record_id": record_id,
                "document_id": "document-1",
                "document_type": "pdf",
                "embedding_text": text,
                "original_content": {"text": text},
                "source_block_ids": [record_id],
                "structure": {},
                "provenance": {"page": 3, "bbox": [0.1, 0.1, 0.4, 0.3]},
                "asset_path": f"assets/{record_id}.png" if modality == "visual" else None,
            }

        anchor = _candidate(
            "text",
            1,
            0.8,
            "text-anchor",
            text="解决送纸问题",
        )
        anchor.metadata.update(row("text-anchor", "text", "解决送纸问题"))
        snapshot = type(
            "Snapshot",
            (),
            {
                "metadata": {
                    "text": [anchor.metadata],
                    "table": [
                        row("table-1", "table", "送纸处理步骤"),
                        row("table-2", "table", "送纸处理步骤补充"),
                    ],
                    "visual": [
                        row("visual-1", "visual", "送纸机构示意图"),
                        row("visual-2", "visual", "送纸机构细节图"),
                    ],
                }
            },
        )()

        blocks = EvidenceAssembler().assemble(snapshot, anchor, [], "怎么解决送纸问题")
        related = [block for block in blocks if block["role"] == "related"]

        self.assertEqual([block["modality"] for block in related], ["table", "visual"])
        self.assertTrue(
            all(block["relation"] == "text_anchor_same_container" for block in related)
        )

    def test_table_evidence_uses_local_chunk_instead_of_repeating_full_table(self) -> None:
        metadata = {
            "record_id": "table-row-5",
            "embedding_text": (
                "Columns: 分类 | 权益说明\n"
                "Row: 无形资产使用 | CTCC围场内品牌产品销售\n"
                "Row: 无形资产使用 | CTCC车辆数据支持"
            ),
            "original_content": {"raw": "完整的38行原表"},
            "structure": {
                "headers": ["分类", "权益说明"],
                "rows": [
                    ["无形资产使用", "CTCC围场内品牌产品销售"],
                    ["无形资产使用", "CTCC车辆数据支持"],
                ],
                "row_start": 5,
                "row_end": 6,
            },
            "provenance": {"sheet": "中文", "cell_range": "A1:H38"},
            "source_block_ids": ["table-block"],
        }

        block = EvidenceAssembler._block(metadata, "table", "core")

        self.assertIn("CTCC围场内品牌产品销售", block["content"]["text"])
        self.assertNotIn("完整的38行原表", block["content"]["text"])
        self.assertEqual(block["content"]["structure"]["row_start"], 5)

    def test_search_returns_public_hits_with_provenance_and_visual_asset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            snapshot = _publish_snapshot(
                index_dir,
                "build-a",
                vectors_by_modality={
                    "text": np.asarray([[0.8, 0.6, 0.0, 0.0]], dtype=np.float32),
                    "table": np.asarray([[0.7, 0.7, 0.0, 0.0]], dtype=np.float32),
                    "visual": np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
                },
            )
            engine = HybridSearchEngine(
                index_dir,
                FixedQueryEmbedder([1.0, 0.0, 0.0, 0.0]),
            )

            response = engine.search(
                RetrievalRequest(" query ", top_k=3, candidate_k=3)
            )

            self.assertEqual(response.query_text, "query")
            self.assertEqual(response.snapshot_build_id, "build-a")
            self.assertEqual(response.searched_modalities, ["text", "table", "visual"])
            self.assertEqual([hit.rank for hit in response.hits], [1, 2, 3])
            self.assertEqual(response.hits[0].modality, "visual")
            visual_path = Path(response.hits[0].asset_path or "")
            self.assertTrue(visual_path.is_file())
            self.assertTrue(visual_path.is_relative_to(snapshot.resolve()))
            self.assertEqual(response.hits[0].provenance["page"], 1)
            evidence = response.hits[0].evidence_blocks
            self.assertEqual(
                {block["modality"] for block in evidence},
                {"text", "table", "visual"},
            )
            visual_blocks = [block for block in evidence if block["modality"] == "visual"]
            self.assertTrue(all(Path(block["asset_path"]).is_file() for block in visual_blocks))
            self.assertTrue(any(block["role"] == "core" for block in evidence))
            self.assertGreaterEqual(response.elapsed_ms, 0)
            json.dumps(response.to_dict())

    def test_response_includes_build_and_empty_modality_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            _publish_snapshot(
                index_dir,
                "build-a",
                counts={"text": 1, "table": 0, "visual": 0},
                manifest_warnings=["source warning", "source warning", "  "],
            )
            response = HybridSearchEngine(
                index_dir,
                FixedQueryEmbedder([1.0, 0.0, 0.0, 0.0]),
            ).search(RetrievalRequest("query"))

            self.assertEqual(
                response.warnings,
                ["source warning", "table index is empty", "visual index is empty"],
            )

    def test_visual_asset_path_rejects_escape_absolute_and_missing_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            _publish_snapshot(index_dir, "build-a")
            snapshot = SnapshotStore(index_dir).get_snapshot()

            for value in ("../escape.png", str(Path(temporary).resolve() / "absolute.png")):
                with self.subTest(value=value):
                    with self.assertRaisesRegex(RuntimeError, "unsafe|escapes"):
                        HybridSearchEngine._visual_asset_path(snapshot, value)
            with self.assertRaisesRegex(RuntimeError, "no asset_path"):
                HybridSearchEngine._visual_asset_path(snapshot, None)
            with self.assertRaisesRegex(RuntimeError, "missing"):
                HybridSearchEngine._visual_asset_path(snapshot, "missing.png")

    def test_external_embedder_is_not_closed_and_closed_engine_rejects_search(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            _publish_snapshot(index_dir, "build-a")

            class ExternalEmbedder(FixedQueryEmbedder):
                def __init__(self) -> None:
                    super().__init__([1.0, 0.0, 0.0, 0.0])
                    self.close_calls = 0

                def close(self) -> None:
                    self.close_calls += 1

            embedder = ExternalEmbedder()
            engine = HybridSearchEngine(index_dir, embedder)
            engine.close()
            engine.close()

            self.assertEqual(embedder.close_calls, 0)
            with self.assertRaisesRegex(RuntimeError, "closed"):
                engine.search(RetrievalRequest("query"))
            with self.assertRaisesRegex(RuntimeError, "closed"):
                engine.__enter__()

    def test_owned_pixelrag_session_is_closed_once(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            _publish_snapshot(index_dir, "build-a")

            class Session:
                def __init__(self, model_name, device) -> None:
                    self.model_name = model_name
                    self.device = device
                    self.close_calls = 0

                def embed_texts(self, texts, instruction=""):
                    return np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

                def close(self) -> None:
                    self.close_calls += 1

            with patch("hybrid_input.vision.PixelRAGEmbeddingSession", Session):
                engine = HybridSearchEngine(index_dir, device="cpu")
                session = engine._owned_session
                with engine:
                    response = engine.search(RetrievalRequest("query"))
                engine.close()

            self.assertEqual(response.snapshot_build_id, "build-a")
            self.assertEqual(session.device, "cpu")
            self.assertEqual(session.close_calls, 1)

    def test_legacy_indexing_import_constructs_the_public_engine(self) -> None:
        from hybrid_input.indexing import HybridSearchEngine as CompatibilityEngine

        with tempfile.TemporaryDirectory() as temporary:
            index_dir = Path(temporary) / "index"
            _publish_snapshot(index_dir, "build-a")
            engine = CompatibilityEngine(
                index_dir,
                FixedQueryEmbedder([1.0, 0.0, 0.0, 0.0]),
            )
            self.assertIsInstance(engine, HybridSearchEngine)


if __name__ == "__main__":
    unittest.main()
