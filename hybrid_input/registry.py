from __future__ import annotations

from collections.abc import Iterable

from .interfaces import DocumentParser


class ParserRegistry:
    """Small registry whose only coupling is the DocumentParser interface."""

    def __init__(self, parsers: Iterable[DocumentParser] = ()) -> None:
        self._parsers: dict[str, DocumentParser] = {}
        for parser in parsers:
            self.register(parser)

    def register(self, parser: DocumentParser) -> None:
        if parser.name in self._parsers:
            raise ValueError(f"Parser already registered: {parser.name}")
        self._parsers[parser.name] = parser

    def get(self, name: str) -> DocumentParser:
        try:
            return self._parsers[name]
        except KeyError as exc:
            choices = ", ".join(sorted(self._parsers)) or "<none>"
            raise KeyError(f"Unknown parser {name!r}; available: {choices}") from exc

    def names(self) -> list[str]:
        return sorted(self._parsers)
