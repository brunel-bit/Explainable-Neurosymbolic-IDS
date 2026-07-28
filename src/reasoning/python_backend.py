from __future__ import annotations

from typing import Any

from src.reasoning.base import ReasoningBackend


class PythonReasoningBackend(ReasoningBackend):
    """
    Adaptateur autour du moteur de règles Python existant.

    Cette première version reste volontairement générique.
    Nous connecterons ensuite le pattern_engine existant.
    """

    name = "python"

    def __init__(self, pattern_engine: Any | None = None) -> None:
        self.pattern_engine = pattern_engine

    def reason(
        self,
        *,
        case_id: str,
        actions: list[dict[str, Any]],
        patterns: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        activated_patterns: list[dict[str, Any]] = []

        if self.pattern_engine is not None:
            activated_patterns = self.pattern_engine.evaluate(
                actions=actions,
                patterns=patterns,
            )

        return {
            "case_id": case_id,
            "backend": self.name,
            "activated_patterns": activated_patterns,
            "derived_relations": [],
            "attack_chains": [],
            "context": context or {},
            "trace": {
                "engine": self.name,
                "status": "completed",
            },
        }
