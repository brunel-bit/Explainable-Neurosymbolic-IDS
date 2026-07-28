from __future__ import annotations

from typing import Any

from src.reasoning.base import ReasoningBackend


class ScallopReasoningBackend(ReasoningBackend):
    """
    Backend déclaratif fondé sur scallopy.

    Il transforme les actions et motifs en faits Scallop,
    exécute les règles SCL et retourne les relations dérivées.
    """

    name = "scallop"

    def __init__(self) -> None:
        try:
            import scallopy
        except ImportError as exc:
            raise RuntimeError(
                "Scallop n'est pas disponible. "
                "Installe scallopy ou utilise le backend Python."
            ) from exc

        self.scallopy = scallopy

    def reason(
        self,
        *,
        case_id: str,
        actions: list[dict[str, Any]],
        patterns: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ctx = self.scallopy.ScallopContext()

        ctx.add_relation(
            "semantic_action",
            (str, str, str, str),
        )

        ctx.add_relation(
            "candidate_pattern",
            (str, str),
        )

        ctx.add_relation(
            "pattern_support",
            (str, str),
        )

        action_facts = []

        for action in actions:
            action_facts.append(
                (
                    case_id,
                    str(action.get("verb", "")),
                    str(action.get("object", "")),
                    str(action.get("state", "")),
                )
            )

        pattern_facts = []

        for pattern in patterns:
            pattern_facts.append(
                (
                    str(pattern.get("pattern_id", "")),
                    str(pattern.get("name", "")),
                )
            )

        if action_facts:
            ctx.add_facts(
                "semantic_action",
                action_facts,
            )

        if pattern_facts:
            ctx.add_facts(
                "candidate_pattern",
                pattern_facts,
            )

        ctx.add_rule(
            """
            pattern_support(pattern_id, case_id) =
                candidate_pattern(pattern_id, _),
                semantic_action(case_id, _, _, _)
            """
        )

        ctx.run()

        derived_relations = []

        try:
            for row in ctx.relation("pattern_support"):
                derived_relations.append(
                    {
                        "pattern_id": row[0],
                        "case_id": row[1],
                    }
                )
        except Exception:
            derived_relations = []

        return {
            "case_id": case_id,
            "backend": self.name,
            "activated_patterns": [],
            "derived_relations": derived_relations,
            "attack_chains": [],
            "context": context or {},
            "trace": {
                "engine": self.name,
                "status": "completed",
                "native_execution": True,
            },
        }
