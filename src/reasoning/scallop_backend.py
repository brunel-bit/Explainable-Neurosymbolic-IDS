from __future__ import annotations
from src.knowledge.knowledge_base import KnowledgeBase
from typing import Any

from src.reasoning.base import ReasoningBackend


class ScallopReasoningBackend(ReasoningBackend):
    """
    Backend de raisonnement déclaratif utilisant Scallopy.

    Scallop établit les correspondances entre les actions observées
    et les conditions des motifs comportementaux.

    L'activation finale d'un motif exige :
    1. la satisfaction de toutes ses conditions obligatoires ;
    2. la satisfaction du nombre minimal de conditions de soutien.
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
        self.kb = KnowledgeBase()

    def reason(
        self,
        *,
        case_id: str,
        actions: list[dict[str, Any]],
        patterns: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ctx = self.scallopy.ScallopContext()

        self._declare_relations(ctx)

        # Charger automatiquement les patterns si aucun n'est fourni
        if not patterns:
            patterns = self.kb.load_patterns()

        # Construire les faits APRÈS avoir chargé les patterns
        action_facts = self._build_action_facts(
            case_id=case_id,
            actions=actions,
        )

        (
            pattern_facts,
            required_condition_facts,
            supporting_condition_facts,
        ) = self._build_pattern_facts(patterns)

        print("\n========== DEBUG ==========")
        print(f"Nombre de patterns : {len(patterns)}")
        print(f"Pattern facts : {len(pattern_facts)}")
        print(f"Required facts : {len(required_condition_facts)}")
        print(f"Supporting facts : {len(supporting_condition_facts)}")
        print(f"Action facts : {len(action_facts)}")
        print("===========================\n")

        if action_facts:
            ctx.add_facts("semantic_action", action_facts)

        if pattern_facts:
            ctx.add_facts("candidate_pattern", pattern_facts)

        if required_condition_facts:
            ctx.add_facts("required_condition", required_condition_facts)

        if supporting_condition_facts:
            ctx.add_facts("supporting_condition", supporting_condition_facts)

        self._add_rules(ctx)
        ctx.run()

        required_matches = self._read_matches(
            ctx=ctx,
            relation_name="required_match",
        )

        supporting_matches = self._read_matches(
            ctx=ctx,
            relation_name="supporting_match",
        )

        activated_patterns = []
        derived_relations = []
        rules_fired = []

        for pattern in patterns:
            evaluation = self._evaluate_pattern(
                case_id=case_id,
                pattern=pattern,
                required_matches=required_matches,
                supporting_matches=supporting_matches,
            )

            derived_relations.extend(
                evaluation["derived_relations"]
            )

            if evaluation["activated_pattern"] is not None:
                activated_patterns.append(
                    evaluation["activated_pattern"]
                )
                rules_fired.append(
                    f"activate_{pattern.get('pattern_id', 'unknown')}"
                )

        return {
            "case_id": case_id,
            "backend": self.name,
            "activated_patterns": activated_patterns,
            "derived_relations": derived_relations,
            "attack_chains": [],
            "context": context or {},
            "trace": {
                "engine": self.name,
                "status": "completed",
                "native_execution": True,
                "rules_fired": rules_fired,
                "evaluated_patterns": len(patterns),
                "activated_pattern_count": len(activated_patterns),
            },
        }

    @staticmethod
    def _declare_relations(ctx: Any) -> None:
        ctx.add_relation(
            "semantic_action",
            (str, str, str, str),
        )

        ctx.add_relation(
            "candidate_pattern",
            (str, str),
        )

        ctx.add_relation(
            "required_condition",
            (str, str, str, str, str),
        )

        ctx.add_relation(
            "supporting_condition",
            (str, str, str, str, str),
        )

        ctx.add_relation(
            "required_match",
            (str, str, str),
        )

        ctx.add_relation(
            "supporting_match",
            (str, str, str),
        )

    @staticmethod
    def _add_rules(ctx: Any) -> None:
        ctx.add_rule(
            """
            required_match(pattern_id, case_id, condition_id) =
                candidate_pattern(pattern_id, _),
                required_condition(
                    pattern_id,
                    condition_id,
                    verb,
                    object,
                    state
                ),
                semantic_action(
                    case_id,
                    verb,
                    object,
                    state
                )
            """
        )

        ctx.add_rule(
            """
            supporting_match(pattern_id, case_id, condition_id) =
                candidate_pattern(pattern_id, _),
                supporting_condition(
                    pattern_id,
                    condition_id,
                    verb,
                    object,
                    state
                ),
                semantic_action(
                    case_id,
                    verb,
                    object,
                    state
                )
            """
        )

    @staticmethod
    def _build_action_facts(
        *,
        case_id: str,
        actions: list[dict[str, Any]],
    ) -> list[tuple[str, str, str, str]]:
        facts = []

        for action in actions:
            verb = str(action.get("verb", "")).strip()
            object_name = str(action.get("object", "")).strip()
            state = str(action.get("state", "")).strip()

            if not verb or not object_name or not state:
                continue

            facts.append(
                (
                    case_id,
                    verb,
                    object_name,
                    state,
                )
            )

        return facts

    @staticmethod
    def _build_pattern_facts(
        patterns: list[dict[str, Any]],
    ) -> tuple[
        list[tuple[str, str]],
        list[tuple[str, str, str, str, str]],
        list[tuple[str, str, str, str, str]],
    ]:
        pattern_facts = []
        required_facts = []
        supporting_facts = []

        for pattern in patterns:
            pattern_id = str(
                pattern.get("pattern_id", "")
            ).strip()

            pattern_name = str(
                pattern.get("name", "")
            ).strip()

            if not pattern_id:
                continue

            pattern_facts.append(
                (
                    pattern_id,
                    pattern_name,
                )
            )

            required_conditions = pattern.get(
                "required_conditions",
                [],
            )

            for index, condition in enumerate(
                required_conditions
            ):
                required_facts.append(
                    (
                        pattern_id,
                        f"required_{index}",
                        str(condition.get("verb", "")).strip(),
                        str(condition.get("object", "")).strip(),
                        str(condition.get("state", "")).strip(),
                    )
                )

            supporting_conditions = pattern.get(
                "supporting_conditions",
                [],
            )

            for index, condition in enumerate(
                supporting_conditions
            ):
                supporting_facts.append(
                    (
                        pattern_id,
                        f"supporting_{index}",
                        str(condition.get("verb", "")).strip(),
                        str(condition.get("object", "")).strip(),
                        str(condition.get("state", "")).strip(),
                    )
                )

        return (
            pattern_facts,
            required_facts,
            supporting_facts,
        )

    @staticmethod
    def _read_matches(
        *,
        ctx: Any,
        relation_name: str,
    ) -> set[tuple[str, str, str]]:
        matches: set[tuple[str, str, str]] = set()

        try:
            for row in ctx.relation(relation_name):
                matches.add(
                    (
                        str(row[0]),
                        str(row[1]),
                        str(row[2]),
                    )
                )
        except Exception:
            return set()

        return matches

    @staticmethod
    def _evaluate_pattern(
        *,
        case_id: str,
        pattern: dict[str, Any],
        required_matches: set[tuple[str, str, str]],
        supporting_matches: set[tuple[str, str, str]],
    ) -> dict[str, Any]:
        pattern_id = str(pattern.get("pattern_id", ""))

        required_conditions = pattern.get(
            "required_conditions",
            [],
        )

        supporting_conditions = pattern.get(
            "supporting_conditions",
            [],
        )

        minimum_support = int(
            pattern.get(
                "minimum_supporting_conditions",
                0,
            )
        )

        matched_required_ids = [
            f"required_{index}"
            for index in range(len(required_conditions))
            if (
                pattern_id,
                case_id,
                f"required_{index}",
            )
            in required_matches
        ]

        matched_supporting_ids = [
            f"supporting_{index}"
            for index in range(len(supporting_conditions))
            if (
                pattern_id,
                case_id,
                f"supporting_{index}",
            )
            in supporting_matches
        ]

        required_count = len(required_conditions)
        supporting_count = len(supporting_conditions)

        all_required_matched = (
            required_count > 0
            and len(matched_required_ids) == required_count
        )

        enough_support = (
            len(matched_supporting_ids) >= minimum_support
        )

        activated = all_required_matched and enough_support

        derived_relations = []

        for condition_id in matched_required_ids:
            derived_relations.append(
                {
                    "relation": "required_condition_matched",
                    "pattern_id": pattern_id,
                    "case_id": case_id,
                    "condition_id": condition_id,
                }
            )

        for condition_id in matched_supporting_ids:
            derived_relations.append(
                {
                    "relation": "supporting_condition_matched",
                    "pattern_id": pattern_id,
                    "case_id": case_id,
                    "condition_id": condition_id,
                }
            )

        activated_pattern = None

        if activated:
            activated_pattern = {
                "pattern_id": pattern_id,
                "name": pattern.get("name", ""),
                "description": pattern.get("description", ""),
                "category": pattern.get("category", ""),
                "severity": pattern.get("severity", ""),
                "associated_labels": pattern.get(
                    "associated_labels",
                    [],
                ),
                "matched_required_conditions": len(
                    matched_required_ids
                ),
                "total_required_conditions": required_count,
                "matched_supporting_conditions": len(
                    matched_supporting_ids
                ),
                "total_supporting_conditions": supporting_count,
                "minimum_supporting_conditions": minimum_support,
                "required_coverage": (
                    len(matched_required_ids) / required_count
                    if required_count
                    else 0.0
                ),
                "supporting_coverage": (
                    len(matched_supporting_ids) / supporting_count
                    if supporting_count
                    else 0.0
                ),
            }

        return {
            "activated_pattern": activated_pattern,
            "derived_relations": derived_relations,
        }