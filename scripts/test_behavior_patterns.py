import json
from pathlib import Path

from src.reasoning.pattern_engine import (
    get_activated_patterns,
    infer_behavior_patterns,
    load_json,
)


def main() -> None:
    actions_path = Path(
        "outputs/semantic/actions_case_0.json"
    )

    patterns_path = Path(
        "config/reasoning/behavior_patterns_nslkdd.json"
    )

    output_path = Path(
        "outputs/reasoning/patterns_case_0.json"
    )

    actions_payload = load_json(actions_path)
    patterns_config = load_json(patterns_path)

    prediction = actions_payload.get(
        "prediction",
        {},
    )

    prediction_label = prediction.get("label")

    semantic_actions = actions_payload.get(
        "semantic_actions",
        [],
    )

    pattern_results = infer_behavior_patterns(
        semantic_actions=semantic_actions,
        patterns_config=patterns_config,
        prediction_label=prediction_label,
    )

    activated_patterns = get_activated_patterns(
        pattern_results
    )

    result = {
        "prediction": prediction,
        "semantic_action_count": len(
            semantic_actions
        ),
        "evaluated_patterns": pattern_results,
        "activated_patterns": activated_patterns,
        "summary": {
            "evaluated_pattern_count": len(
                pattern_results
            ),
            "activated_pattern_count": len(
                activated_patterns
            ),
        },
    }

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=4,
            ensure_ascii=False,
        )

    print("=== Semantic Action → Behavioral Pattern ===")
    print(
        f"Prédiction : "
        f"{prediction_label} "
        f"(p={float(prediction.get('probability', 0.0)):.4f})"
    )
    print(f"Actions analysées : {len(semantic_actions)}")
    print(f"Motifs évalués    : {len(pattern_results)}")
    print(f"Motifs activés    : {len(activated_patterns)}")
    print()

    if not activated_patterns:
        print("Aucun motif comportemental activé.")

    for index, pattern in enumerate(
        activated_patterns,
        start=1,
    ):
        print(
            f"{index}. {pattern['pattern_id']} "
            f"→ {pattern['name']}"
        )

        print(
            f"   catégorie={pattern['category']} "
            f"| sévérité={pattern['severity']}"
        )

        print(
            f"   requis="
            f"{pattern['matched_required_count']}/"
            f"{pattern['required_condition_count']} "
            f"| support="
            f"{pattern['matched_supporting_count']}/"
            f"{pattern['supporting_condition_count']}"
        )

        print(
            f"   poids total des preuves="
            f"{pattern['total_evidence_weight']:.4f}"
        )

        print(
            f"   cohérent avec la classe "
            f"{prediction_label} : "
            f"{pattern['label_consistent']}"
        )

        print("   Preuves requises :")

        for evidence in pattern["required_evidence"]:
            action = evidence["matched_action"]

            print(
                f"   - {action['verb']}"
                f"({action['object']}) "
                f"| état={action['state']} "
                f"| poids={action['weight']:.4f}"
            )

            print(
                f"     feature={action['source_feature']} "
                f"| valeur={action['source_value']} "
                f"| shap={action['signed_contribution']:.4f}"
            )

        if pattern["supporting_evidence"]:
            print("   Preuves complémentaires :")

            for evidence in pattern["supporting_evidence"]:
                action = evidence["matched_action"]

                print(
                    f"   - {action['verb']}"
                    f"({action['object']}) "
                    f"| état={action['state']} "
                    f"| poids={action['weight']:.4f}"
                )

    print()
    print(
        f"Résultat enregistré : "
        f"{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()