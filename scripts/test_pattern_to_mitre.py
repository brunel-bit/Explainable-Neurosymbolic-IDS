import json
from pathlib import Path
from typing import Any

from src.reasoning.mitre_mapper import (
    load_json,
    map_patterns_to_mitre,
)


def format_probability(
    prediction: dict[str, Any],
) -> str:
    """Format the prediction probability safely."""
    probability = prediction.get("probability")

    if probability is None:
        return "indisponible"

    return f"{float(probability):.4f}"


def main() -> None:
    patterns_path = Path(
        "outputs/reasoning/patterns_case_0.json"
    )

    mitre_config_path = Path(
        "config/mitre/pattern_to_mitre.json"
    )

    output_path = Path(
        "outputs/mitre/mitre_case_0.json"
    )

    patterns_payload = load_json(patterns_path)
    mitre_config = load_json(mitre_config_path)

    prediction = patterns_payload.get(
        "prediction",
        {},
    )

    activated_patterns = patterns_payload.get(
        "activated_patterns",
        [],
    )

    mapping_result = map_patterns_to_mitre(
        activated_patterns=activated_patterns,
        mitre_config=mitre_config,
    )

    result = {
        "prediction": prediction,
        "activated_patterns": [
            {
                "pattern_id": pattern.get("pattern_id"),
                "name": pattern.get("name"),
                "category": pattern.get("category"),
                "severity": pattern.get("severity"),
                "total_evidence_weight": pattern.get(
                    "total_evidence_weight"
                ),
            }
            for pattern in activated_patterns
        ],
        **mapping_result,
        "metadata": {
            "mapping_version": mitre_config.get(
                "version"
            ),
            "mapping_source": mitre_config.get(
                "source",
                {},
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

    prediction_label = prediction.get(
        "label",
        "inconnue",
    )

    print(
        "=== Behavioral Pattern → MITRE ATT&CK ==="
    )

    print(
        f"Prédiction : {prediction_label} "
        f"(p={format_probability(prediction)})"
    )

    print(
        f"Motifs activés   : "
        f"{len(activated_patterns)}"
    )

    print(
        f"Candidats MITRE  : "
        f"{mapping_result['summary']['candidate_count']}"
    )

    print(
        f"Motifs non mappés: "
        f"{mapping_result['summary']['unmapped_pattern_count']}"
    )

    print()

    candidates = mapping_result["mitre_candidates"]

    if not candidates:
        print("Aucun candidat MITRE identifié.")

    for index, candidate in enumerate(
        candidates,
        start=1,
    ):
        print(
            f"{index}. {candidate['technique_id']} "
            f"→ {candidate['technique_name']}"
        )

        print(
            f"   tactique={candidate['tactic_id']} "
            f"({candidate['tactic_name']})"
        )

        print(
            f"   motif={candidate['pattern_id']} "
            f"→ {candidate['pattern_name']}"
        )

        print(
            f"   confiance de la règle="
            f"{candidate['base_confidence']:.2f}"
        )

        print(
            f"   poids des preuves="
            f"{candidate['pattern_evidence_weight']:.4f}"
        )

        print(
            f"   couverture requise="
            f"{candidate['required_coverage']:.2f} "
            f"| complémentaire="
            f"{candidate['supporting_coverage']:.2f}"
        )

        print(
            f"   cohérence avec la prédiction="
            f"{candidate['label_consistent']}"
        )

        print(
            f"   granularité="
            f"{candidate['granularity']}"
        )

        print(
            f"   justification : "
            f"{candidate['justification']}"
        )

        print("   Traçabilité des preuves :")

        for evidence in candidate["evidence"]:
            print(
                f"   - {evidence['verb']}"
                f"({evidence['object']}) "
                f"| état={evidence['state']} "
                f"| poids={evidence['action_weight']:.4f}"
            )

            print(
                f"     feature={evidence['source_feature']} "
                f"| valeur={evidence['source_value']} "
                f"| shap="
                f"{evidence['signed_contribution']:.4f}"
            )

        excluded = candidate.get(
            "excluded_subtechniques",
            [],
        )

        if excluded:
            print(
                "   Sous-techniques non affirmées :"
            )

            for subtechnique in excluded:
                print(
                    f"   - {subtechnique['technique_id']} "
                    f"{subtechnique['technique_name']}"
                )

                print(
                    f"     raison : "
                    f"{subtechnique['reason']}"
                )

        print()

    for unmapped in mapping_result[
        "unmapped_patterns"
    ]:
        print(
            f"Motif non mappé : "
            f"{unmapped['pattern_id']} "
            f"→ {unmapped['pattern_name']}"
        )

        print(
            f"Raison : {unmapped['reason']}"
        )

    print(
        f"Résultat enregistré : "
        f"{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()