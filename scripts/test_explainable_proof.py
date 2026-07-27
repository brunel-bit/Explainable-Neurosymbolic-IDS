import json
from pathlib import Path

from src.explanation.proof_builder import (
    build_explainable_proof,
    load_json,
)


def main() -> None:
    mitre_path = Path(
        "outputs/mitre/mitre_case_0.json"
    )

    output_path = Path(
        "outputs/proofs/proof_case_0.json"
    )

    mitre_payload = load_json(mitre_path)

    prediction = mitre_payload.get(
        "prediction",
        {},
    )

    mitre_candidates = mitre_payload.get(
        "mitre_candidates",
        [],
    )

    proof = build_explainable_proof(
        prediction=prediction,
        mitre_candidates=mitre_candidates,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            proof,
            file,
            indent=4,
            ensure_ascii=False,
        )

    explanation = proof[
        "natural_language_explanation"
    ]

    audit = proof["audit"]

    conclusion = proof["C_conclusion"]

    print("=== Preuve explicable E=<O,R,J> ===")

    print(
        f"Identifiant : {proof['proof_id']}"
    )

    print(
        f"Prédiction  : "
        f"{conclusion['prediction']['label']} "
        f"(p={conclusion['prediction']['probability']:.4f})"
    )

    print(
        f"Observations: {audit['observation_count']}"
    )

    print(
        f"Règles      : {audit['rule_count']}"
    )

    print(
        f"Candidats   : "
        f"{audit['mitre_candidate_count']}"
    )

    print(
        f"Traçable    : {audit['traceable']}"
    )

    print()
    print("Explication destinée à l’analyste :")
    print(f"- {explanation['summary']}")
    print(f"- {explanation['behavior']}")
    print(f"- {explanation['mitre']}")
    print(f"- {explanation['evidence']}")
    print(f"- {explanation['caution']}")

    print()
    print("Observations principales :")

    for index, observation in enumerate(
        proof["O_observations"][:5],
        start=1,
    ):
        print(
            f"{index}. {observation['feature']}="
            f"{observation['value']} "
            f"| concept={observation['concept']} "
            f"| SHAP="
            f"{observation['signed_contribution']:+.4f}"
        )

        action = observation["semantic_action"]

        print(
            f"   → {action['verb']}"
            f"({action['object']}) "
            f"| état={action['state']}"
        )

    if proof["limitations"]:
        print()
        print("Limites :")

        for limitation in proof["limitations"]:
            print(
                f"- {limitation['message']}"
            )

            for context in limitation.get(
                "required_context",
                [],
            ):
                print(
                    f"  contexte requis : {context}"
                )

            for subtechnique in limitation.get(
                "excluded_subtechniques",
                [],
            ):
                print(
                    f"  non affirmée : "
                    f"{subtechnique['technique_id']} "
                    f"{subtechnique['technique_name']}"
                )

    print()
    print(
        f"Résultat enregistré : "
        f"{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()