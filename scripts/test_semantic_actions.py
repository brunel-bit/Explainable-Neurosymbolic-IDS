import json
from pathlib import Path

from src.semantic.mapper import (
    load_xai_json,
    xai_to_semantic_concepts,
    semantic_concepts_to_actions,
)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    xai_path = Path(
        "outputs/comat/input_case_0.json"
    )

    concepts_path = Path(
        "config/semantic/concepts_nslkdd.json"
    )

    actions_path = Path(
        "config/semantic/actions_nslkdd.json"
    )

    output_path = Path(
        "outputs/semantic/actions_case_0.json"
    )

    xai_payload = load_xai_json(xai_path)
    concepts_config = load_json(concepts_path)
    actions_config = load_json(actions_path)

    concepts = xai_to_semantic_concepts(
        xai_payload=xai_payload,
        concepts_config=concepts_config,
    )

    actions = semantic_concepts_to_actions(
        semantic_concepts=concepts,
        actions_config=actions_config,
    )

    result = {
        "prediction": xai_payload.get(
            "prediction",
            {},
        ),
        "semantic_concepts": concepts,
        "semantic_actions": actions,
        "summary": {
            "concept_count": len(concepts),
            "action_count": len(actions),
            "supporting_actions": sum(
                1 for action in actions
                if action["supports_prediction"]
            ),
            "opposing_actions": sum(
                1 for action in actions
                if action["direction"]
                == "opposes_prediction"
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

    prediction = result["prediction"]

    print("=== Concept → Semantic Action ===")
    print(
        f"Prédiction : "
        f"{prediction.get('label')} "
        f"(p={prediction.get('probability', 0):.4f})"
    )
    print(f"Concepts   : {len(concepts)}")
    print(f"Actions    : {len(actions)}")
    print()

    for index, action in enumerate(
        actions,
        start=1,
    ):
        print(
            f"{index}. "
            f"{action['verb']}"
            f"({action['object']}) "
            f"| état={action['state']} "
            f"| poids={action['weight']:.4f}"
        )

        print(
            f"   concept={action['concept']} "
            f"| feature={action['source_feature']} "
            f"| valeur={action['source_value']}"
        )

        print(
            f"   direction={action['direction']} "
            f"| shap={action['signed_contribution']:.4f}"
        )

    print()
    print(
        f"Résultat enregistré : "
        f"{output_path.resolve()}"
    )


if __name__ == "__main__":
    main()