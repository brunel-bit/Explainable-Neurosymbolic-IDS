import json
from pathlib import Path

from src.semantic.mapper import (
    load_xai_json,
    xai_to_semantic_actions,
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

    config_path = Path(
        "config/comat/parser_config_etape18a.json"
    )

    output_path = Path(
        "outputs/semantic/semantic_actions_case_0.json"
    )

    print("=== Test du Semantic Mapper ===")
    print(f"XAI       : {xai_path.resolve()}")
    print(f"Config    : {config_path.resolve()}")

    xai_payload = load_xai_json(xai_path)
    config = load_json(config_path)

    rules = (
        config.get("semantic_mapping", {})
        .get("feature_pattern_rules", [])
    )

    print(f"Features  : {len(xai_payload.get('features', []))}")
    print(f"Règles    : {len(rules)}")

    actions = xai_to_semantic_actions(
        xai_payload=xai_payload,
        config=config,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "prediction": xai_payload.get("prediction", {}),
        "semantic_actions_from_xai": actions,
    }

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

    print(f"Actions   : {len(actions)}")
    print()

    if not actions:
        print(
            "Aucune action sémantique produite. "
            "Les noms des features SHAP ne correspondent "
            "probablement à aucune règle de configuration."
        )
    else:
        print("Actions sémantiques produites :")

        for index, action in enumerate(actions, start=1):
            print(
                f"{index}. {action['verb']}"
                f"({action['object']}) "
                f"— poids={action['weight']}"
            )

            for evidence in action["evidence"]:
                print(
                    f"   source={evidence['feature']}, "
                    f"value={evidence['value']}, "
                    f"weight={evidence['weight']}"
                )

    print()
    print(f"Résultat enregistré : {output_path.resolve()}")


if __name__ == "__main__":
    main()