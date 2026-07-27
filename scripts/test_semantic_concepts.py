import json
from pathlib import Path

from src.semantic.mapper import (
    load_xai_json,
    xai_to_semantic_concepts,
)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Fichier introuvable : {path.resolve()}"
        )

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    xai_path = Path("outputs/comat/input_case_0.json")
    concepts_path = Path(
        "config/semantic/concepts_nslkdd.json"
    )
    output_path = Path(
        "outputs/semantic/concepts_case_0.json"
    )

    xai_payload = load_xai_json(xai_path)
    concepts_config = load_json(concepts_path)

    concepts = xai_to_semantic_concepts(
        xai_payload=xai_payload,
        concepts_config=concepts_config,
    )

    mapped_count = sum(
        1 for concept in concepts
        if concept["mapped"]
    )

    result = {
        "prediction": xai_payload.get("prediction", {}),
        "dataset": concepts_config.get("dataset"),
        "semantic_concepts": concepts,
        "coverage": {
            "mapped_features": mapped_count,
            "total_features": len(concepts),
            "mapping_rate": (
                round(mapped_count / len(concepts), 4)
                if concepts else 0.0
            )
        }
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

    print("=== Feature → Semantic Concept ===")
    print(
        f"Prédiction : "
        f"{result['prediction'].get('label')}"
    )
    print(
        f"Couverture : {mapped_count}/{len(concepts)} "
        f"({result['coverage']['mapping_rate']:.0%})"
    )
    print()

    for item in concepts:
        if item["mapped"]:
            print(
                f"✓ {item['feature']} "
                f"→ {item['concept']} "
                f"| valeur={item['value']} "
                f"| contribution={item['contribution']:.4f}"
            )
        else:
            print(
                f"✗ {item['feature']} "
                "→ concept non défini"
            )

    print()
    print(
        f"Résultat enregistré : {output_path.resolve()}"
    )


if __name__ == "__main__":
    main()