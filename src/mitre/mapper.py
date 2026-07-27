from __future__ import annotations

import json
from pathlib import Path


class MitreMapper:

    def __init__(self):
        self.mapping = self._load_mapping()

    def _load_mapping(self):

        path = Path("config/mitre/pattern_to_mitre.json")

        with open(path, encoding="utf-8") as f:
            data = json.load(f)

        mapping = {}

        for item in data["mappings"]:
            mapping[item["pattern_id"]] = item

        return mapping

    def map_patterns(self, activated_patterns):

        results = []

        for pattern in activated_patterns:

            pattern_id = pattern["pattern_id"]

            if pattern_id not in self.mapping:
                continue

            mitre = self.mapping[pattern_id]

            results.append(
                {
                    "pattern_id": pattern_id,
                    "technique_id": mitre["technique_id"],
                    "technique_name": mitre["technique_name"],
                    "tactic_id": mitre["tactic_id"],
                    "tactic_name": mitre["tactic_name"],
                    "confidence": mitre["base_confidence"],
                    "mapping_status": mitre["mapping_status"],
                    "justification": mitre["justification"],
                    "supporting_behavior": mitre["supporting_behavior"],
                    "required_additional_context":
                        mitre["required_additional_context"],
                }
            )

        return results