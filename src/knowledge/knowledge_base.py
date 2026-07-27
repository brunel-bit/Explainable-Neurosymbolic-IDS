from __future__ import annotations

import json
from pathlib import Path


class KnowledgeBase:
    def __init__(self):
        self.root = Path("knowledge_base")

    def load_patterns(self):
        with open(
            self.root / "patterns" / "nslkdd.json",
            encoding="utf-8",
        ) as f:
            return json.load(f)["patterns"]

    def load_actions(self, json_file):
        with open(json_file, encoding="utf-8") as f:
            return json.load(f)["semantic_actions"]

    def load_concepts(self):
        with open(
            self.root / "concepts" / "nslkdd.json",
            encoding="utf-8",
        ) as f:
            return json.load(f)