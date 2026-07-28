import json

from src.reasoning.scallop_backend import ScallopReasoningBackend
from src.knowledge.knowledge_base import KnowledgeBase


def load_actions(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)["semantic_actions"]


def test_probe_pipeline():

    backend = ScallopReasoningBackend()

    result = backend.reason(
        case_id="probe",
        actions=load_actions(
            "outputs/semantic/actions_probe_01.json"
        ),
        patterns=[]
    )

    ids = {
        p["pattern_id"]
        for p in result["activated_patterns"]
    }

    assert "BP-SCAN-001" in ids


def test_dos_pipeline():

    backend = ScallopReasoningBackend()

    result = backend.reason(
        case_id="dos",
        actions=load_actions(
            "outputs/semantic/actions_dos_01.json"
        ),
        patterns=[]
    )

    ids = {
        p["pattern_id"]
        for p in result["activated_patterns"]
    }

    assert "BP-DOS-001" in ids