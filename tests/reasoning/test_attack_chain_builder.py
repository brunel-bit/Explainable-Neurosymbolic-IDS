from src.reasoning.attack_chain_builder import AttackChainBuilder


def test_chain():

    builder = AttackChainBuilder()

    chain = builder.build(
        [
            {
                "pattern_id": "BP-DOS-001",
                "technique_id": "T1498",
                "technique_name": "Network DoS",
                "tactic_name": "Impact",
                "confidence": 0.90,
            },
            {
                "pattern_id": "BP-SCAN-001",
                "technique_id": "T1046",
                "technique_name": "Network Service Discovery",
                "tactic_name": "Discovery",
                "confidence": 0.85,
            },
        ]
    )

    assert chain[0]["pattern_id"] == "BP-SCAN-001"
    assert chain[1]["pattern_id"] == "BP-DOS-001"