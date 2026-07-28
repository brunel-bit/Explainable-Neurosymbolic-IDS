from src.mitre.mapper import MitreMapper


def test_mapper():

    mapper = MitreMapper()

    activated = [
        {
            "pattern_id": "BP-SCAN-001",
            "required_coverage": 1.0,
        }
    ]

    result = mapper.map_patterns(activated)

    assert result[0]["technique_id"] == "T1046"