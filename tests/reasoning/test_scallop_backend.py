from src.reasoning.scallop_backend import ScallopReasoningBackend


SCAN_PATTERN = {
    "pattern_id": "BP-SCAN-001",
    "name": "icmp_host_exploration_pattern",
    "description": "ICMP host exploration.",
    "required_conditions": [
        {
            "verb": "observe",
            "object": "icmp_protocol_activity",
            "state": "present",
        },
        {
            "verb": "observe",
            "object": "icmp_echo_activity",
            "state": "present",
        },
    ],
    "supporting_conditions": [
        {
            "verb": "increase",
            "object": "source_port_reuse",
            "state": "high",
        },
        {
            "verb": "increase",
            "object": "multi_host_service_activity",
            "state": "elevated",
        },
    ],
    "minimum_supporting_conditions": 1,
    "associated_labels": ["Probe"],
    "category": "network_discovery",
    "severity": "medium",
}


def test_scallop_activates_complete_scan_pattern() -> None:
    backend = ScallopReasoningBackend()

    result = backend.reason(
        case_id="probe_01",
        actions=[
            {
                "verb": "observe",
                "object": "icmp_protocol_activity",
                "state": "present",
            },
            {
                "verb": "observe",
                "object": "icmp_echo_activity",
                "state": "present",
            },
            {
                "verb": "increase",
                "object": "source_port_reuse",
                "state": "high",
            },
        ],
        patterns=[SCAN_PATTERN],
    )

    assert result["backend"] == "scallop"
    assert result["trace"]["native_execution"] is True
    assert result["trace"]["status"] == "completed"
    assert result["trace"]["activated_pattern_count"] == 1

    assert len(result["activated_patterns"]) == 1

    activated = result["activated_patterns"][0]

    assert activated["pattern_id"] == "BP-SCAN-001"
    assert activated["matched_required_conditions"] == 2
    assert activated["matched_supporting_conditions"] == 1
    assert activated["required_coverage"] == 1.0


def test_scallop_rejects_pattern_with_missing_required_condition() -> None:
    backend = ScallopReasoningBackend()

    result = backend.reason(
        case_id="probe_incomplete",
        actions=[
            {
                "verb": "observe",
                "object": "icmp_protocol_activity",
                "state": "present",
            },
            {
                "verb": "increase",
                "object": "source_port_reuse",
                "state": "high",
            },
        ],
        patterns=[SCAN_PATTERN],
    )

    assert result["activated_patterns"] == []
    assert result["trace"]["activated_pattern_count"] == 0


def test_scallop_rejects_pattern_without_minimum_support() -> None:
    backend = ScallopReasoningBackend()

    result = backend.reason(
        case_id="probe_without_support",
        actions=[
            {
                "verb": "observe",
                "object": "icmp_protocol_activity",
                "state": "present",
            },
            {
                "verb": "observe",
                "object": "icmp_echo_activity",
                "state": "present",
            },
        ],
        patterns=[SCAN_PATTERN],
    )

    assert result["activated_patterns"] == []
    assert result["trace"]["activated_pattern_count"] == 0