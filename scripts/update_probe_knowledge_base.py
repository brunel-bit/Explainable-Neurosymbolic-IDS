from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONCEPTS_PATH = (
    PROJECT_ROOT
    / "config"
    / "semantic"
    / "concepts_nslkdd.json"
)

ACTIONS_PATH = (
    PROJECT_ROOT
    / "config"
    / "semantic"
    / "actions_nslkdd.json"
)

PATTERNS_PATH = (
    PROJECT_ROOT
    / "config"
    / "reasoning"
    / "behavior_patterns_nslkdd.json"
)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(
    path: Path,
    data: dict[str, Any],
) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False,
        )

        file.write("\n")


def update_concepts() -> None:
    data = load_json(CONCEPTS_PATH)

    features = data.setdefault("features", {})

    features.update(
        {
            "protocol_type_icmp": {
                "concept": "icmp_protocol_usage",
                "category": "protocol",
                "definition": (
                    "Indicates whether the analyzed connection "
                    "uses the ICMP protocol."
                ),
                "soc_interpretation": (
                    "Provides protocol context associated with "
                    "ICMP traffic, which may be used during host "
                    "discovery when combined with other evidence."
                ),
                "value_type": "binary",
                "source": (
                    "Derived from the one-hot encoding of the "
                    "NSL-KDD protocol_type feature"
                ),
            },
            "service_eco_i": {
                "concept": "icmp_echo_service_usage",
                "category": "service",
                "definition": (
                    "Indicates whether the connection uses the "
                    "eco_i service associated with ICMP echo traffic."
                ),
                "soc_interpretation": (
                    "Represents ICMP echo activity that may contribute "
                    "to evidence of host discovery when supported by "
                    "additional observations."
                ),
                "value_type": "binary",
                "source": (
                    "Derived from the one-hot encoding of the "
                    "NSL-KDD service feature"
                ),
            },
            "dst_host_srv_diff_host_rate": {
                "concept": "same_service_multi_host_rate",
                "category": "host_based_traffic",
                "definition": (
                    "Proportion of connections to the same service "
                    "that involve different destination hosts."
                ),
                "soc_interpretation": (
                    "Measures whether a service is contacted across "
                    "multiple destination hosts, which may support "
                    "evidence of distributed exploration."
                ),
                "value_type": "rate",
                "source": (
                    "KDD Cup 1999 host-based traffic features"
                ),
            },
            "dst_host_serror_rate": {
                "concept": "destination_syn_error_rate",
                "category": "host_based_traffic",
                "definition": (
                    "Proportion of historical connections to the "
                    "destination host that contain SYN errors."
                ),
                "soc_interpretation": (
                    "Measures unsuccessful SYN-based connection "
                    "activity involving the destination host."
                ),
                "value_type": "rate",
                "source": (
                    "KDD Cup 1999 host-based traffic features"
                ),
            },
            "service_ftp_data": {
                "concept": "ftp_data_service_usage",
                "category": "service",
                "definition": (
                    "Indicates whether the connection uses the "
                    "FTP-DATA service."
                ),
                "soc_interpretation": (
                    "Provides service context about the use or absence "
                    "of an FTP data-channel connection."
                ),
                "value_type": "binary",
                "source": (
                    "Derived from the one-hot encoding of the "
                    "NSL-KDD service feature"
                ),
            },
            "protocol_type_tcp": {
                "concept": "tcp_protocol_usage",
                "category": "protocol",
                "definition": (
                    "Indicates whether the analyzed connection "
                    "uses the TCP protocol."
                ),
                "soc_interpretation": (
                    "Provides protocol context about the presence "
                    "or absence of TCP traffic."
                ),
                "value_type": "binary",
                "source": (
                    "Derived from the one-hot encoding of the "
                    "NSL-KDD protocol_type feature"
                ),
            },
        }
    )

    data["version"] = "1.1"

    save_json(CONCEPTS_PATH, data)


def update_actions() -> None:
    data = load_json(ACTIONS_PATH)

    actions = data.setdefault("actions", {})

    actions.update(
        {
            "icmp_protocol_usage": {
                "conditions": [
                    {
                        "operator": "==",
                        "threshold": 1,
                        "state": "present",
                        "verb": "observe",
                        "object": "icmp_protocol_activity",
                        "weight": 1.8,
                        "tag": "protocol",
                    },
                    {
                        "operator": "==",
                        "threshold": 0,
                        "state": "absent",
                        "verb": "observe",
                        "object": "icmp_protocol_activity",
                        "weight": 1.0,
                        "tag": "protocol",
                    },
                ]
            },
            "icmp_echo_service_usage": {
                "conditions": [
                    {
                        "operator": "==",
                        "threshold": 1,
                        "state": "present",
                        "verb": "observe",
                        "object": "icmp_echo_activity",
                        "weight": 2.0,
                        "tag": "discovery",
                    },
                    {
                        "operator": "==",
                        "threshold": 0,
                        "state": "absent",
                        "verb": "observe",
                        "object": "icmp_echo_activity",
                        "weight": 1.0,
                        "tag": "discovery",
                    },
                ]
            },
            "same_service_multi_host_rate": {
                "conditions": [
                    {
                        "operator": ">=",
                        "threshold": 0.2,
                        "state": "elevated",
                        "verb": "increase",
                        "object": "multi_host_service_activity",
                        "weight": 1.6,
                        "tag": "discovery",
                    },
                    {
                        "operator": "<",
                        "threshold": 0.2,
                        "state": "low",
                        "verb": "observe",
                        "object": "multi_host_service_activity",
                        "weight": 1.0,
                        "tag": "discovery",
                    },
                ]
            },
            "destination_syn_error_rate": {
                "conditions": [
                    {
                        "operator": ">=",
                        "threshold": 0.5,
                        "state": "high",
                        "verb": "increase",
                        "object": "syn_error_activity",
                        "weight": 1.7,
                        "tag": "network",
                    },
                    {
                        "operator": "<",
                        "threshold": 0.5,
                        "state": "low",
                        "verb": "observe",
                        "object": "syn_error_activity",
                        "weight": 1.0,
                        "tag": "network",
                    },
                ]
            },
            "ftp_data_service_usage": {
                "conditions": [
                    {
                        "operator": "==",
                        "threshold": 1,
                        "state": "present",
                        "verb": "observe",
                        "object": "ftp_data_service",
                        "weight": 1.2,
                        "tag": "service",
                    },
                    {
                        "operator": "==",
                        "threshold": 0,
                        "state": "absent",
                        "verb": "observe",
                        "object": "ftp_data_service",
                        "weight": 1.0,
                        "tag": "service",
                    },
                ]
            },
            "tcp_protocol_usage": {
                "conditions": [
                    {
                        "operator": "==",
                        "threshold": 1,
                        "state": "present",
                        "verb": "observe",
                        "object": "tcp_protocol_activity",
                        "weight": 1.2,
                        "tag": "protocol",
                    },
                    {
                        "operator": "==",
                        "threshold": 0,
                        "state": "absent",
                        "verb": "observe",
                        "object": "tcp_protocol_activity",
                        "weight": 1.0,
                        "tag": "protocol",
                    },
                ]
            },
        }
    )

    data["version"] = "1.1"

    save_json(ACTIONS_PATH, data)


def update_probe_pattern() -> None:
    data = load_json(PATTERNS_PATH)

    patterns = data.get("patterns", [])

    for pattern in patterns:
        if pattern.get("pattern_id") != "BP-SCAN-001":
            continue

        pattern["name"] = "icmp_host_exploration_pattern"

        pattern["description"] = (
            "ICMP echo activity associated with contextual evidence "
            "of repeated or distributed host exploration."
        )

        pattern["required_conditions"] = [
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
        ]

        pattern["supporting_conditions"] = [
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
            {
                "verb": "increase",
                "object": "destination_service_diversity",
                "state": "high",
            },
            {
                "verb": "increase",
                "object": "service_variation",
                "state": "high",
            },
            {
                "verb": "increase",
                "object": "syn_error_activity",
                "state": "high",
            },
        ]

        pattern["minimum_supporting_conditions"] = 1
        pattern["associated_labels"] = ["Probe"]
        pattern["category"] = "network_discovery"
        pattern["severity"] = "medium"

        break
    else:
        raise ValueError(
            "Le motif BP-SCAN-001 est introuvable."
        )

    data["version"] = "1.1"

    save_json(PATTERNS_PATH, data)


def main() -> None:
    update_concepts()
    update_actions()
    update_probe_pattern()

    print("Base de connaissances Probe mise à jour.")
    print(f"- Concepts : {CONCEPTS_PATH}")
    print(f"- Actions  : {ACTIONS_PATH}")
    print(f"- Motifs   : {PATTERNS_PATH}")


if __name__ == "__main__":
    main()