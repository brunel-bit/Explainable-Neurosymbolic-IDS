from __future__ import annotations


class AttackChainBuilder:
    """
    Construit une chaîne d'attaque logique à partir
    des patterns comportementaux et de leur mapping MITRE.
    """

    def __init__(self):

        self.tactic_order = [
            "Reconnaissance",
            "Resource Development",
            "Initial Access",
            "Execution",
            "Persistence",
            "Privilege Escalation",
            "Defense Evasion",
            "Credential Access",
            "Discovery",
            "Lateral Movement",
            "Collection",
            "Command and Control",
            "Exfiltration",
            "Impact",
        ]

    def build(self, mitre_results):

        if not mitre_results:
            return []

        chain = sorted(
            mitre_results,
            key=lambda x: self._rank(
                x["tactic_name"]
            ),
        )

        attack_chain = []

        previous = None

        for item in chain:

            node = {
                "pattern_id": item["pattern_id"],
                "technique_id": item["technique_id"],
                "technique_name": item["technique_name"],
                "tactic": item["tactic_name"],
                "confidence": item["confidence"],
            }

            if previous is not None:

                node["previous_pattern"] = previous[
                    "pattern_id"
                ]

            attack_chain.append(node)

            previous = node

        return attack_chain

    def _rank(self, tactic):

        try:
            return self.tactic_order.index(tactic)
        except ValueError:
            return 999