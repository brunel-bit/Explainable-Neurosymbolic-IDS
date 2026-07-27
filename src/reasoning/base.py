from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ReasoningBackend(ABC):
    """
    Interface commune pour les moteurs de raisonnement.

    Les backends Python et Scallop doivent retourner un résultat
    ayant une structure compatible afin que le reste du pipeline
    reste indépendant du moteur choisi.
    """

    name: str = "abstract"

    @abstractmethod
    def reason(
        self,
        *,
        case_id: str,
        actions: list[dict[str, Any]],
        patterns: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Exécute le raisonnement symbolique pour un cas.

        Returns
        -------
        dict
            Résultat contenant au minimum :
            - backend
            - activated_patterns
            - derived_relations
            - attack_chains
            - trace
        """
        raise NotImplementedError
