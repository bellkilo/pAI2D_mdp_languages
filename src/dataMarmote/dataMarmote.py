from typing import Optional, Dict, List, Set, Union

from marmote.core import MarmoteBox, FullMatrix, SparseMatrix


class DataMarmote:
    def __init__(
        self,
        name: str,
        type: str,
        criterion: str,
        transitions: Union[Dict, List[SparseMatrix]],
        stateSpace: MarmoteBox,
        actionSpace: Optional[Set[str]] = None,
        rewards: Optional[List, FullMatrix] = None,
        beta: Optional[float] = None,
        horizon: Optional[int] = None
    ):
        self.name: str = name
        self.type: str = type  # 'mc' or 'mdp'
        self.criterion: str = criterion  # 'min' or 'max'
        self.transitions: Union[Dict, List[SparseMatrix]] = transitions
        self.stateSpace: MarmoteBox = stateSpace
        self.actionSpace: Optional[Set[str]] = actionSpace
        self.rewards: Optional[List, FullMatrix] = rewards
        self.beta: Optional[float] = beta
        self.horizon: Optional[int] = horizon

        self._validate()

    def _validate(self) -> None:
        if self.type not in {"dtmc", "mdp", "DiscountedMDP", "AverageMDP", "TotalRewardMDP", "FiniteHorizonMDP"}:
            raise ValueError("Invalid type. Must be 'dtmc' or 'mdp'.")

        if not isinstance(self.stateSpace, MarmoteBox):
            raise TypeError("stateSpace must be a MarmoteBox.")

        if self.type != "dtmc":
            if self.actionSpace is None:
                raise ValueError("MDP must define an actionSpace.")
            if self.rewards is None:
                raise ValueError("MDP must define rewards.")

        if self.type == "DiscountedMDP":
            if self.beta is None:
                raise ValueError("Discounted MDP must define a discounted factor.")
            if self.beta < 0 or self.beta >= 1:
                raise ValueError("Discounted factor must have a value between 0 and 1.")

        if self.type == "FiniteHorizonMDP":
            if self.beta is None:
                raise ValueError("FiniteHorizon MDP must define a discounted factor.")
            if self.beta < 0 or self.beta >= 1:
                raise ValueError("Discounted factor must have a value between 0 and 1.")
            if self.horizon is None:
                raise ValueError("FiniteHorizon MDP must define an horizon.")


    def create_marmote_object(self, type="AverageMDP"):
        pass