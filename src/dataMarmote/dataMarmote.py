import json
import os
from typing import Optional, Dict, List, Set, Union

from marmote.core import MarmoteBox, FullMatrix, SparseMatrix, MarmoteInterval, DISCRETE
from marmote.markovchain import MarkovChain
from marmote.mdp import AverageMDP, DiscountedMDP, FiniteHorizonMDP, TotalRewardMDP


class DataMarmote:
    def __init__(
        self,
        name: str,
        model_type: str,
        transitions: Union[Dict, List[SparseMatrix]],
        stateSpace: MarmoteBox,
        criterion: str = "min",
        actionSpace: Optional[Set[str]] = None,
        rewards: Optional[List, FullMatrix] = None,
        beta: Optional[float] = None,
        horizon: Optional[int] = None
    ):
        self.name: str = name
        self.type: str = model_type  # 'mc' or 'mdp'
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


    def create_marmote_object(self, model_type="AverageMDP"):
        stateSpace = self.stateSpace

        if self.type != "dtmc":
            nb_action = len(self.actionSpace)
            actionSpace = MarmoteInterval(0, nb_action-1)

            if isinstance(self.transitions, dict):
                transitions = self.transitions
                trans = [SparseMatrix(stateSpace) for _ in range(nb_action)]
                actions = list(self.actionSpace)
                actions.sort()
                for index, action in enumerate(actions):
                    transition = transitions[action]
                    for state_out, destinations in transition.items():
                        for destination, value in destinations:
                            i = stateSpace.Index(state_out)
                            j = stateSpace.Index(destination)
                            trans[index].setEntry(i,j,value[0])
            else:
                trans = self.transitions

            if self.type == "mdp":
                chosen_type = model_type
            else:
                chosen_type = self.type
            if chosen_type == "AverageMDP":
                return AverageMDP(self.criterion, stateSpace, actionSpace, trans, self.rewards)
            elif chosen_type == "DiscountedMDP":
                return DiscountedMDP(self.criterion, stateSpace, actionSpace, trans, self.rewards, self.beta)
            elif chosen_type == "FiniteHorizonMDP":
                return FiniteHorizonMDP(self.criterion, stateSpace, actionSpace, trans, self.rewards, self.horizon, self.beta)
            elif chosen_type == "TotalRewardMDP":
                return TotalRewardMDP(self.criterion, stateSpace, actionSpace, trans, self.rewards)
            else:
                raise ValueError(f"Invalid type: {chosen_type}.")
        else:
            P = SparseMatrix(stateSpace)
            # to do, not sure
            transitions = self.transitions
            for state_out, destinations in transitions.items():
                for destination, value in destinations:
                    i = stateSpace.Index(state_out)
                    j = stateSpace.Index(destination)
                    P.setEntry(i,j,value)
            P.set_type(DISCRETE)
            return MarkovChain(P)

    @staticmethod
    def __build_expression(variables, values):
        def recurse(items):
            if len(items) == 1:
                var, val = items[0]
                return {"op": "=", "left": var, "right": int(val)}
            else:
                mid = len(items) // 2
                left_expr = recurse(items[:mid])
                right_expr = recurse(items[mid:])
                return {
                    "op": "∧",
                    "left": left_expr,
                    "right": right_expr
                }

        variable_value_pairs = list(zip(variables, values))
        expression = {
            "exp": recurse(variable_value_pairs)
        }
        return expression

    def __create_jani_model(self):
        stateSpace = self.stateSpace
        actionSpace = self.actionSpace
        if actionSpace is not None:
            num_actions = len(actionSpace)
        else:
            num_actions = 1

        dims = stateSpace.tot_nb_dims()
        variable_names = [f"x{i + 1}" for i in range(dims)]
        variables = []

        for i, name in enumerate(variable_names):
            variable_dict = {
                "name": name,
                "type": {
                    "kind": "bounded",
                    "base": "int",
                    "lower-bound": 0,
                    "upper-bound": stateSpace.CardinalbyDim(i) - 1
                },
            }
            variables.append(variable_dict)

        model = {
            "jani-version": 1,
            "name": self.name,
            "type": self.type,
            "criterion": self.criterion,
            "features": ["rewards"],
            "variables": variables,
            "actions": [
                {"name": f"action{i}"} for i in range(num_actions)
            ],
            "automata": [{
                "name": "automata0",
                "locations": [{"name": "loc0"}],
                "initial-locations": ["loc0"],
                "edges": []
            }],
            "system": {
                "elements": [
                    {
                        "automaton": "automata0"
                    }]
            }
        }

        automaton = model["automata"][0]
        if isinstance(self.transitions, dict):
            transitions = self.transitions
            actions = list(self.actionSpace)
            actions.sort()
            for index, action in enumerate(actions):
                transition = transitions[action]
                for state_out, destinations in transition.items():
                    destinations = []
                    for destination, value in destinations:
                        prob, reward = value
                        destinations.append({
                            "location": "loc0",
                            "probability": {
                                "exp": prob
                            },
                            "assignments": [
                                {
                                    "ref": variable_names[n],
                                    "value": int(destination[n])
                                } for n in range(dims)
                            ],
                            "rewards": {
                                "exp": reward
                            }
                        })
                    automaton["edges"].append({
                        "location": "loc0",
                        "action": action,
                        "guard":
                            self.__build_expression(variable_names, state_out)
                        ,
                        "destinations": destinations
                    })
        return model


    def save(self, filename: str):
        filename = f"{filename}.janiR"
        counter = 1
        while os.path.exists(filename):
            filename = f"{filename.split('.')[0]}_{counter}.janiR"
            counter += 1

        model = self.__create_jani_model()
        jani_content = json.dumps(model, indent=2)

        with open(filename, 'w') as file:
            file.write(jani_content)
        print(f"Model saved as {filename}")

        #  action        i       dest    prob  reward
        # {"action1" : {(0,0): {(0,1) : (0.5, 100)}}}