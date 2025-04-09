import json
import os
import marmote.core as mc
import marmote.mdp as md


def build_expression(variables, values):
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


def create_jani_model(model, criterion, stateSpace, transitions, actionSpace=None, reward=None):
    num_states = stateSpace.Cardinal()
    if actionSpace is not None:
        num_actions = actionSpace.Cardinal()
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
            "initial-value": 0
        }
        variables.append(variable_dict)

    model = {
        "jani-version": 1,
        "name": "MDP Model",
        "type": model.className(),
        "criterion": criterion,
        "features": ["rewards"],
        "variables": variables,
        "actions": [
            {"name": f"action{i}"} for i in range(num_actions)
        ],
        "automata": [{
            "name": "MDPProcess",
            "locations": [{"name": "loc0"}],
            "initial-locations": ["loc0"],
            "edges": []
        }],
        "system": {
            "elements": [
                {
                    "automaton": "MDPProcess"
                }]
        }
    }

    automaton = model["automata"][0]

    for a in range(num_actions):
        for i in range(num_states):
            stateIn = stateSpace.DecodeState(i)
            destinations = []
            for j in range(num_states):
                stateOut = stateSpace.DecodeState(j)
                if isinstance(transitions, list):
                    prob = transitions[a].getEntry(i, j)
                else:
                    prob = transitions.getEntry(i, j)
                if prob > 0:
                    if reward is not None:
                        reward_value = reward.getEntry(i, a)
                    else:
                        reward_value = 0
                    destinations.append({
                        "location": "loc0",
                        "probability": {
                            "exp": prob
                        },
                        "assignments": [
                            {
                                "ref": variable_names[n],
                                "value": int(stateOut[n])
                            } for n in range(dims)
                        ],
                        "rewards": {
                            "exp": reward_value
                        }
                    })
            automaton["edges"].append({
                "location": "loc0",
                "action": f"act{a}",
                "guard":
                    build_expression(variable_names, stateIn)
                ,
                "destinations": destinations
            })

    return model


def save_jani_model_to_file(model, filename):
    filename = f"{filename}.janiR"
    counter = 1
    while os.path.exists(filename):
        filename = f"{filename.split('.')[0]}_{counter}.janiR"
        counter += 1

    jani_content = json.dumps(model, indent=2)

    with open(filename, 'w') as file:
        file.write(jani_content)
    print(f"Model saved as {filename}")


if __name__ == "__main__":
    # matrix for the a_0 action
    dim_SS = 4  # dimension of the state space
    dim_AS = 3  # dimension of the action space

    stateSpace = mc.MarmoteInterval(0, 3)
    actionSpace = mc.MarmoteInterval(0, 2)

    P0 = mc.SparseMatrix(dim_SS)

    P0.setEntry(0, 1, 0.875)
    P0.setEntry(0, 2, 0.0625)
    P0.setEntry(0, 3, 0.0625)
    P0.setEntry(1, 1, 0.75)
    P0.setEntry(1, 2, 0.125)
    P0.setEntry(1, 3, 0.125)
    P0.setEntry(2, 2, 0.5)
    P0.setEntry(2, 3, 0.5)
    P0.setEntry(3, 3, 1.0)

    P1 = mc.SparseMatrix(dim_SS)
    P1.setEntry(0, 1, 0.875)
    P1.setEntry(0, 2, 0.0625)
    P1.setEntry(0, 3, 0.0625)
    P1.setEntry(1, 1, 0.75)
    P1.setEntry(1, 2, 0.125)
    P1.setEntry(1, 3, 0.125)
    P1.setEntry(2, 1, 1.0)
    P1.setEntry(3, 3, 1.0)

    P2 = mc.SparseMatrix(dim_SS)
    P2.setEntry(0, 1, 0.875)
    P2.setEntry(0, 2, 0.0625)
    P2.setEntry(0, 3, 0.0625)
    P2.setEntry(1, 0, 1.0)
    P2.setEntry(2, 0, 1.0)
    P2.setEntry(3, 0, 1.0)

    trans = [P0, P1, P2]

    Reward = mc.FullMatrix(dim_SS, dim_AS)
    Reward.setEntry(0, 0, 0)
    Reward.setEntry(0, 1, 4000)
    Reward.setEntry(0, 2, 6000)
    Reward.setEntry(1, 0, 1000)
    Reward.setEntry(1, 1, 4000)
    Reward.setEntry(1, 2, 6000)
    Reward.setEntry(2, 0, 3000)
    Reward.setEntry(2, 1, 4000)
    Reward.setEntry(2, 2, 6000)
    Reward.setEntry(3, 0, 3000)
    Reward.setEntry(3, 1, 4000)
    Reward.setEntry(3, 2, 6000)

    criterion = "min"

    mdp1 = md.AverageMDP(criterion, stateSpace, actionSpace, trans, Reward)

    model = create_jani_model(mdp1, "minimize", stateSpace, trans, actionSpace, Reward)

    save_jani_model_to_file(model, 'simpleJaniFile')


