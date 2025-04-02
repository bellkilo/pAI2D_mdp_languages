import json
import os
import marmote.core as mc
import marmote.mdp as md

def create_jani_model(criterion, trans, reward=None):
    if isinstance(trans, list):
        num_states = trans[0].size()
        num_actions = len(trans)
    else:
        num_states = trans.size()
        num_actions = 1

    model = {
        "jani-version": 1,
        "name": "MDP Model",
        "type": "mdp",
        "features": ["rewards"],
        "variables": [
            {
                "name": "x",
                "type": {
                    "kind": "bounded",
                    "base": "int",
                    "lower-bound": 0,
                    "upper-bound": num_states - 1
                },
                "initial-value": 0
            }
        ],
        "actions": [
            {"name": f"action{i}"} for i in range(num_actions)
        ],
        "automata": [{
            "name": "MDPProcess",
            "locations": [{"name": "loc0"}],
            "initial-locations": ["loc0"],
            "edges": []
        }],
        "properties": [
            {"name": "expected_reward", "expression": "R{\"time\"}min=? [F \"final\"]"}
        ],
        "system": {
            "elements": [
                {
                    "automaton": "MDPProcess"
                }]
        },
        "criterion": criterion
    }

    automaton = model["automata"][0]

    for a in range(num_actions):
        for i in range(num_states):

            destinations = []
            for j in range(num_states):
                if isinstance(trans, list):
                    prob = trans[a].getEntry(i, j)
                else:
                    prob = trans.getEntry(i, j)
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
                                "ref": "x",
                                "value": j
                            }
                        ],
                        "rewards": {
                            "exp": reward_value
                        }
                    })
            automaton["edges"].append({
                "location": "loc0",
                "action": f"act{a}",
                "guard": {
                    "exp": {
                        "op": "=",
                        "left": "x",
                        "right": i
                    }
                },
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

    model = create_jani_model("minimize", trans, Reward)

    save_jani_model_to_file(model, 'simpleJaniFile')


