# Documentation

## `janiParser.dataMarmote.dataMarmote.DataMarmote`
### Constructor
**`DataMarmote(data: dict)`**

**Parameters**:

* **data**: A dictionary that contains at least the following information:

    * **name**: The name of the model.

    * **type**: The model type. Supported types include: `DiscountedMDP`, `AverageMDP`, `TotalRewardMDP`, `FiniteHorizonMDP` and `MarkovChain`.

    * **states**: A set of states, where each state is represented by a tuple to reduce memory usage.

    * **actions**: A set of actions available.

    * **transition-dict**: A dictionary that contains transition probabilities and rewards. Its structure depends on the model types:

        * For MDPs: `[action][state][next state]` -> `[probability, reward]`
        
        * For MCs: `[state][next state]` -> `probability`
    
    * **absorbing-states**: A set of absorbing states (i.e., states with no outgoing transitions).

    * **state-template**: A template that defines the encoding of states. It is a dictionary that maps each state variable to a unique index, indicating its position in the tuple representation. This is typically used to interpret tuple-encoded states and to ensure a stable state variable order.

    * **state-variable-types**: A dictionary that maps each state variable to its type definition. This is typically used when converting to a `.janir` file.

    * **initial-states**: A list of initial states, where each state is represented by a tuple.

    * **state-variable-initial-values**: A dictionary that maps state variables to their initial values.

    * **criterion**: The optimization criterion used for the model. **If the model type is `MarkovChain`, then this field is not required.**

### Methods
* **`createMarmoteObject()`**

    **Construct and return a Marmote object using the data given during class construction.**

* **`buildTransitionRewardForMDPToolbox()`**

    **Build the transition matrices as a list of SciPy sparse matrices and the reward matrix as a full matrix (NumPy 2D array). These parameters are then passed to `MDPToolbox` to solve the `MDP` problem.**

* **`saveAsJaniRFile(path: str)`**

    **Convert the current model to a `.janir` file and save it to the specified path.**

    **Parameters**:

    * **path**: File path where the generated `.janir` file will be saved.

* **`getInitStatesIdx()`**

    **Return the list of initial states.**

* **`stateToIdx(state: tuple)`**

    **Return the unique index corresponding to the specified state.**

    **Parameters**:

    * **state**: A tuple representation of a specific state.

* **`ìdxToState(idx: int)`**

    **Return the state corresponding to the index.**

    **Parameters**:

    * **idx**: An integer that represents the index of the constructed matrices.

## `janiParser.reader.reader.JaniReader`
### Constructor
`JaniReader(path: str, modelParams: dict, isLocalPath: bool)`

**Parameters**:

* **path**: A path to the `.jani` file, which can be a local path or a URL path.

* **modelParams**: Optional parameter used to configure a scalable model. It defaults to an empty dictionary.

* **isLocalPath**: A boolean that indicates whether the given path is a local file path or not. It defaults to True.

**Examples**:
```python
from janiParser.reader.reader import JaniReader

# Example of using a local file path.
reader = JaniReader("./benchmarks/prism2jani/consensus.2.v1.jani", modelParams={ "K": 10 })
```

```python
# Example of using a URL path.
reader = JaniReader("https://qcomp.org/benchmarks/mdp/consensus/consensus.2.jani", modelParams={ "K": 10 }, isLocalPath=False)
```

### Methods
* **`build()`**

    **Load and parse the target file specified during class construction to build a JANI model representation. It returns an instance of class `JaniModel`.**

## `janiParser.reader.reader.JaniRReader`
### Constructor
`Bases: janiParser.reader.reader.JaniReader`

`JaniRReader(path: str, modelParams: dict)`

**Parameters**:

* **path**: A path to the `.janir` file. It accepts only the local file path.

* **modelParams**: Optional parameter used to configure a scalable model. It defaults to an empty dictionary.

### Methods
* **`build()`**

    **Load and parse the target file specified during class construction to build a JANIR model representation. It returns an instance of class `JaniRModel`.**

## `janiParser.reader.model.JaniModel`
### Constructor
`JaniModel(name: str, type: str)`

**Parameters**:

* **name**: The name of the model.

* **type**: The type of the model specified according to the JANI standard. **In this project, only `dtmc` and `mdp` types are supported.**

### Methods
* **`getMDPData(name: str)`**

    **Starting from the initial states, it explores all reachale states under the specified property and returns a dictionary with all necessary data to build a corresponding Marmote instance (`MDPs`). See the `DataMarmote` class documentation above for more details on the returned data.**

    **Parameters**:

    * **name**: The name of the property to be modeled.

* **`getMCData()`**

    **This method is similar to `getMDPData`. It explores all reachable states and returns a dictionary with all necessary data to build a corresponding Marmote instance (`MCs`). See the `DataMarmote` class documentation above for more details on the returned data.**

## `janiParser.reader.model.JaniModel`
### Constructor
`Bases: janiParser.reader.model.JaniModel`

`JaniRModel(name: str, type: str, criterion: str, gamma: float, horizon: int)`

**Parameters**:

* **name**: The name of the model.

* **type**: The type of the model. Supported types include: `DiscountedMDP`, `AverageMDP`, `TotalRewardMDP`, `FiniteHorizonMDP` and `MarkovChain`.

* **criterion**: The optimization criterion used for the model (`"max"` or `"min"`).

* **gamma**: The discount factor used in discounted and finite horizon models.

* **horizon**: The finite horizon length used in finite horizon models.

### Methods
* **`getMDPData()`**

    **This method takes no arguments. Starting from the initial states, it explores all reachale states under the specified property and returns a dictionary with all necessary data to build a corresponding Marmote instance (`MDPs`). See the `DataMarmote` class documentation above for more details on the returned data.**

* **`getMCData()`**

    **This method is similar to `getMDPData`. It explores all reachable states and returns a dictionary with all necessary data to build a corresponding Marmote instance (`MCs`). See the `DataMarmote` class documentation above for more details on the returned data.**

## `janiParser.utils`
### Functions
* **`buildAndSaveMDPModelFromQCompBenchmark(fullModelName: str, modelParams: dict, replace: bool)`**
* **`loadDataMarmoteFromPickleFile(path: str)`**