<!-- Projet réalisé par Sorbonne Université  
Encadré par monsieur Emmanuel Hyon et monsieur Pierre-Henri Wuillemin  
Auteur : Zeyu TAO et Jiahua Li -->

## Description
We mainly implemented 2 parsers:

* **JaniReader** which parses a sub-language of **JANI** (only for dtmc (Discrete-Time Markov Chain) or mdp (Markov Secision Process) models)

* **JaniRReader** which parses a .janir file, this is a **JANI** language extension and only for dtmc and mdp models.

This projet is carried out by Zeyu TAO et Jiahua LI, and and supervised by Mr Emmanuel HYON and Mr Pierre-Henri WUILLEMIN.

## Architecture
![](architecture.png)

# Usage
```python
from src.reader.reader import *
from src.dataMarmote.dataMarmote import *

path = "./benchmarks/prism2jani/csma.2-2.v1.jani"
# Parse .jani file.
model = JaniReader(path).build()
# Choose a property to convert and build the corresponding state space.
MDPData = model.getMDPData("all_before_max")
# [Out]
# 1 initial states.
# 1037 states, 10 actions, 1280 transitions

dataMarmote = DataMarmote(MDPData)
# Build a Marmote instance.
mdp = dataMarmote.createMarmoteObject()

# Resolution.
opt = mdp.ValueIteration(1e-10, 2000)
for idx in dataMarmote.getInitStatesIdx():
    print(opt.getValueIndex(idx))
# [Out]
# 0.875


# Save as a .janir file.
dataMarmote.saveAsJaniRFile("csma.janir")
# Parse .janir file.
model = JaniRReader("csma.janir").build()

dataMarmote = DataMarmote(model.getMDPData())
mdp = dataMarmote.createMarmoteObject()
# [Out]
# 1 initial states.
# 1037 states, 10 actions, 1280 transitions

opt = mdp.ValueIteration(1e-10, 2000)
for idx in dataMarmote.getInitStatesIdx():
    print(opt.getValueIndex(idx))
# [Out]
# 0.875
```