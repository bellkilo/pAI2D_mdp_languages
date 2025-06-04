# AI2D project 2024-2025: Comparison of methods for solving MDPs on modeling language instances

**For this project, we primarily implemented a parser capable of translating a specific subset of JANI files into corresponding Marmote instances**. This parser does not support all model types defined in the JANI specification; instead, it is limited to discrete-time Markov models, specifically **Discrete-Time Markov Chains (dtmc)** and **Discrete-Time Markov Decision Process (mdp)**.

We have also introduced **an extension to JANI**, named **JANIR**, since we noticed that standard JANI files cannot represent all types of MDP models. In particular, they are limited to modeling only **shortest-path or longest-path MDPs**.

This projet is carried out by **Zeyu TAO** et **Jiahua LI**, and and supervised by **Mr Emmanuel HYON** and **Mr Pierre-Henri WUILLEMIN**.

## Installation
To install via git:
```bash
git clone git@github.com:bellkilo/pAI2D_mdp_languages.git
```

<!-- To install via pip (TODO no implemented yet):
```bash
pip install git+https://github.com/bellkilo/pAI2D_mdp_languages
``` -->

## Usage
### Baseline architecture
<img src="architecture.png" width="75%">

### JANI to Marmote instance
```python
from janiParser.reader.reader import JaniReader
from janiParser.dataMarmote.dataMarmote import DataMarmote


# Create an instance of JaniReader. This class requires at least the local path or URL path of the file. (Details about the JaniReader class are provided in another ./janiParser/README.md file.)
# Example of using the URL path.
# reader = JaniReader("https://qcomp.org/benchmarks/mdp/philosophers-mdp/philosophers-mdp.3.jani", isLocalPath=False)
# Example of using the local path.
reader = JaniReader("./benchmarks/prism2jani/philosophers-mdp.3.v1.jani")

# Parser and create an instance of JaniModel.
model = reader.build()

# Get all the data needed to build the Marmote instance.
# The getMDPData function takes as argument the name of the property in which we want to check (JANI limitation).
MDPData = model.getMDPData("eat")

# Create an instance of DataMarmote using the data returned by getMDPData or getMCData.
dataMarmote = DataMarmote(MDPData)

# Create the corresponding Marmote instance.
mdp = dataMarmote.createMarmoteObject()
```

### JANI to JANIR
```python
from janiParser.reader.reader import JaniReader
from janiParser.dataMarmote.dataMarmote import DataMarmote

# Create an instance of JaniReader.
reader = JaniReader("./benchmarks/prism2jani/philosophers-mdp.3.v1.jani")

# Create an instance of JaniModel.
model = reader.build()

# Get all the data needed to build the Marmote instance.
MDPData = model.getMDPData("eat")

# Create an instance of DataMarmote.
dataMarmote = DataMarmote(MDPData)

# Save as a .janir file.
dataMarmote.saveAsJaniRFile("./philosophers-mdp.3.v1.janir")
```

### JANIR to Marmote instance
```python
from janiParser.reader.reader import JaniRReader
from janiParser.dataMarmote.dataMarmote import DataMarmote


# Create an instance of JaniRReader. This class works in the same way as JaniReader, but it does not support the URL path, which only takes as argument the local path of the file.
reader = JaniRReader("./philosophers-mdp.3.v1.janir")

# Parser and create an instance of JaniRModel.
model = reader.build()

# Get all the data needed to build the Marmote instance.
# The getMDPData function written in the JaniRModel class takes no arguments, since it defines an MDP in a general way, so there's no property to specify.
MDPData = model.getMDPData()

# Create an instance of DataMarmote.
dataMarmote = DataMarmote(MDPData)

# Create the corresponding Marmote instance.
mdp = dataMarmote.createMarmoteObject()
```

## Description of main source files
We provide brief descriptions of each main source file used in the project, along with the classes implemented in them (**Detailed documentation for each class is available in the ./janiParser/README.md file.**)
* **janiParser/reader/reader.py:**
    **This file contains the main implementation for parsing JANI or JANIR files.** It includes the ```JaniReader``` and ```JaniRReader``` classes, which are responsible for **reading** and **interpreting** both JANI and extended JANIR models from a local file path or from a URL path (specifically if the JANI file is hosted on https://qcomp.org/benchmarks/).

* **janiParser/reader/model.py:**
    **This file defines two main classes, ```JaniModel``` and ```JaniRModel```**, which encapsulate the **internal structure** of JANI and JANIR files. These classes provide two main methods, ```getMDPData```and ```getMCData```, which extract information from the JANI structure and return relevant data - such as the state space, action space, and state transitions - in the form of a Python dictionary.

* **janiParser/dataMarmote/dataMarmote.py:**
    This file contains a single class, ```DataMarmote```, which serves as an intermediary between JANI or JANIR files and Marmote instances. **This class primarily handles the translation of data extracted from JANI or JANIR models into a format compatible with Marmote, and also supports conversion from standard JANI to its extended JANIR form**.