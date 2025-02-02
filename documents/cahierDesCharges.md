# Cahier des charges

## 1 - Introduction
Ce projet fait partie de l’UE S2 du Master d’informatique de Sorbonne Université, encadrés par **Monsieur Emmanuel HYON** et **Monsieur Pierre-Henri WUILLEMIN**, et réalisés par les étudiants **Zeyu TAO** et **Jiahua LI**.

## 2 - Description du projet
Le projet peut être décomposé en 3 parties principales :

* **Les recherches sur les différents langages de modélisation :**  
Dans un premier temps, nous allons d'abord trouver les différents langages de modélisation pouvant être transformés en un modèle **MDP**. Ensuite, nous allons explorer en détails: leur sémantique, leur domaine d'utilisation, ainsi que leurs avantages et inconvénients.

* **Les implémentations de parser (y compris de lexer) :**  
Pour résoudre les instances issues d'un langage de modélisation à l'aide un solver de **MDP** (dans le cadre de ce projet, nous allons utiliser MarmoteMDP), nous allons implémenter un parser qui transforme une instance d'un langage spécifique en un modèle **MDP**.  
En général, un modèle **MDP** est représenté par un 4-uplet: $(S, A, P, R)$, où :
    * $S$ est un ensemble d'états possibles.
    * $A$ est un ensemble d'actions possibles.
    * $P$ est une matrice qui représente les probabilités de transition : $Pr(s_{t+1} | s_t, a)$.
    * $R$ une matrice qui représente une fonction de récompense : $R(s, a)$.

* **Les comparaisons de méthodes de résolutions de MDP :**  
Dans cette dernière partie du projet, nous allons comparer les différentes méthodes de résolution de **MDP** sur les différentes instances de langages de modélisation et analyser les résultats obtenus.
