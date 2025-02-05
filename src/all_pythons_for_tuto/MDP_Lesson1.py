#!/usr/bin/env python
# coding: utf-8

# # MDP Lesson 1: discounted MDP  

# ## Using the library

# **Import the modules**

# The following commands allows to import the modules

# In[ ]:


import marmote.core as mc 
import marmote.mdp as mmdp  


# It is necessary to import the modules *marmote.core* and *marmote.mdp* from *pyMarmote* library.  
# Hence *marmote.core* handles the basic objects (*Sets* (very similar to tensors), *distributions*, *matrices*,...)  
# Hence *marmote.mdp* handles the objects for Markov Decision Processes.

# In this first lesson, we show how to make and how to solve a simple infinite-horizon discounted criteria MDP.

# ## Build a simple MDP

# ### Reminders about  MDP

# Formally a MDP is a tuple *(S,A,P_a,R)* where  
# + S is the state space  
# + A is the action space  
# + P_a is a collection of transition matrices. A matrix for each action  
# + R is a reward (or cost) matrix  

# #### Description of the exemple implemented here

# We assume a simple model with two states x1=0 and x2=1 and in each state: two actions a1=0 and a2=1.
# 
# The reward matrix is:  
# 
# |      |       |
# |:----:| :----:|
# | 4.5  |  2    |
# | -1.5 | 3.0   |
# 
# where *r(x,a)* is the entry with row coordinate $x$ and column coordinate $a$.   
# The entry *r(x,a)* represents the reward when in state *x* action *a* is performed.
# 
# 
# The transition matrices are :
# 
# Transition matrix of the action 0:    
# 
# |      |       |
# |:----:| :----:|
# | 0.6  |  0.4  |
# | 0.5  |  0.5  |
# 
#   
# Transition matrix of the action 1:  
# 
# |      |       |
# |:----:| :----:|
# | 0.2  |  0.8  |
# | 0.7  |  0.3  |

# ### Elements of a MDP object

# #### Attributes of the object

# A MDP in marmote is an object that receives (at least) four important attributes
#  
# 1. The *state space* that is a *MarmoteSet* object (an object to deal with Sets).   
# The simplest *MarmoteSet* object is the object *MarmoteInterval*. Here we use a *MarmoteInterval* for the state space.
# 2. The action space is also a *MarmoteSet* object. 
# In our case also a *MarmoteInterval* object.
# 3. A *list* of transition structures.  
# Each entry in the list corresponds with a *TransitionStructure* associated with a given action.  
#   - The *TransitionStructure* at the *a*-th entry of the list is the *TransitionStructure* associated 
#  with the action whose index is *a*.
#   - A *TransitionStructure* describes the probability transition to move from a state *i* into state *j*.
#   - The *TransitionStructure* can be a *FullMatrix* or a *SparseMatrix*. Here the *(i,j)* entry of a matrix gives the probability to move from state with index *i* to state with index *j*.
# 4. a reward (or cost).  
# This is a TransitionStructure (preferably a FullMatrix) in which the cost for an action *a* in a state *x* is defined.
# Rows are used for indexes of states and columns are used for indexes of action. Hence the entry with indexes *(x,a)* represents the  cost of action of index *a* in state of index *x*;
# 

# #### How to build the DiscountedMDP object

# **Create state space and action space**

# Here we define two objects `MarmoteInterval` for state and action spaces. The two following lines define two intervals going from 0 to 1.

# In[ ]:


actionSpace = mc.MarmoteInterval(0,1) 
stateSpace = mc.MarmoteInterval(0,1)


# **Storing matrices**

# A `list` is used to store all transition matrices.  
# The number of matrices should correspond with the size of the action space.

# In[ ]:


trans=list()


# **Build transition matrices**

# Now, we create P0 an object `SparseMatrix` with size 2x2. The P0 matrix is a transition matrix.

# In[ ]:


P0 = mc.SparseMatrix(2)


# The command to initialize an entry is `setEntry` with syntax `setEntry(row,column,value)`   
# Hence: the command `P0.setEntry(0,0,0.6)` assigns the value 0.6 to the entry of index (0, 0) of P0.
# 
# Now, we define 4 no null entries to P0 matrix.

# In[ ]:


P0.setEntry(0,0,0.6)
P0.setEntry(0,1,0.4)
P0.setEntry(1,0,0.5)
P0.setEntry(1,1,0.5)


# Then one prints the matrix

# In[ ]:


print("Matrix",P0)


# The next instruction assigns the matrix P0 to the index 0 coordinate of the `trans` list, which means that the transition matrix for the action a0 is now stored at index 0 of the `trans` list.

# In[ ]:


trans.append(P0) 


# We now add a new `SparseMatrix`. Note that a `SparseMatrix` is an object of *marmote.core*.

# In[ ]:


P1 = mc.SparseMatrix(2)
P1.setEntry(0,0,0.2)
P1.setEntry(0,1,0.8)
P1.setEntry(1,0,0.7)
P1.setEntry(1,1,0.3)
trans.append(P1)


# **Build reward matrix**

# We create a `FullMatrix`object with size 2x2. This matrix is used for storing the rewards associated with each couple (state,action).

# In[ ]:


Reward  = mc.FullMatrix(2,2)


# The following lines add non-zero values to the matrix *Reward*. More precisely: 
# 
# - The first line adds the value 4.5 to position (0, 0) in the matrix.
# - The second line adds the value 2 to position (1, 1) in the matrix.
# - The third line adds the value -1.5 to position (1, 0) in the matrix.
# - The fourth line adds the value  3 to position (1, 1) in the matrix.
# 

# In[ ]:


Reward.setEntry(0,0,4.5) 
Reward.setEntry(0,1,2)
Reward.setEntry(1,0,-1.5) 
Reward.setEntry(1,1,3)


# **Additional parameters of the discounted MDP**

# Two additional parameters should be entered : *beta* and *critere*     
# - *beta* is the discount factor for incorporating future values.  
# - *critère* indicates the optimisation criterion, which is either maximisation (*"max"*) or minimisation (*"min"*).

# In[ ]:


beta = 0.95
criterion="max"


# **Build a discounted MDP**

# We now construct a `DiscountedMDP` object using the constructor, to which we pass the parameters defined in this page. Parameters of a `DiscountedMDP` are:     
# - *criterion* either *min* or *max*  
# - an object that encodes the state space  
# - an object that encodes the action space  
# - a list of TransitionStructure  
# - a reward  
# - the discount factor  

# In[ ]:


mdp = mmdp.DiscountedMDP(criterion, stateSpace, actionSpace, trans, Reward, beta) 


# Now one prints the MDP. This is used to display the various components of the MDP, such as state and action spaces, transition matrices and reward.

# In[ ]:


print(mdp)


# ## Solving the MDP

# ### List of solution methods

# We list here the different methods implemented to solve discounted Markov Decision processes. Detail of the methods can be found in the literature.  All these methods return a `FeedbackSolutionMDP` object.
# 1. method *Value Iteration* method name `ValueIteration`  
# 2. method *Value Iteration using Gauss Seidel* method name `ValueIterationGS`   
# 3. method Value Iteration with a given value function for initiate the process method name `ValueIterationInit`   
# 4. method Policy Iteration Modified method name `PolicyIterationModified`  
# 5. method Policy Iteration Modified with Gauss Seidel method name `PolicyIterationModifiedGS`  

# ### Running a solution method

# **Value-iteration**

# Parameters:  
# 1. *epsilon* a precision threshold used to determine the convergence of the  algorithm. The algorithm continues to iterate as long as the maximal difference between the new  values and the old values in a state is greater than epsilon.   
# 2. *maxIter* gives the maximal number of authorized iterations. 

# In[ ]:


epsilon = 0.00001
maxIter = 700


# To run a resolution by iterating the value. In order to find both the optimal policy and the optimal value in each of the states. The method returns a stationary solution.  
# The function returns a `FeedbackSolutionMDP` object.

# In[ ]:


optimum = mdp.ValueIteration(epsilon, maxIter) 


# Let us display the optimal solution of the Markov Decision Process.

# In[ ]:


print(optimum)


# **Gauss Seidel Value Iteration**

# The next line performs ten iterations of the value on the MDP to find the optimal policy and the optimal value of the states, but now using the Gauss-Seidel improvement for evaluating the value in a state.
# 

# In[ ]:


optimum2 = mdp.ValueIterationGS(epsilon,10)
print(optimum2)


# **Value Iteration Init**

# It is also possible to choose which value function will be used to start the value iteration process. To do this, one should enter a third parameter, which is an `FeedbackSolution` object whose `value` attribute will be used (see later for details about `FeedbackSolution`) to initiate the process.

# In[ ]:


optimum3 = mdp.ValueIterationInit(epsilon,200,optimum2)
print("Optimum 3",optimum3)


# **Policy Iteration Modified**

# In a policy-based approach, it is possible to evaluate a policy with a given precision that is not the same as the precision used to stop the process.  
# Please note that there is no implementation of the *Policy Iteration* algorithm. It has been chosen instead to implement the variant called *Policy Iteration Modified* in the book of Puterman.   
# Thus, *Policy Iteration Modified*  the third and fourth parameters will be the precision with which a policy will be evaluated and the maximum number of iterations allowed to approach the value.

# In[ ]:


optimum4 = mdp.PolicyIterationModified(epsilon, maxIter, 0.001, 100)
print("optimum4",optimum4)


# **Printing Information during the solving process**

# The following instruction modifies the printing of internal information during the solving methods such as the number of iterations performed and the precision reached. When using notebook the print depends on the OS

# In[ ]:


mdp.changeVerbosity(True)


# **Policy Iteration Modified with Gauss Seidel**

# The policy Iteration method can use (as described in Puterman's book) a Gauss Seidel evaluation. Please note that Policy Iteration Modified With Gauss Seidel Evaluation is not proven for all criteria (but for the disounted criteria it is).

# In[ ]:


optimum5 = mdp.PolicyIterationModifiedGS(epsilon, maxIter, 0.001, 100)
print("last test",optimum5)


# ## About the SolutionMDP object

# The solution is stored with a `FeedbackSolutionMDP` object. This object has attributes that store a value and an action for each state.  
# The printing of a `FeedbackSolutionMDP` gives first information about the policy. The information for all the states in the state space is then displayed. All the information for a state is shown on one line, starting with the state index, the state value and the action associated with the value.

# In[ ]:




