#!/usr/bin/env python
# coding: utf-8

# # MDP Lesson 3:  average MDP and policy handling 

# **Import the modules**

# In[ ]:


import marmote.core as mc 
import marmote.mdp as md


# ## Build an average MDP

# ### A  machine repair model

# Assume a machine with 4 states: 
# - 0 = new
# - 1 = usable with minor deterioration
# - 2 = usable with major deterioration
# - 3 = unusable  
# 
# and has 3 actions
# - 1 = Do nothing: remain as is
# - 2 = Tune-up: return to status 1
# - 3 = Total repair: return to state 0  
# 
# Costs depend on the state and the chosen action.
# 
# - In state 0 *"Do nothing"* costs 0,    *"Tune-up"* costs 4000 and *"Total Repair"* costs 6000
# - In state 1 *"Do nothing"* costs 1000, *"Tune-up"* costs 4000 and *"Total Repair"* costs 6000
# - In state 2 *"Do nothing"* costs 3000, *"Tune-up"* costs 4000 and *"Total Repair"* costs 6000
# - In state 3 *"Do nothing"* costs 3000, *"Tune-up"* costs 4000 and *"Total Repair"* costs 6000

# ### Create all objects

# Spaces

# In[ ]:


dim_SS = 4 # dimension of the state space
dim_AS = 3 # dimension of the action space

stateSpace =  mc.MarmoteInterval(0,3)
actionSpace =  mc.MarmoteInterval(0,2)


# Create transition matrices

# In[ ]:


# matrix for the a_0 action
P0 = mc.SparseMatrix(dim_SS)

P0.setEntry(0,1,0.875)
P0.setEntry(0,2,0.0625)
P0.setEntry(0,3,0.0625)
P0.setEntry(1,1,0.75)
P0.setEntry(1,2,0.125)
P0.setEntry(1,3,0.125)
P0.setEntry(2,2,0.5)
P0.setEntry(2,3,0.5)
P0.setEntry(3,3,1.0)

P1 =  mc.SparseMatrix(dim_SS)
P1.setEntry(0,1,0.875)
P1.setEntry(0,2,0.0625)
P1.setEntry(0,3,0.0625)
P1.setEntry(1,1,0.75)
P1.setEntry(1,2,0.125)
P1.setEntry(1,3,0.125)
P1.setEntry(2,1,1.0)
P1.setEntry(3,3,1.0)

P2 =  mc.SparseMatrix(dim_SS)
P2.setEntry(0,1,0.875)
P2.setEntry(0,2,0.0625)
P2.setEntry(0,3,0.0625)
P2.setEntry(1,0,1.0)
P2.setEntry(2,0,1.0)
P2.setEntry(3,0,1.0)

trans = [P0, P1, P2]


# Create Cost Matrix

# In[ ]:


Reward  =  mc.FullMatrix(dim_SS, dim_AS)
Reward.setEntry(0,0,0)
Reward.setEntry(0,1,4000)
Reward.setEntry(0,2,6000)
Reward.setEntry(1,0,1000)
Reward.setEntry(1,1,4000)
Reward.setEntry(1,2,6000)
Reward.setEntry(2,0,3000)
Reward.setEntry(2,1,4000)
Reward.setEntry(2,2,6000)
Reward.setEntry(3,0,3000)
Reward.setEntry(3,1,4000)
Reward.setEntry(3,2,6000)


# Create AverageMDP object. Please note, that AverageMDP is an object with specific implemented algorithms.  

# In[ ]:


criterion="min"

mdp1 =  md.AverageMDP(criterion, stateSpace, actionSpace, trans,Reward)
print(mdp1)


# ### Solve the MDP

# **Solving with Value Iteration (VI) and Policy Iteration Modified (PIM)**

# The VI and PIM methods are available in an `AverageMPD` object used for average criteria MDP (the code is specific for the `AverageMDP` object since algorithms differ from the discounted case and total case). 

# In[ ]:


#create and initialize epsilon.
epsilon = 0.00001
#create and initialize the maximum number of iterations allowed.
maxIter = 500

print("Compute with value Iteration\n")
optimum = mdp1.ValueIteration(epsilon, maxIter)
print(str(optimum))

print("Computation with Policy Iteration modified")
optimum2 = mdp1.PolicyIterationModified(epsilon, maxIter, 0.001, 1000)
print(optimum2)


# **Solving with Relative Value Iteration (RVI)**

# It is also possible to solve an Average criteria MDP with the *relative Value Iteration* algorithm. 

# In[ ]:


print("Computation with relative value iteration")
optimum3 = mdp1.RelativeValueIteration(epsilon, maxIter)
print(optimum3)


# ## Policy (FeedbackSolutionMDP) handling

# The *FeedbackSolutionMDP* class is used to represent any deterministic Markov decision rule. A Feedback object stores both a decision rule (*action* field) and the associated value function (*value* field). Hence, it stores information about the policy, such as the actions to be taken in each state and the value associated with the policy. It can be manipulated and modified using setter and getter functions.

# ### Describing the output of  FeedbackSolutionMDP in the average case

# The fields of a `FeedbackSolutionMDP` object are filled in, for each state, 
# 
# - with the index of the state 
# - with the bias value
# - with the optimal action
# 
# which are calculated by the solution algorithms.  
# 
# The average optimal gain is given just before the state-by-state enumeration. This last point is only valid for average criterion MDPs. 

# ### Building Solution

# A FeedbackSolution is created during the running of the algorithm and is returned by the resolution methods. But the object can be directly manipulated. This is now described.

# **Creating a new Solution object**

# We create now a `FeedbackSolutionMDP` with dimension `stateSpace.Cardinal()`

# In[ ]:


policy =  md.FeedbackSolutionMDP(stateSpace.Cardinal())


# #### Accessors

# **Defining a given policy (setters)**

# The following lines of code define the policy actions for each state. The arguments passed to the `setActionIndex` method are :  
# 
# - The first argument is the index of the state for which you want to define the action.
# - The second argument is the index of the action you want to assign to this state.
# 
# For example: For state 0, the action assigned is action 0.

# In[ ]:


policy.setActionIndex(0,0)
policy.setActionIndex(1,0)
policy.setActionIndex(2,1)
policy.setActionIndex(3,2)


# It is also possible to put all the values of `FeedbackSolutionMDP` to zeros by the method `resetValues`. If we now print the variable politicy then the actions will be the ones described above and the values will all be zero. 

# In[ ]:


policy.resetValue()
print(policy)


# **Getting the values of a given policy (getters)**

# Information about the average value can be retrieved using `getAvgCost()`, as well as information about values using `getValueIndex` or policies using ` getValuePolicy`, as illustrated below. 

# In[ ]:


print("Getting Average Cost of policy",policy.getAvgCost())
print("Getting value in 0:", policy.getValueIndex(0))
print("Getting value in 1:", policy.getValueIndex(1))
print("Getting value in 2:", policy.getValueIndex(2))
print("Getting value in 3:", policy.getValueIndex(3))


# ### Assessing a policy  

# A policy can also be evaluated independently of any search for the optimal policy. The policyCost method is used to evaluate a policy whose action values are defined in the action element of a FeedbackPolicy object. The values of the object will be set to 0 at the start of the calculation so that the calculation of the average value of this policy does not depend on the value element.  
# 
# Important: the policyCost method is implemented in each MDP object (with an algorithm adapted to the model in each case).

# In[ ]:


mdp1.PolicyCost(policy, epsilon, maxIter)
print(policy)


# **Computing average costs of a some given policies**

# In[ ]:


print("Define Policy Ra")
politique =  md.FeedbackSolutionMDP(stateSpace.Cardinal())
politique.setActionIndex(0,0)
politique.setActionIndex(1,0)
politique.setActionIndex(2,0)
politique.setActionIndex(3,2)

print("Print solution Ra")
mdp1.PolicyCost(politique,epsilon, maxIter)
print(politique)

print("Modify the previous Policy and a defining a new policy Rc")
politique.setActionIndex(0,0)
politique.setActionIndex(1,0)
politique.setActionIndex(2,2)
politique.setActionIndex(3,2)


politique.resetValue()
print("Print solution of Rc")
mdp1.PolicyCost(politique,epsilon, maxIter)
print(politique)

print("Define Policy Rd")
politique.setActionIndex(0,0)
politique.setActionIndex(1,2)
politique.setActionIndex(2,2)
politique.setActionIndex(3,2)

print("Print solution of Rd")
mdp1.PolicyCost(politique,epsilon, maxIter)
print(politique)


# ## Structural Analysis

# The marmoteMDP software also integrates a set of functions for processing and studying structural properties of value function or policy as presented in the book *Monotonicity in Markov Reward and Decision Chains: Theory and Applications (Foundations and Trends in Stochastic Systems)* of G. Koole

# ### Structural analysis of the value

# The structural analysis of a value function is carried out using a `PropertiesValue` object, which is constructed from a state space. This object has two methods (depending on the properties to be checked): `Monotonicity` and `Convexity` which checks the property of the solution given in parameter. These two functions :  
# 
# -`Monotonicity` returns 1 (if the VF is increasing), 0 (VF has no property), -1 (if the VF is decreasing)  
# -`Convexity` returns 1 (if the VF is convex), -1 (if the VF is concave), 0 otherwise
# 
#  Some of the details can be clarified with the methods `avoidDetail` and `GetDetail`, in particular the indices for which the properties are broken.

# In[ ]:


checkValue =  md.PropertiesValue(stateSpace)
checkValue.avoidDetail()
monotone=checkValue.Monotonicity(optimum)
print("Printing monotonicity property of value function (1 if increasing -1 if decreasing 0 otherwise) : " + str(monotone))

print("Verif convexity with details")
checkValue.getDetail()
convex=checkValue.Convexity(optimum)
print("Printing convexity property of value function (1 if convex -1 concave 0 otherwise) : " + str(convex))


# ### Structural analysis of the policy

# The structural analysis of a property is carried out using a `PropertiesValue` object, which is constructed from a state space. This object has two methods (depending on the properties to be checked): `Monotonicity` or ̀̀`sSpol` for checking is the policy is *(s,S)*.

# In[ ]:


print("Checking Structural Properties of value")
checkPolicy =  md.PropertiesPolicy(stateSpace)

monotone=checkPolicy.Monotonicity(optimum)
print("PropertiesPolicy::MonotonicityOptimalPolicy="+str(monotone) + " (1 if increasing -1 if decreasing 0 otherwise) : ")


# End of the notebook

# In[ ]:




