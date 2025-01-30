#!/usr/bin/env python
# coding: utf-8

# # Application Lesson 2: Application to the control of a tandem  multi server system

# In[ ]:


import marmote.core as mco
import marmote.markovchain as mch
import marmote.mdp as md
import numpy as np


# ## The model

# We consider the tandem multi server queue model (below) (credit of the picture  [Tournaire 2021]) presented also in [Ohno and  Ichiki, 1987], or [Tournaire 2023].

# ![tandemQueue.jpg](attachment:tandemQueue.jpg)

# **Parameters**
# 
# Size of the systems: B1 and B2 with B1=B2=B
# 
# Number of servers: K1 and K2 with K1=K2=K
# 
# Statistics:  
# arrival  Poisson with rate lambda  
# service homogeneous with rate mu1 and mu2  
# 
# Costs:  
# Instantaneous costs:  
# activation cost: *Ca*, deactivation cost: *Cd*, reject cost: *Cr*.   
# Rates costs (also called accumulated cost):    
# cost per time unit of using a VM: *Cs*, cost per time unit of holding a request in the system *Ch*.
# 
# **Numerical values**
# 
# B=10  
# K=5  
# lam=5  
# mu1=1  
# mu2=1   
# Ca=1  
# Cd=1  
# Cr=10  
# Cs=2  
# Ch=2

# Build the model with a python dictionary

# In[ ]:


model=dict()
model['B1']=3 #10
model['B2']=3 #10
model['K1']=2 #5
model['K2']=2 #5
model['lam']=5.0
model['mu1']=1.0
model['mu2']=1.0
model['Ca']=1  
model['Cd']=1 
model['Cr']=10 
model['Cs']=2 
model['Ch']=2
model['beta']=1

print(model)


# ## Build a discrete time discounted MDP

# ### Build the states

# The state is *(m1,k1,m2,k2)* with m from 0 to B-1 and  k from 0 to K-1. 
# An action is *(k1,k2)* with k from 0 to K-1.

# In[ ]:


dims=np.array([model['B1'],model['K1'],model['B2'],model['K2']])
print(dims) 
states= mco.MarmoteBox(dims)
#
actions=mco.MarmoteBox([model['K1'],model['K2']])

print("Number of states",states.Cardinal())
print(states)
print("Number of actions",actions.Cardinal())
print("actions",actions)


# ### Build matrices

# #### Transitions matrices

# We begin by defining a function which computes the transition matrix associated with an action such that the action index is: index_action.
# 
# In a state, there is three types of event: arrival in the system, departure of the system, departure of the system 1 and arrival in system 2.

# In[ ]:


def fill_in_matrix(index_action,modele,ssp,asp):
    # retrieve the action asscoiated with index
    action_buf = asp.DecodeState(index_action)
    #*#print("index action",index_action,"action",action_buf)
    #define the states
    etat=np.array([0,0,0,0])
    afteraction=np.array([0,0,0,0])
    jump=np.array([0,0,0,0])
    # define transition matrix
    P=mco.SparseMatrix(ssp.Cardinal()) 
    #browsing state space
    ssp.FirstState(etat)
    for k in range(ssp.Cardinal()):
        # compute the index of the state
        indexL=ssp.Index(etat)
        # compute the state after the action
        afteraction[0]=etat[0]
        afteraction[1]=action_buf[0]
        afteraction[2]=etat[2]
        afteraction[3]=action_buf[1]
        #*# print("####index State=",k,"State",etat,"State after action",afteraction)
        # then detail all the possible transitions
        ## Arrival (increases the number of customer in first coordinate with rate lambda)
        if (afteraction[0]<modele['B1']-1) :
            jump[0]=afteraction[0]+1
            jump[1]=afteraction[1]
            jump[2]=afteraction[2]
            jump[3]=afteraction[3]
        else: 
            jump[0]=afteraction[0]
            jump[1]=afteraction[1]
            jump[2]=afteraction[2]
            jump[3]=afteraction[3]
        #compute the index of the jump
        indexC=ssp.Index(jump)
        #fill in the entry
        #*# print("*Event: Arrival. Index=",indexC,"Jump State=",jump,"rate=",modele['lam'])
        P.setEntry(indexL,indexC,modele['lam'])
        #
        ## departure of the first system entry in the second one
        if (afteraction[2]<modele['B2']-1) :
            jump[0]=max(0,afteraction[0]-1)
            jump[1]=afteraction[1]
            jump[2]=afteraction[2]+1
            jump[3]=afteraction[3]
        else: 
            jump[0]=max(0,afteraction[0]-1)
            jump[1]=afteraction[1]
            jump[2]=afteraction[2]
            jump[3]=afteraction[3]
        #index of the jump
        indexC=ssp.Index(jump)
        # rate of the transition
        rate=min(afteraction[1],afteraction[0])*modele['mu1']
        #fill in the entry
        #*# print("*Event: Departure s1 entry s2. Index=",indexC,"Jump State=",jump,"rate=",rate)
        P.setEntry(indexL,indexC,rate)
        #
        ##departure of the second  system
        jump[0]=afteraction[0]
        jump[1]=afteraction[1]
        jump[2]=max(0,afteraction[2]-1)
        jump[3]=afteraction[3]
        #compute the index of the jump
        indexC=ssp.Index(jump)
        # compute the rate
        rate=min(afteraction[2],afteraction[3])*modele['mu2']
        #fill in the entry
        #*# print("*Event: Departure s2. Index=",indexC,"Jump State=",jump,"rate=",rate)
        P.setEntry(indexL,indexC,rate)
        #change state
        ssp.NextState(etat)
    return P


# #### Cost Matrix

# We define now a function to fill in the cost matrix. 
# 
# Instantaneous Costs are:   
# Costs of activations = *max(action1-k1,0) \* Ca + max(action2-k2,0) \* Ca*  
# Costs of deactivations = *max(K1-action1,0) \* Cd + max(K2-action2,0) \* Cd*  
# rejection cost= *Cr \* lambda/Lambda(s,a)* in states where *m1=B* added by  *Cr \* action1 mu/Lambda(s,a)* in states where *m2=B*.  
# *Lambda(s,a)* is the total rate. It is equal to *lambda + action1 \* mu + action2 \* mu* .
# 
# Accumulated Costs are:  
# (number of customers in the system)Ch = *(m1+m2)\*Ch*   
# (number of activated VM) = *(action1+action2)\*Cs*  

# In[ ]:


def fill_in_cost(modele,ssp,asp):
    R= mco.FullMatrix(ssp.Cardinal(),asp.Cardinal())
    #define the states
    etat=np.array([0,0,0,0])
    #define the actions
    acb=asp.StateBuffer()
    ssp.FirstState(etat)
    for k in range(ssp.Cardinal()):
        # compute the index of the state
        indexL=ssp.Index(etat)
        #*#print("##State",etat)
        asp.FirstState(acb)
        for j in range(asp.Cardinal()):
            #*#print("---Action",acb,end='  ')
            action1=acb[0]
            action2=acb[1]
            totalrate=modele['lam']+action1*modele['mu1']+ action2*modele['mu2']
            activationcosts=modele['Ca']*(max(0,action1-etat[1])+max(0,action2-etat[3])) 
            deactivationcosts=modele['Cd']*(max(0,etat[1]-action1)+max(0,action2-etat[3]))
            rejectioncosts=0.0
            if ((modele['B1']-1)==etat[0]):
                rejectioncosts+=(modele['lam']*modele['Cr']) / totalrate
            if ((modele['B2']-1)==etat[2]):
                rejectioncosts+=( min(etat[0],action1)*modele['mu1']*modele['Cr']) / totalrate
            instantaneouscosts=activationcosts+deactivationcosts+rejectioncosts
            accumulatedcosts=(etat[0]+etat[2])*modele['Ch'] + (action1 +action2)*modele['Cs']
            accumulatedcosts/=(totalrate+model['beta'])
            #*#print("Instantaneous=",instantaneouscosts," Rejection=",rejectioncosts,end= ' ')
            #*#print("Accumulatedcosts=",accumulatedcosts)
            R.setEntry(indexL,j,accumulatedcosts+instantaneouscosts)
            asp.NextState(acb)
        ssp.NextState(etat)
    return R;


# ### Build the continuous time MDP

# Build all the transition matrices

# In[ ]:


trans=list()

action_buf = actions.StateBuffer()
actions.FirstState(action_buf)
for k in range(actions.Cardinal()):
    trans.append(fill_in_matrix(k,model,states,actions))
    print("---Matrix kth=",k, "filled in")


# Fill in the costs

# In[ ]:


print("Matrice of Costs")
Costs=fill_in_cost(model,states,actions)


# Build the MDP

# In[ ]:


print("Begining of Building MDP")
ctmdp=md.ContinuousTimeDiscountedMDP("min",states,actions,trans,Costs,model['beta'])
print(ctmdp)


# Uniformization of the MDP. After uniformization the MDP is a discrete time discounted MDP.

# In[ ]:


ctmdp.UniformizeMDP()
print("Rate of Uniformization",ctmdp.getMaximumRate())
#*# print(ctmdp)


# ### Solve the MDP

# In[ ]:


optimum=ctmdp.ValueIteration(0.01,75)
print(optimum)


# ## Structural Analysis

# The structural analysis is mainly related to the policy handling. In what follow we :
# 
# 1. Check property of the MDP by building a Markov Chain Associated with a policy
# 2. Check property of the value function 

# ### Check property using Markov Chain analysis

# #### Check if the MDP is "multichain"

# Actually the multichain property is useless for discounted criteria and is solely valid for average multichain criteria. This is presented here for an example purpose. 
# 
# To assess the property, we build a special policy

# In[ ]:


policy=md.FeedbackSolutionMDP(states.Cardinal())


# Now we fill-in policy. The policy is defined as foolows: in any states such that the number of customer is less than 2 the server si activated and deactivated otherwise. 

# In[ ]:


etat=states.StateBuffer()
states.FirstState(etat)
for k in range(states.Cardinal()):
    if(etat[0]==(model['B1']-1) or etat[2]==(model['B2']-1) ):
        policy.setActionIndex(k,0)
    else :
        policy.setActionIndex(k,1)
    states.NextState(etat)
print(policy)


# **Build a Markov Chain from a policy**

# In[ ]:


Mat=ctmdp.GetChain(optimum)
Mat.set_type(mco.DISCRETE)
#*# print(Mat)

initial = mco.UniformDiscreteDistribution(0,states.Cardinal()-1)


# Making the chain

# In[ ]:


chaine = mch.MarkovChain( Mat )
chaine.set_init_distribution(initial)
chaine.set_model_name( "Chain issued from the MDP")


# **Analysis of the transition matrix**

# In[ ]:


Mat.FullDiagnose()


# **Evaluate the policy**
# 
# Now we can evaluate the policy by the way of the `PolicyCost` method

# In[ ]:


ctmdp.PolicyCost(policy,0.01,75)
print(policy)


# ### Check if the value function has structural property (convex,monotone)

# This is done by building a specific object `PropertiesValue`.

# In[ ]:


checkValue =  md.PropertiesValue(states)
checkValue.avoidDetail()
monotonicity=checkValue.Monotonicity(optimum)
print("Printing monotonicity property of value function (1 if increasing -1 if decreasing 0 otherwise) : "\
      + str(monotonicity) )

print("Checking convexity")
convexity=checkValue.Convexity(optimum)
print("Printing convexity property of value function (1 if convex -1 concave 0 otherwise) : " + \
      str(convexity))


# The analysis can be made dimension by dimension. Now we check the monotonicty of the first dimension by letting vary the entries with index 0 and keeping the other dimensions fixed.

# In[ ]:


monotonicity=checkValue.MonotonicityByDim(optimum,0)
print("Following dimension 0 monotonicity is",str(monotonicity))


# ### Check if the optimal policy has structural property

# The structural analysis of a policy property is carried out using a `PropertiesValue` object.

# In[ ]:


print("Checking Structural Properties of value")
checkPolicy =  md.PropertiesPolicy(states)

monotonicity=checkPolicy.Monotonicity(optimum)
print("PropertiesPolicy::MonotonicityOptimalPolicy="+str(monotonicity))


# End
