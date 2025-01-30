#!/usr/bin/env python
# coding: utf-8

# # MDP Lesson 4:  Total Reward MDP with two-dimensional state space

# ### Using the library

# In[ ]:


import marmote.core as mc 
import marmote.mdp as md


# In[ ]:


import numpy as np


# ## Build a MDP associated with a Stochastic Shortest Path

# ### Description of the model

# We consider the (classic) four rooms model which is represented on the figure below. This model is related to a *Stochastic Shortest Path*. The state space is divided into 4 rooms, each with (5 x 5) positions. You can only move from one room to another at particular locations. There is also an exit state from the system, which yields a gain. Moving from one position to another incurs a cost.

# ![FourRooms.png](attachment:FourRooms.png)

# More Precisely:  
# 
# We assume that the state space has two dimensions 11 lines x 10 columns ([0,10]*[0,9])  
# 
# If we reach the state (9,2) we receive a reward of -1 and we go in state (10,2). In any state in line 10 we stay in this state without receiving anything (absorbing states).
#  
# There is wall between line 4 and 5 that can be crossed   
#  -> at states (4,2)->(5,2) and (4,7)->(5,7)    
#  -> at states (5,2)->(4,2) and (5,7)->(4,7)  
# 
# There is a wall between column 4 and 5 that can be crossed   
#  -> at states (2,4)->(2,5) and (7,4)->(7,5)  
#  -> at states (2,5)->(2,4) and (7,5)->(7,4)  
#    
# There are 4 actions: 0 is up 1 is down 2 is left and 3 is right. With a probability *p=0.9* the action has the desired effect and with probability *1-p* the action has no effect. 
# Performing an action in any state except (9,2) has a cost of 1.

# ## Multidimensional State Space

# In all what follows we consider a two dimensionals state space with first dimension equals to 11 and second dimension equal to 10.

# ### Creating Spaces

# **Definitions of the state**

# The object used to create the state space is a `MarmoteBox` with dimension 11 x 10. 
# First define the dimensions of the box in a numpy array

# In[ ]:


dims = np.array([11, 10])


# Then create the box (and illustrate this by printing the cardinal, the dimension and the object)

# In[ ]:


stateSpace = mc.MarmoteBox(dims)

dim_SS = stateSpace.Cardinal()
print("State Space cardinal",dim_SS)
print("State Space dimension",stateSpace.tot_nb_dims())
print("State Space type",stateSpace)


# Create the action space as an interval between 0 and 3

# In[ ]:


actionSpace = mc.MarmoteInterval(0,3)
dim_AS = actionSpace.Cardinal()


# **Associate a state to a variable** 

# It is possible to associate a state either to a buffer of to a numpy array (both ways are used in this notebook). The buffer or the array stores the different values of a given state.
# Furthermore to each state is associated an index. it is then possible to pass from a state to an index and conversely.
# 
# Here one creates two arrays: one to represent the initial state (before transition) and the second one to represent the state after transition. The state is *(0,0)* in the following instruction.

# In[ ]:


# First tab allows to manage initial state 
etat=np.array([0,0])
# Second tab allows to manage final state (after transition)
sortie=np.array([0,0])


# **Filling Cost matrix**

# We fill in the matrix : all the costs are equal to 1 except in 9,2 in which it is equal to -1 and in line 10 in which it is equal to 0

# In[ ]:


print("Fill in Cost Matrix")
CostMat  =  mc.FullMatrix(dim_SS, dim_AS)

#definition of some indexes
k=0  # to iterate on states
l=0 # to iterate on lines
c=0 # to iterate on columns
indexO=0 # indexes of the initial state
indexD=0 # indexes of the final state


stateSpace.FirstState(etat)
for k in range(dim_SS):
        # computing the index of the state
        indexO=stateSpace.Index(etat)
        # for each state we give a value of any action
        for a in range(dim_AS) :
                CostMat.setEntry(indexO,a,1)
        stateSpace.NextState(etat)

# replacing the term in (9,2) with action UP to -1
etat[0]=9
etat[1]=2
indexO=stateSpace.Index(etat)
print("index of state (9,2) ",indexO,"\n")
CostMat.setEntry(indexO,0,-1.0)

# Fill in the line 10 all costs are equal to zero
# Define the line
etat[0]=10
for k in range(10):
    etat[1]=k
    indexO=stateSpace.Index(etat)
    CostMat.setEntry(indexO,0,0.0)
    CostMat.setEntry(indexO,1,0.0)
    CostMat.setEntry(indexO,2,0.0)
    CostMat.setEntry(indexO,3,0.0)


# ### Use a third way to build the MDP

# In what follows, we use a constructor that does not require to build the list of matrices. They are added one by one.  
# Please notice that the name of the matrices should differ.

# In[ ]:


# create and initialize elements of MDP
criterion="min"

print("Begining of building MDP")
# create the MDP
mdpSSP =  md.TotalRewardMDP(criterion, stateSpace, actionSpace, CostMat)
print("End of building MDP")


# Then we fill in the transition matrices. Then we have four matrices to fill in.
# 
# First we complete matrix for action 0 (UP). All the states are browsed by iterating over the rows and columns. For each new state value, the index is calculated, then the possible output states and their indexes are computed to fill in the entry. 

# In[ ]:


print("Add matrices")


p=0.9

P0 =  mc.SparseMatrix(dim_SS)
for l in range(10) :
        for c in range(10):
                # define a state and get its index
                etat[0]=l # initialize the value of the first dim
                etat[1]=c # initialize the value of the second dim
                indexO=stateSpace.Index(etat)
                if (( l==4) or (l==9)) :
                        if  ( (l==4) and  ( (c==2) or (c==7)) ) :
                                # I am on a door two possibilities
                                # either I move up with probability p
                                sortie[0]=l+1
                                sortie[1]=c
                                indexD=stateSpace.Index(sortie) #computing the index of destination
                                P0.setEntry(indexO,indexD,p)
                                # either I stay
                                P0.setEntry(indexO,indexO,1-p)
                        else:
                                if  ( (l==9) and (c==2)):
                                # I am on a door two possibilities
                                    # either I move up with probability p
                                        sortie[0]=l+1
                                        sortie[1]=c
                                        indexD=stateSpace.Index(sortie) #computing the index of destination
                                        P0.setEntry(indexO,indexD,p)
                                #either I stay
                                        P0.setEntry(indexO,indexO,1-p)
                                else:
                                # I am on the wall l=4 or l=9 I stay in the same state
                                        P0.setEntry(indexO,indexO,1.0)
                else:
                #I am whatever in a rooms I have two possibilities
                # either I move up with probability p
                        sortie[0]=l+1
                        sortie[1]=c
                        indexD=stateSpace.Index(sortie) #computing the index of destination
                        P0.setEntry(indexO,indexD,p)
                #either I stay
                        P0.setEntry(indexO,indexO,1-p)

# fill in last line
for c in range(10):
        # define a state and get its index
                etat[0]=10 # initialize the value of the first dim
                etat[1]=c # initialize the value of the second dim
                indexO=stateSpace.Index(etat)
                P0.setEntry(indexO,indexO,1.0)


mdpSSP.AddMatrix(0,P0)
print("Added matrix (action 0)")


# We complete the remaining matrices

# In[ ]:


# Complete matrix for action 1 (DOWN)
P1 =  mc.SparseMatrix(dim_SS)

for l in range(10) :
        for c in range(10) :
                # define a state and get its index
                etat[0]=l # initialize the value of the first dim
                etat[1]=c # initialize the value of the second dim
                indexO=stateSpace.Index(etat)
                if ((l==5) or (l==0)) :
                        if ((l==5) and ( (c==2) or (c==7)) ) :
                                #two possibilities I am on a door
                                #either I move down with probability p
                                sortie[0]=l-1
                                sortie[1]=c
                                indexD=stateSpace.Index(sortie) #computing the index of destination
                                P1.setEntry(indexO,indexD,p)
                                #either I stay
                                P1.setEntry(indexO,indexO,1-p)
                        else:
                                        #I am on the wall l=5 or l=0 I stay in the same state
                                        P1.setEntry(indexO,indexO,1.0)
                else:
                                # I am whatever in a rooms I have two possibilities
                                # either I move down with probability p
                                # or I stay with a probability 1-p
                                sortie[0]=l-1
                                sortie[1]=c
                                indexD=stateSpace.Index(sortie) #computing the index of destination
                                P1.setEntry(indexO,indexD,p)
                                # either I stay
                                P1.setEntry(indexO,indexO,1-p)

# fill in last line
for c in range(10) :
        # define a state and get its index
        etat[0]=10 # initialize the value of the first dim
        etat[1]=c # initialize the value of the second dim
        indexO=stateSpace.Index(etat)
        P1.setEntry(indexO,indexO,1.0)


mdpSSP.AddMatrix(1,P1)
print("Added matrix (action 1)")

# Define matrix for action 2 (LEFT)
P2 =  mc.SparseMatrix(dim_SS)
for l in range(10) :
        for c in range(10) :
                # define a state and get its index
                etat[0]=l # initialize the value of the first dim
                etat[1]=c # initialize the value of the second dim
                indexO=stateSpace.Index(etat)
                if ((c==5) or (c==0)) :
                        if ((c==5) and ( (l==2) or (l==7)) ) :
                                # I am on a door
                                # either I move left with probability p
                                # or I stay with propability 1-p
                                sortie[0]=l
                                sortie[1]=c-1
                                indexD=stateSpace.Index(sortie) #computing the index of destination
                                P2.setEntry(indexO,indexD,p)
                                #either I stay
                                P2.setEntry(indexO,indexO,1-p)
                        else :
                                # I am on the wall c=5 or c=0 I stay in the same state
                                P2.setEntry(indexO,indexO,1.0)
                else :
                        #I am whatever in a rooms I have two possibilities
                        #either i move left with probability p
                        sortie[0]=l
                        sortie[1]=c-1
                        indexD=stateSpace.Index(sortie) #computing the index of destination
                        P2.setEntry(indexO,indexD,p)
                        # or I stay
                        P2.setEntry(indexO,indexO,1-p)

# fill in last line
for c in range(10) :
        # define a state and get its index
        etat[0]=10 # initialize the value of the first dim
        etat[1]=c # initialize the value of the second dim
        indexO=stateSpace.Index(etat)
        P2.setEntry(indexO,indexO,1.0)

mdpSSP.AddMatrix(2,P2)
print("Added matrix (action 2)")

# Define matrix for action 3 (RIGHT)
P3 =  mc.SparseMatrix(dim_SS)
for l in range(10) :
        for c in range(10) :
                # define a state and get its index
                etat[0]=l # initialize the value of the first dim
                etat[1]=c # initialize the value of the second dim
                indexO=stateSpace.Index(etat)
                if ((c==4) or (c==9)) :
                        if ((c==4) and ( (l==2) or (l==7)) ) :
                                # I am on a door
                                # either I move right with probability p
                                # or I stay with propability 1-p
                                sortie[0]=l
                                sortie[1]=c+1
                                indexD=stateSpace.Index(sortie) #computing the index of destination
                                P3.setEntry(indexO,indexD,p)
                                # either I stay
                                P3.setEntry(indexO,indexO,1-p)
                        else :
                # I am on the wall c=4 or c=9 I stay in the same state
                                P3.setEntry(indexO,indexO,1.0)
                else :
                        # I am whatever in a rooms I have two possibilities
                        # either i move left with probability p
                        sortie[0]=l
                        sortie[1]=c+1
                        indexD=stateSpace.Index(sortie) #computing the index of destination
                        P3.setEntry(indexO,indexD,p)
                        # either I stay
                        P3.setEntry(indexO,indexO,1-p)

# fill in last line
for c in range(10) :
        # define a state and get its index
        etat[0]=10 # initialize the value of the first dim
        etat[1]=c # initialize the value of the second dim
        indexO=stateSpace.Index(etat)
        P3.setEntry(indexO,indexO,1.0)


mdpSSP.AddMatrix(3,P3)
print("Added matrix (action 3)")


print("Finishing Adding matrices MDP")
print("Writing MDP")
print(mdpSSP)


# In[ ]:


# create and initialize elements of solving
epsilon = 0.0001
maxIter=250

print("\nPrinting solution from value iteration")
optimum2 = mdpSSP.ValueIteration(epsilon, maxIter)
print(optimum2)


# ### Print policy dimension by dimension

# A `FeedbackSolutionMDP` can be printed dimension by dimension with method `SolutionByDim` whose first parameter is the dimension to be scanned. Below we scan the columns. THe line is fixed and we let vary the columns.

# In[ ]:


print("Print solution by dimension (line by line)")
line=optimum2.SolutionByDim(1,stateSpace)
print(line)


# ### Enumerating the policy

# We can scan the policy by the way of the *iterator* of space. We also create a *buffer* to store the state.

# In[ ]:


#create the buffer
bbuf = stateSpace.StateBuffer()
print("Printing State Space Path and value function with a browsing by iterating space")
# initial state (bbuf receive the value of the first state of the state space
stateSpace.FirstState(bbuf)
# scan
for k in range(stateSpace.Cardinal()):
        # getting the index of the state
        indexO = stateSpace.Index(bbuf)
        # the different values of the states are in the array
        l=bbuf[0] # getting value of the first dimension of the box
        c=bbuf[1] # getting value of the second dimension of the box    
        print("--State in line=", l , " column=", c, end = ' ')
        if ((c<=4) and (l<=4)) :
                print(" --in Room at Bottom Left  ", end = ' ')
        if ((c<=4) and (l>=5)) :
                print(" --in Room at Top Left     ", end = ' ')
        if ((c>=5) and (l<=4)) :
                print(" --in Room at Bottom Right ", end = ' ')
        if ((c>=5) and (l>=5)) :
                print(" --in Room at Top Right    ", end = ' ')
        # getting the values and the action at the index of the state
        print( " --Optimal Action=" , optimum2.getActionIndex(indexO) , " --Value=" , optimum2.getValueIndex(indexO) )
        # Move to next state
        stateSpace.NextState(bbuf)


# In[ ]:





# In[ ]:




