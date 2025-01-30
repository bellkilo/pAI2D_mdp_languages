#!/usr/bin/env python
# coding: utf-8

# # Application Lesson 1: Application to Mitrani's hysteresis model
# <img name="Mitrani_chain.jpg">

# In[ ]:


import marmote.core as marmotecore
import marmote.markovchain as marmotemarkovchain
import numpy as np


# ### Picture of the model (credit: Mitrani 2013, Figure 2)

# <img src="./Mitrani_chain.png">

# Defining the constants/parameters of the model

# In[ ]:


m = 4          # Number of reserve servers
n = 2          # Number of permanent servers
N = m+n        # Total number of servers
lambda_ = 3.0  # Arrival rate
mu_ = 1.0      # Individual server service rate
nu_ = 5.0      # Inverse of average warmup time of the block of m reserve servers
up = 6         # Upper threshold
down = 4       # Lower threshold
K = 10         # Queue truncation parameter


# In[ ]:


space = marmotecore.MarmoteBox( [ 3, K+1 ] )


# In[ ]:


space.Enumerate()


# In[ ]:


Qtrans = marmotecore.SparseMatrix( space )
Qtrans.set_type( marmotecore.CONTINUOUS )


# In[ ]:


Qtrans


# ## Filling the matrix

# ### Naming elements to make code readable

# In[ ]:


# naming of coordinates
QUEUE = 1
SERV = 0
# naming of server states
SLOW = 0
WARMING = 1
FAST = 2


# ### Setting up the loop on states

# In[ ]:


stateBuffer = space.StateBuffer()
nextBuffer = space.StateBuffer()
space.FirstState(stateBuffer)
idx = 0
print(stateBuffer)


# ### Looping

# In[ ]:


looping = True
while looping:
    # print( "Transitions for state ", stateBuffer )
    # convenience local variables, also used for restoring stateBuffer
    q = stateBuffer[QUEUE]
    s = stateBuffer[SERV]
    
    nextBuffer = np.array(stateBuffer) # copy current state
    # Event: arrivals
    if ( q < K ):
      nextBuffer[QUEUE] += 1;
      if ( ( nextBuffer[QUEUE] > up ) and ( nextBuffer[SERV] == SLOW ) ):
        nextBuffer[SERV] = WARMING;
      
      Qtrans.addToEntry( idx, space.Index(nextBuffer), lambda_ );
      Qtrans.addToEntry( idx, idx, -lambda_ );
    
    nextBuffer = np.array(stateBuffer) # copy current state
    # Event: departure
    if ( q > 0 ):
        # number of active servers
        if ( s == FAST ):
            nbServ = min( q, n+m )
        else:
            nbServ = min( q, n )
        nextBuffer[QUEUE] -= 1
        if ( nextBuffer[QUEUE] == down ): # whatever state of server: becomes SLOW
            nextBuffer[SERV] = SLOW;
      
        Qtrans.addToEntry( idx, space.Index(nextBuffer), mu_ * nbServ )
        Qtrans.addToEntry( idx, idx, -mu_ * nbServ )
    
    nextBuffer = np.array(stateBuffer) # copy current state
    # Event: end of warmup
    if ( s == WARMING ):
        nextBuffer[SERV] = FAST
        Qtrans.addToEntry( idx, space.Index(nextBuffer), nu_ )
        Qtrans.addToEntry( idx, idx, -nu_ )
    
    # next state
    space.NextState( stateBuffer )
    idx += 1
    looping = not space.IsFirst(stateBuffer)


# ### Inspecting the matrix. 
# 
# The `FullDiagnose` method produces a summary of various metrics **plus** the structural analysis of the matrix (recurrent/transient classes).

# In[ ]:


marmotecore.setStateWritePolicy( marmotecore.STATE_INDEX )
Qtrans.FullDiagnose()


# In[ ]:


marmotecore.setStateWritePolicy( marmotecore.STATE_BOTH )
Qtrans.FullDiagnose()


# In[ ]:


hymc = marmotemarkovchain.MarkovChain( Qtrans )
hymc.set_model_name( "Hysteresis_box")


# In[ ]:


print( [ hymc.IsAccessible(3,31), hymc.IsAccessible(3,6), hymc.IsAccessible(1,30), hymc.IsAccessible(30,27) ] )
print( hymc.IsIrreducible() )


# ## Stationary distribution and average cost
# 
# Let us compute the stationary distribution.

# In[ ]:


dista = hymc.StationaryDistribution()


# As expected, the distribution has many zeroes.

# In[ ]:


print(dista)


# In[ ]:


# print( Qtrans.toString( marmotecore.FORMAT_NUMPY ) )


# Computing an average linear cost.

# In[ ]:


def avg_cost( c1, c2, dis ):
    avgL = 0
    avgS = 0
    space.FirstState(stateBuffer)
    # print(stateBuffer)
    go_on = True
    index = 0
    while go_on:
        prob = dis.getProbaByIndex(index)
        nbServ = n
        if ( stateBuffer[SERV] == FAST ):
            nbServ += m
        avgL = avgL + prob*stateBuffer[QUEUE]
        avgS = avgS + prob*nbServ
        space.NextState(stateBuffer)
        # print(stateBuffer)
        index = index+1
        go_on = not space.IsFirst(stateBuffer)
    return( c1*avgL + c2*avgS )


# In[ ]:


cost = avg_cost( 1.0, 1.0, dista )
print(cost)


# And checking that the distribution is consistent.

# In[ ]:


total = 0
for i in range(space.Cardinal()):
    total = total + dista.getProbaByIndex(i)
print(total)


# ## Tayloring the state space
# 
# Now defining a state space which fits exactly the recurrent class

# In[ ]:


smaller_space = marmotecore.MarmoteUnionSet()
smaller_space.AddSet( marmotecore.MarmoteInterval(0,up) )
smaller_space.AddSet( marmotecore.MarmoteInterval(down+1,K) )
smaller_space.AddSet( marmotecore.MarmoteInterval(down+1,K) )


# In[ ]:


smaller_space.Enumerate()


# In[ ]:


smaller_space.Cardinal()


# In[ ]:


smaller_space.Belongs( [ 2, 3 ] )


# Defining a new matrix on this smaller state space, and filling it.

# In[ ]:


Qtrans_alt = marmotecore.SparseMatrix( smaller_space )
Qtrans_alt.set_type( marmotecore.CONTINUOUS )


# In[ ]:


# naming of coordinates
QUEUE = 1
SERV = 0
# naming of server states
SLOW = 0
WARMING = 1
FAST = 2
# preparation of buffers
stateBuffer = smaller_space.StateBuffer()
smaller_space.FirstState(stateBuffer)
looping = True
idx = 0
while looping:
    # print( "Transitions for state ", stateBuffer )
    # convenience local variables, also used for restoring stateBuffer
    q = stateBuffer[QUEUE]
    s = stateBuffer[SERV]

    nextBuffer = np.array(stateBuffer) # copy current state
    # Event: arrivals
    if ( q < K ):
      nextBuffer[QUEUE] += 1;
      if ( ( nextBuffer[QUEUE] > up ) and ( nextBuffer[SERV] == SLOW ) ):
        nextBuffer[SERV] = WARMING;
      
      Qtrans_alt.addToEntry( idx, smaller_space.Index(nextBuffer), lambda_ );
      Qtrans_alt.addToEntry( idx, idx, -lambda_ );
      # print( stateBuffer, " to ", nextBuffer )
      # print( smaller_space.Belongs(nextBuffer), smaller_space.Index(nextBuffer) )
    
    nextBuffer = np.array(stateBuffer) # copy current state
    # Event: departure
    if ( q > 0 ):
        # number of active servers
        if ( s == FAST ):
            nbServ = min( q, n+m )
        else:
            nbServ = min( q, n )
        nextBuffer[QUEUE] -= 1
        if ( nextBuffer[QUEUE] == down ): # whatever state of server: becomes SLOW
            nextBuffer[SERV] = SLOW;
      
        Qtrans_alt.addToEntry( idx, smaller_space.Index(nextBuffer), mu_ * nbServ )
        Qtrans_alt.addToEntry( idx, idx, -mu_ * nbServ )
    
    nextBuffer = np.array(stateBuffer) # copy current state
    # Event: end of warmup
    if ( s == WARMING ):
        nextBuffer[SERV] = FAST
        Qtrans_alt.addToEntry( idx, smaller_space.Index(nextBuffer), nu_ )
        Qtrans_alt.addToEntry( idx, idx, -nu_ )
    
    # next state
    smaller_space.NextState( stateBuffer )
    idx += 1
    looping = not smaller_space.IsFirst(stateBuffer)


# In[ ]:


print( Qtrans_alt.toString( marmotecore.FORMAT_NUMPY ) )


# Using the "full state" feature to inspect in detail the transitions.

# In[ ]:


print( Qtrans_alt.toString( marmotecore.FORMAT_FULLSTATE ) )


# Looking at the structural analysis.
# 
# Now all states are recurrent.

# In[ ]:


Qtrans_alt.FullDiagnose()


# Indeed, irreducibility can be tested on Markov chains with the `IsIrreducible()` method.

# In[ ]:


hymc_alt = marmotemarkovchain.MarkovChain( Qtrans_alt )
print( hymc_alt.IsIrreducible() )
print( hymc.IsIrreducible() )


# Computing the stationary distribution of this new chain, then computing the average cost.

# In[ ]:


dista_alt = hymc_alt.StationaryDistribution()


# In[ ]:


print(dista_alt)
total = 0
for i in range(smaller_space.Cardinal()):
    total = total + dista_alt.getProbaByIndex(i)
print(total)


# In[ ]:


def avg_cost_alt( c1, c2, dis ):
    avgL = 0
    avgS = 0
    smaller_space.FirstState(stateBuffer)
    # print(stateBuffer)
    go_on = True
    index = 0
    while go_on:
        prob = dis.getProbaByIndex(index)
        nbServ = n
        if ( stateBuffer[SERV] == FAST ):
            nbServ += m
        avgL = avgL + prob*stateBuffer[QUEUE]
        avgS = avgS + prob*nbServ
        smaller_space.NextState(stateBuffer)
        # print(stateBuffer)
        index = index+1
        go_on = not smaller_space.IsFirst(stateBuffer)
    return( c1*avgL + c2*avgS )


# In[ ]:


cost_alt = avg_cost_alt( 1.0, 1.0, dista_alt )


# In the end we have the same cost

# In[ ]:


print(cost_alt)
print( cost )


# In[ ]:




