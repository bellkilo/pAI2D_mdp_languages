#!/usr/bin/env python
# coding: utf-8

# # Lesson 5: Predefined Markov Chains

# In[ ]:


import marmote.core as marmotecore
import marmote.markovchain as mmc
import numpy as np


# Some of the well-known Markov models are implemented in `Marmote` as objects heritating from `MarkovChain`. Their current list is:
# 
# * `TwoStateDiscrete`
# * `TwoStateContinuous`
# * `Homogeneous1DRandomWalk`
# * `Homogeneous1DBirthDeath`
#   * `PoissonProcess`
# * `HomogeneousMultiDRandomWalk`
# * `HomogeneousMultiDBirthDeath`
# * `MMPP` 

# ## The continuous-time birth-death process

# ### Infinite-state birth-death process

# In[ ]:


mc = mmc.Homogeneous1DBirthDeath( 0.5, 1.0 )


# Although it has theoretically an infinite state space, this chain can be simulated.
# 
# Simulating the chain with options:
# 
# * 10.0 units of time
# * no stats collected
# * no trajectory kept in memory
# * trajectory printed along the way
# * increments in times displayed as well

# In[ ]:


simres = mc.SimulateChain( 10.0, stats=False, traj=False, _print=True, withIncrements=True )


# Computing the stationary distribution. It is well-known that this distribution is geometric.

# In[ ]:


stadis = mc.StationaryDistribution()


# In[ ]:


print(stadis)
print(stadis.Mean())
print(stadis.getProba(0))
print(stadis.getProba(1))
print(stadis.Ccdf(4))


# Embedding and Uniformizing.
# 
# Uniformizing results in another well-known Markov chain.

# In[ ]:


umc = mc.Uniformize()
print(umc.className())


# However embedding does not result in a standard Markov chain.

# In[ ]:


mc.Embed()


# ### Finite-space birth-death proces

# A birth-death process with 11 states

# In[ ]:


mcf = mmc.Homogeneous1DBirthDeath( 11, 0.4, 0.8 )


# In[ ]:


stadisf = mcf.StationaryDistribution()


# In[ ]:


print(stadisf)


# In[ ]:


print(stadisf.Mean())
print(stadisf.getProba(0))
print(stadisf.getProba(1))
print(stadisf.Ccdf(4))


# Investigating hitting times.
# 
# First set up the hitting set. Here, states 3 and 7 are in the set: \[ - - - X - - - X - - - \]

# In[ ]:


hitSet = np.zeros( [11], dtype=bool)
hitSet[3] = True
hitSet[7] = True
print(hitSet)


# Average hitting times are computed (numerically in this case).

# In[ ]:


avg = mcf.AverageHittingTime( hitSet )
print(avg)


# Also available, joint expected hitting time and hitting state: E( tau * ind(X(tau)=s) )

# In[ ]:


avgcon = mcf.AverageHittingTime_Conditional( hitSet )
print( avgcon[0:10,2:8] )


# Also available: simulation of hitting times.

# In[ ]:


simres = mcf.SimulateHittingTime( 0, hitSet, 20, 10000 )


# In[ ]:


simres.Diagnose()


# In[ ]:


print(simres.CT_dates())


# ### Multidimensional birth-death

# Consider a 3-d birth-death process on a 4 x 4 x 4 box.
# 
# The constructor to `HomogeneousMultiDBirthDeath` has arguments: sizes, birth rates, death rates.

# In[ ]:


mdbd = mmc.HomogeneousMultiDBirthDeath( [ 4, 4, 4 ], [ 1.0, 1.0, 1.0], [ 0.8, 0.8, 0.2 ] )


# In[ ]:


simres = mdbd.SimulateChain( 10.0, stats=True, traj=False, withIncrements=True, _print=True )


# Statistics can be performed on this trajectory also

# In[ ]:


sd = simres.Distribution()


# ## Arrival processes

# There are two pure arrival processes in `Marmote`: `PoissonProcess` and `MMPP`.

# In[ ]:


poi = mmc.PoissonProcess( 1.0 )


# In[ ]:


simres = poi.SimulateChain( 10.0, False, True, True )


# Creating an MMPP with modulating chain the 8-state birth-death of this lesson. Arrival rates are 0 except in states 1 and 7.

# In[ ]:


mmpp = mmc.MMPP( mcf.generator(), [ 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0 ] )


# Simulation stating from 0 arrivals and state 0 of the environment. 
# 
# Coding of 'state' trajectory (experimental) is 2*nb of arrivals + current state of the environment.

# In[ ]:


simres = mmpp.SimulateChain( 10.0, marmotecore.DiracDistribution(0.0), marmotecore.DiracDistribution(0.0), False, True, True )

