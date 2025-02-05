#!/usr/bin/env python
# coding: utf-8

# # Lesson 2: solving Markov chains

# In[ ]:


import marmote.core as mc
import marmote.markovchain as mmc
import numpy as np


# In Lesson1, we have seen how to create and inspect Markov chains. We now illustrates the different metrics that can be computed on them
# (what is called "the solution").

# ## First example: solving discrete-time Markov chains

# We first (re)create the 3-state discrete-time Markov chain of Lesson 1

# In[ ]:


states = np.array( [0, 1, 2] )
n = states.shape[0]
P = mc.FullMatrix(n)
P.set_type(mc.DISCRETE)
P.setEntry(0,0,0.25)
P.setEntry(0,1,0.5)
P.setEntry(0,2,0.25)
P.setEntry(1,0,0.4)
P.setEntry(1,1,0.2)
P.setEntry(1,2,0.4)
P.setEntry(2,0,0.4)
P.setEntry(2,1,0.3)
P.setEntry(2,2,0.3)
initial_prob = np.array( [0.2, 0.2, 0.6] )
initial = mc.DiscreteDistribution(states, initial_prob)


# In[ ]:


c1 = mmc.MarkovChain( P )
c1.set_init_distribution(initial)
c1.set_model_name( "Demo" )


# ### Transient distributions

# To compute transient distributions, the method to be called is `TransientDistributionDT` with argument the 'time' or number of steps.

# In[ ]:


pi1 = c1.TransientDistributionDT( 1 )
pi2 = c1.TransientDistributionDT( 2 )
pi3 = c1.TransientDistributionDT( 3 )


# In[ ]:


print( pi1 )
print( pi2 )
print( pi3 )


# It is of course possible to change the initial distribution. Other distributions of the `DiscreteDistribution` family can be used.<br>
# A typical one is `DiracDistribution`.

# In[ ]:


pi0 = mc.DiracDistribution(0)
c1.set_init_distribution(pi0)
pi1 = c1.TransientDistributionDT( 1 )
pi2 = c1.TransientDistributionDT( 2 )
pi3 = c1.TransientDistributionDT( 3 )
print( pi0 )
print( pi1 )
print( pi2 )
print( pi3 )


# Another possibility is `UniformDiscreteDistribution`.

# In[ ]:


pi0 = mc.UniformDiscreteDistribution(0,2)
c1.set_init_distribution(pi0)
pi1 = c1.TransientDistributionDT( 1 )
pi2 = c1.TransientDistributionDT( 2 )
pi3 = c1.TransientDistributionDT( 3 )
print( pi0 )
print( pi1 )
print( pi2 )
print( pi3 )


# ### Stationary distribution

# Computing the stationary distribution of a Markov chain is a typical activity of the Markov modeler.
# `Marmote` provides several ways to perform this task.

# There exists a default method `StationaryDistribution()` for users who don't want to bother about details.

# In[ ]:


pista = c1.StationaryDistribution()


# In[ ]:


print(pista)


# This is an iterative and approximate method. To see that the result is not exact, use the `TransientDistribution()` method to perform one step of the transition matrix. Then compute the distance between the two distributions.

# In[ ]:


c1.set_init_distribution( pista )
dis = c1.TransientDistributionDT(1)
print( "Distance between pi and pi.P:", mc.DiscreteDistribution.DistanceL1( pista, dis ) )


# `Marmote` provides an improved iterative method, the **red light green light** algorithm recently published by Brown, Avrachenkov and Nitvak.

# In[ ]:


pista2 = c1.StationaryDistributionRLGL( 100, 1e-10, mc.UniformDiscreteDistribution(0,2), False )


# In[ ]:


mc.DiscreteDistribution.DistanceL1( pista, pista2 )


# Of course, for such a small example, the exact distribution can be computed.
# Compare the approximate solution with the exact one.

# In[ ]:


prosta_ex = np.array( [ 8/23, 85/253, 80/253 ], dtype=float )
pista_ex = mc.DiscreteDistribution( states, prosta_ex )
print(pista_ex)


# In[ ]:


print( "Distance between numerical pi (standard) and exact pi:", mc.DiscreteDistribution.DistanceL1( pista, pista_ex ) )
print( "Distance between numerical pi (RLGL) and exact pi:", mc.DiscreteDistribution.DistanceL1( pista2, pista_ex ) )


# Finally, check that the exact stationary distribution is exact.

# In[ ]:


c1.set_init_distribution(pista_ex)
didi = c1.TransientDistributionDT(1)


# In[ ]:


mc.DiscreteDistribution.DistanceL1( pista_ex, didi )


# ### Simulation

# Simulation has plenty of controls. The basic syntax is `SimulateChain( nb_steps, stats, traj, trace )` where:
# 
# * `nb_steps` is the number of time instants to simulate
# * `stats` is a boolean specifying if occupancy statistics are to be kept
# * `traj` is a boolean specifying if a trajectory is to be kept
# * `trace` is a boolean specifying if the trajectory is to be printed along the way.

# For instance, a simulation of 10 time steps, no statistics, a trajectory which is not printed.

# In[ ]:


simRes = c1.SimulateChainDT( 10, stats=False, traj=True, trace=False )


# Inspecting the features of the simulation object: there is indeed a DT (discrete-time) trajectory, but no CT (continuous-time) trajectory.

# In[ ]:


simRes.Diagnose()


# Listing the details.

# In[ ]:


print( simRes.DT_dates() )
print( simRes.states() )
print( simRes.lastDate() )
print( simRes.lastState() )


# Running again with trajectory printed but not kept, and stats.
# 
# Observe the repeated last columns with states. More explanations on this below.

# In[ ]:


simRes = c1.SimulateChainDT( 10, True, False, True )


# The empirical distribution can be extracted from the `SimulationResult` object via its method `Distribution()`.
# This produces some information on the output and stores the result in a `DiscreteDistribution` object.

# In[ ]:


trDis = simRes.Distribution()


# In[ ]:


print( trDis )


# ## Second example: solving continuous-time Markov chains

# We first recreate the continuous-time Markov chain of Lesson 1

# In[ ]:


Q = mc.SparseMatrix(6)
Q.set_type(mc.CONTINUOUS)
Q.setEntry(0,1,1.0)
Q.setEntry(0,0,-1.0)
for i in range(1,6):
    if i > 0:
        Q.setEntry(i,0,1.0)
        Q.addToEntry(i,i,-1.0)
    if i < 5:
        Q.setEntry(i,i+1,1.0)
        Q.addToEntry(i,i,-1.0)


# In[ ]:


c2 = mmc.MarkovChain( Q )
c2.set_init_distribution(initial)
c2.set_model_name( "Demo_Continuous" )


# ### State distributions

# Standard call to stationary distribution method

# In[ ]:


stadis = c2.StationaryDistribution()
print(stadis)


# ### Simulation

# Simulation is as for discrete-time chains, but there is an additional control:
# `withIncrements` specifies whether time increments (sojourn time in each state) should be printed
# when tracing is enabled.

# Example of a simulation of 10 time units, with all details printed. 
# Each line of the output contains:
# 
# * the sequence number of the transition event inside square brackets
# * the time at which this event occured
# * the state *index* that *results* from the transition
# * the time increment between is event and the previous one
# * the state again, but now in full representation (in this example this is the same as the index).

# In[ ]:


simres = c2.SimulateChainCT( 10.0, stats=False, traj=True, withIncrements=True, trace=True )


# Observe the following conventions:
# 
# * the first event (#0) is always at time 0, and it results in a transition to the initial state;
# * the last date is exactly the simulation time specified;
# * the sojourn time in the state resulting from event #n is to be read in the following line: that of event #n+1.

# The trajectory has been kept. It can be accessed e.g. for post-processing.

# In[ ]:


print( simres.CT_dates() )
print( simres.states() )


# ### Hitting time distributions

# For general Markov chains, hitting time distributions can be simulated.
# Hitting time methods all require as argument a *hit set indicator* which is an array of booleans where `True` marks the states belonging to the hitting set.

# Here, the hitting set contains just the last state (index 5).

# In[ ]:


hitset = np.array( [ False, False, False, False, False, True ], dtype=bool )


# Hitting time simulation methods have several controls.
# The basic syntax is `SimulateHittingTime( init, hitset, sample_nb, max_time )` where:
# 
# * `init` specifies the initial distribution: either a state number or a `DiscreteDistribution` object
# * `hitset` is the hitting set indicator
# * `sample_nb` is the number of samples to be drawn
# * `max_time` is a time limit for simulations: hitting times larger than this value will not be found.

# Example: simulating 25 values of the hitting time from state 0.

# In[ ]:


simres = c2.SimulateHittingTime( 0, hitset, 25, 100  )


# The samples of the simulation are stored in the attribute `CT_dates` of the simulation object.

# In[ ]:


print( simres.CT_dates() )


# It is also possible to compute average hitting times, starting from all states.
# The method for this is `AverageHittingTime`. It takes the hitting set as unique argument.

# In[ ]:


avghit = c2.AverageHittingTime( hitset )
print( avghit )


# Comparing with the empirical average of the simulation.

# In[ ]:


avg = 0
for i in range(25):
    avg = avg + simres.CT_dates()[i]
print( "Empirical average of hitting time from state 0 =", avg/25 )

