#!/usr/bin/env python
# coding: utf-8

# # Lesson 1: making Markov chains

# In[ ]:


import marmote.core as mc
import marmote.markovchain as mmc
import numpy as np


# A Markov Chain is composed of
# + a **state space**
# + a **transition structure** (probability matrix or infinitesimal generator)
# + an **initial distribution** of the state
# 
# In this first lesson, we show various ways of creating such objects. We will in particular highlight *transition structures* and the class `MarkovChain`.

# ## First example: a discrete-time Markov chain with 3 states

# ### State space

# We first create a state space as vector of state indices.<br>
# The size `n` of this vector is needed in the following.

# In[ ]:


states = np.array( [0, 1, 2] )
n = states.shape[0]


# ### Transition structure

# Now create the transition structure and enter values.<br>
# Two objects are available for this: `SparseMatrix` and `FullMatrix`.<br>
# Let us start with a `FullMatrix`.

# In[ ]:


P = mc.FullMatrix(n)


# `Marmote` has to know if this is a discrete-time or continuous-time transition structure.

# In[ ]:


P.set_type(mc.DISCRETE)


# Now fill the values in. Two method are available for this: `setEntry` and `addToEntry`.
# Both have parameters `(row,column,value)`.

# In[ ]:


P.setEntry(0,0,0.25)
P.setEntry(0,1,0.5)
P.setEntry(0,2,0.25)
P.setEntry(1,0,0.4)
P.setEntry(1,1,0.2)
P.setEntry(1,2,0.4)
P.setEntry(2,0,0.4)
P.setEntry(2,1,0.3)
P.setEntry(2,2,0.3)


# We inspect the result.

# In[ ]:


print(P)


# ### Initial distribution

# Distributions exist as objects in `Marmote`. The most common one is `DiscreteDistribution` which represents general distributions on discrete state spaces. 
# It can be constructed from two arrays:
# * the state space array (already defined above)
# * the array of probabilities of these states

# In[ ]:


initial_prob = np.array( [0.2, 0.2, 0.6] )
initial = mc.DiscreteDistribution(states, initial_prob)


# Inspecting the object

# In[ ]:


print( initial )


# We will show more about `Distribution` objects in Lesson3.

# ### The Markov chain

# The object of type `MarkovChain` is created directly from the generator (probability transition matrix).<br>
# Then other (optional) features are specified:
# 
# * the initial distribution
# * the name of the Markov chain model

# In[ ]:


c1 = mmc.MarkovChain( P )
c1.set_init_distribution(initial)
c1.set_model_name( "Demo_Discrete" )


# Printing the Markov chain. Several formats are available. This one is the default `Marmote` format (it is adapted to sparse matrices).
# It lists:
# 
# * the generator (probability transition matrix or infinitesimal generator)
# * the initial distribution

# In[ ]:


print(c1)


# The characteristics of the object can be inspected.

# In[ ]:


print( c1.type(), " = ", mc.DISCRETE )             # The type of the chain (DISCRETE or CONTINUOUS) as a numerical representation
print( c1.state_space_size() )
print( c1.model_name() )


# ### Input/Output of Markov Chains and transition structures

# `Marmote` support the export of these objects in a variety of formats.

# Few languages have a format for Markov chains, but many have one for matrices.
# 
# For instance: with the Matlab format for sparse matrices, the numpy format, the R format and the Maple format.

# In[ ]:


print( c1.generator().toString( mc.FORMAT_MATLAB_SPARSE ) )


# In[ ]:


print( c1.generator().toString( mc.FORMAT_NUMPY ) )


# In[ ]:


print( c1.generator().toString( mc.FORMAT_R ) )


# In[ ]:


print( c1.generator().toString( mc.FORMAT_MAPLE ) )


# ## Second example with a continuous-time Markov chain

# Creating a continuous-time chain, this time with a `SparseMatrix` as transition stucture support.

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


print(Q)


# Creation of the Markov chain object

# In[ ]:


c2 = mmc.MarkovChain( Q )
c2.set_init_distribution(initial)
c2.set_model_name( "Demo_Continuous" )


# Inspection of its features

# In[ ]:


print( c2.type(), " = ", mc.CONTINUOUS )             # The type of the chain (DISCRETE or CONTINUOUS) as a numerical representation
print( c2.state_space_size() )
print( c2.model_name() )


# In[ ]:


c2.generator().className()
Q2 = c2.generator()
print( Q2.className() )
print( Q2.toString() )


# ## Transformation of Markov chains

# There are two well-known ways to transform continuous-time chains into discrete-time ones: **uniformization** and **embedding**.
# Both are implemented in Marmote.

# ### Uniformization

# In[ ]:


c2uni = c2.Uniformize()


# In[ ]:


print(c2uni)


# ### About the uniformization factor.

# The `MarkovChain` method `Uniformize` has no parameter: the uniformization factor is automatically chosen, as small as possible.

# In some cases, a finer control is needed. This can be done via the `TransitionStructure` objects.

# First inspect the uniformization rate chosen:

# In[ ]:


c2uni.generator().uniformization_rate()


# Redo uniformization with a larger rate.

# In[ ]:


c2uni2 = mmc.MarkovChain( c2.generator().Uniformize(4.0) )


# In[ ]:


mc.setInoutFormatPolicy(mc.FORMAT_NUMPY)
print(c2uni2.generator().toString())
print( "Unif rate = ", c2uni2.generator().uniformization_rate() )


# ### Embedding

# In[ ]:


c2embed = c2.Embed()
print(c2embed.generator().toString())


# 
