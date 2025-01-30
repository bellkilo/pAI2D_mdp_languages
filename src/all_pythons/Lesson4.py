#!/usr/bin/env python
# coding: utf-8

# # Lesson 4: working with Distributions

# In[ ]:


import marmote.core as marmotecore
import marmote.markovchain as mmc
import numpy as np


# Distributions are everywhere in `Marmote`: as inputs and as outputs of many procedures.
# 
# The complete hierarchy of distributions is currently:
# 
# * `Distribution`
#   * `DiscreteDistribution`
#     * `DiracDistribution`
#     * `BernoulliDistribution`
#     * `UniformDiscreteDistribution`
#     * `PoissonDistribution`
#     * `ShiftedGeometricDistribution`
#       * `GeometricDistribtion`
#     * `PhaseTypeDiscreteDistribution`
#   * `GammaDistribution`
#     * `ErlangDistribution`
#       * `ExponentialDistribution`
#   * `GaussianDistribution`
#   * `UniformDistribution`
#   * `PhaseTypeDistribution`

# ## Common Features

# ### Statistical information

# `Distribution` objects have all usual statistical methods, plus Laplace transform ones useful in some stochastic modeling.
# 
# The following code is self-explanatory.

# In[ ]:


udis = marmotecore.UniformDistribution( 4, 10 )
print( udis.className() )
print( udis.Mean() )
print( udis.Rate() )         ## Rate is the inverse of Mean
print( udis.Variance() )
print( udis.Cdf(6.0) )
print( udis.Ccdf(6.0) )
print( udis.Laplace(0.01) )  ## Laplace transform (only at real values)
print( udis.DLaplace(0.01) ) ## DLaplace computes the derivative of the Laplace transform


# In[ ]:


uddis = marmotecore.UniformDiscreteDistribution( 4, 10 )
print( uddis.className() )
print( uddis.Mean() )
print( uddis.Rate() )
print( uddis.Variance() )
print( uddis.Cdf(6.0) )
print( uddis.Ccdf(6.0) )
print( uddis.Laplace(0.01) )
print( uddis.DLaplace(0.01) )


# In[ ]:


edis = marmotecore.ExponentialDistribution( 4.0 )
print( edis.Mean() )
print( edis.Rate() )
print( edis.Variance() )
print( edis.Cdf(6.0) )
print( edis.Ccdf(6.0) )
print( edis.Laplace(0.0) )
print( edis.DLaplace(0.0) )


# ### Some structural information

# Some properties can be tested (mostly useful for `Marmote` developpers).

# In[ ]:


print( edis.HasProperty("integerValued") )
print( edis.HasProperty("continuous"))
print( uddis.HasProperty("integerValued") )
print( uddis.HasProperty("compactSupport") )


# ### Sampling

# Distributions can be sampled.

# In[ ]:


for i in range(4):
    print( edis.Sample() )
print()
for i in range(4):
    print( udis.Sample() )
print()
for i in range(4):
    print( uddis.Sample() )


# ## Some distributions can be manipulated

# Rescaling is possible for some families of distributions.

# In[ ]:


print( udis )
print( udis.Rescale( 0.5 ) )


# In[ ]:


print( edis )
print( edis.Rescale( 5.0 ) )


# In[ ]:


didis = marmotecore.DiscreteDistribution( [ 3.4, 4.5, 6.7, 8.9 ], [ 0.1, 0.2, 0.3, 0.4 ] )
print( didis )
print( didis.Rescale(10.0) )


# But rescaling does not work for all distributions...

# In[ ]:


print( uddis )
try:
    print( uddis.Rescale( 0.5 ) )
except:
    pass


# ## Comparison of distributions

# The distance between some of the classes of distributions can be computed.
# Available distances are:
# 
# * L1 distance
# * L2 distance
# * L-infinity distance
# * Total variation distance

# In[ ]:


d1 = marmotecore.UniformDiscreteDistribution( 0, 19 )
d2 = marmotecore.GeometricDistribution( 0.5 )
d3 = marmotecore.UniformDiscreteDistribution( 0, 24 )
d4 = marmotecore.GeometricDistribution( 0.55 )


# In[ ]:


print( "L1 = ", marmotecore.Distribution.DistanceL1( d1, d2 ) )
print( "L2 = ", marmotecore.Distribution.DistanceL2( d1, d2 ) )
print( "Linf = ", marmotecore.Distribution.DistanceLInfinity( d1, d2 ) )
print( "TV = ", marmotecore.Distribution.DistanceTV( d1, d2 ) )


# In[ ]:


print( "Computable L-infinity distance:", marmotecore.Distribution.DistanceLInfinity( d1, d3 ) )
try:
    print( "Not computable distance:", marmotecore.Distribution.DistanceLInfinity( d2, d4 ) )
except:
    pass


# ## Markov Chain operations which return distributions

# ### State distributions in Markov Chains

# Example: with a 4-state continuous birth-death Markov Chain

# In[ ]:


four = mmc.Homogeneous1DBirthDeath( 4, 3.0, 1.0 )


# Distributions of one-step transitions

# In[ ]:


print( four.generator().TransDistrib(0) )
print( four.generator().TransDistrib(1) )
print( four.generator().TransDistrib(2) )
print( four.generator().TransDistrib(3) )


# Transient and stationary distributions

# In[ ]:


print( four.TransientDistribution(4) )
print( four.StationaryDistribution() )


# Empirical state distributions through simulation

# In[ ]:


simres = four.SimulateChain( 20, True, False, False, False )
print( simres.Distribution() )


# ### Hitting time distributions

# For some Markov chains, hitting time distributions are avaliable.
# 
# More on these special Markov chains in Lesson5.

# In[ ]:


two = mmc.TwoStateContinuous( 5.0, 1.0 )


# In[ ]:


hitset = np.array( [ 0, 1 ], dtype=bool )
hd = two.HittingTimeDistribution( hitset )


# In[ ]:


print( hd[0] )
print( hd[1] )


# In[ ]:


f81 = mmc.Felsenstein81( [ 0.1, 0.2, 0.3, 0.4 ], 1.0 )


# In[ ]:


hitset = np.array( [ False, False, True, False ], dtype=bool )
hd = f81.HittingTimeDistribution( hitset )
print( hd[0] )
print( hd[1] )
print( hd[2] )
print( hd[3] )

