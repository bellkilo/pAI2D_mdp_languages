#!/usr/bin/env python
# coding: utf-8

# # Lesson 3: working with state spaces

# In[ ]:


import marmote.core as marmotecore
import marmote.markovchain as mmc
import numpy as np


# The basic set in `Marmote` is an interval

# ## Functionalities of MarmoteSet: example with intervals and boxes

# Basic sets in `Marmote` are intervals and "boxes": the object classes are
# `MarmoteInterval` and `MarmoteBox`.

# Creation of an (integer) interval between 0 and 6 included.

# In[ ]:


itl = marmotecore.MarmoteInterval(0,6)


# In[ ]:


print( "name = ", itl.className())
print( "cardinal = ", itl.Cardinal())
print(itl)


# Creation of a 2-d box with size 2 in the first dimension and 3 in the second one.
# Note that ranges will be 0..1 and 0..2.

# In[ ]:


box = marmotecore.MarmoteBox( [ 2, 3 ] )


# Inspecting the propreties of the object

# In[ ]:


print( "name = ", box.className())
print( "cardinal = ", box.Cardinal())
print(box)


# Listing all the states

# In[ ]:


box.Enumerate()


# ## States and their indices

# Every state has an index and conversely. Indices range from 0 to Cardinal()-1.
# The relevant methods are:
# 
# * `Index( state )` to get the index of some state
# * `DecodeState( idx )` to get the state corresponding to some index

# In[ ]:


box.Index( [1,0] )


# In[ ]:


box.DecodeState( 4 )


# ### Printing states

# When printing states, it is possible to get both its index information and its description.
# 
# `Marmote` has a "policy" for writing states which is controlled as follows. First we save the current policy
# using the method `stateWritePolicy()`.

# In[ ]:


polsave = marmotecore.stateWritePolicy()


# Next we manipulate the way states are printed with the method `setStateWritePolicy()`. We set in sequence the policy to its three possible values: `STATE_INDEX`, `STATE_BOTH` and `STATE_FULL`.

# In[ ]:


marmotecore.setStateWritePolicy( marmotecore.STATE_INDEX )
print( "Format with index: ", box.FormatState( 4 ) )
marmotecore.setStateWritePolicy( marmotecore.STATE_BOTH )
print( "Format with both: ", box.FormatState( 4 ) )
marmotecore.setStateWritePolicy( marmotecore.STATE_FULL )
print( "Format with full description: ", box.FormatState( 4 ) )


# Finally we restore the policy to what it was previously.

# In[ ]:


marmotecore.setStateWritePolicy(polsave)


# ## Looping through the states

# There exist other ways to inspect `MarmoteSet` objects and make the correspondence between *states* and *indices*.
# The following one uses the standard way to walk through a set using methods:
# 
# * `FirstState(state)` to initialize the state to the "zero state"
# * `Index(state)` to convert a state to its index
# * `NextState(state)` to move the state its follower in the set's order
# * `IsFirst(state)` to test if the state is the "zero state".
#   
# All use a `state` variable which is an array of integers. Some methods *modify* this array.

# In[ ]:


marmotecore.setStateWritePolicy( marmotecore.STATE_BOTH )
bbuf = box.StateBuffer()
box.FirstState(bbuf)           # the buffer is modified!
isover = False
while not isover:
    idx = box.Index(bbuf)
    print( box.FormatState( box.Index(bbuf) ) )
    box.NextState(bbuf)        # the buffer is modified!
    isover = box.IsFirst(bbuf)


# ## Other sets

# The complete hierarchy of `MarmoteSet` objects is currently:
# 
# * `EmptySet`
# * `MarmoteInterval`
#   + `Integers`
# * `MarmoteBox`
# * `BinarySequence`
# * `BinarySimplex`
# * `Simplex`

# ### Binary sequences

# In[ ]:


biseq = marmotecore.BinarySequence( 6 )


# In[ ]:


print( "name = ", biseq.className())
print( "cardinal = ", biseq.Cardinal())
biseq.Enumerate()


# In[ ]:


bbuf = biseq.StateBuffer()
biseq.FirstState(bbuf)
isover = False
while not isover:
    print( biseq.FormatState( biseq.Index(bbuf) ) )
    biseq.NextState(bbuf)
    isover = biseq.IsFirst(bbuf)


# Illustration of membership tests. When a state does not belong to the set, its index is 0 by convention and a warning is issued.

# In[ ]:


st1 = np.array( [1, 0, 1, 1, 0, 1], dtype=int )
print( biseq.Belongs( st1 ) )
print( biseq.Index( st1 ) )


# In[ ]:


st2 = np.array( [2, 4, 1, 1, 0, 1], dtype=int )
print( biseq.Belongs( st2 ) )
print( biseq.Index( st2 ) )


# ### The set of integers

# `Marmote` has also a set for all integers!

# In[ ]:


itg = marmotecore.MarmoteIntegers()


# In[ ]:


print( "name = ", itg.className())
print( "cardinal = ", itg.Cardinal()) ## -2 is the convention for 'infinite cardinal'
print( itg.IsFinite() )


# Membership tests

# In[ ]:


print( itg.Belongs( [4] ) )
print( itg.Belongs( [-3] ) )


# In[ ]:


print( itg.Index( [-3] ) )


# ### Simplices

# Simplex sets are sometimes difficult to enumerate. Not with `Marmote`!

# In[ ]:


splx = marmotecore.Simplex( 7, 3 )


# In[ ]:


splx.Enumerate()


# In[ ]:


print( splx.Belongs( [ 0, 0, 0, 0, 0, 0 ] ) )
print( splx.Belongs( [ 0, 0, 0, 0, 0, 0, 3 ] ) )
print( splx.Belongs( [ 0, -1, 1, 0, 0, 0, 3 ] ) )


# In[ ]:


bsplx = marmotecore.BinarySimplex( 7, 3 )


# In[ ]:


bsplx.Enumerate()


# More set manipulations to be seen in App_Lesson1!
