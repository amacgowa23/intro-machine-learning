# -*- coding: utf-8 -*-
"""
Created on Sat Jan 30 13:23:47 2016

@author: AMacGowan
"""

""" quiz materials for feature scaling clustering """

### FYI, the most straightforward implementation might 
### throw a divide-by-zero error, if the min and max
### values are the same
### but think about this for a second--that means that every
### data point has the same value for that feature!  
### why would you rescale it?  Or even use it at all?
def featureScaling(arr):        
    maxv = max(arr)
    minv = min(arr)
    fprime = []
    if (maxv - minv) > 0:
        for xval in arr:
            xprime = float(xval - minv) / float(maxv - minv)
            fprime.append(xprime) 
    else:
        fprime.append(arr[0])

    return fprime

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)