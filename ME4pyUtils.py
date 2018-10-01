# -*- coding: utf-8 -*-
"""
This file is part of ME4PyUtils, This module aims to ease some data type 
conversion between matlab engine API for python and numpy python module.
Copyright (C) 2018 -- Benoit Nennig, benoit.nennig@supmeca.fr

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
import numpy as np
import matlab.engine

def mlarray2np(ma):
    """
    Convert matlab array to numpy
    ma : must be a matlab array and return (without copy for real case) a numpy ndarray
    The np ndarray type depend of the matlab data, /!\ all the type are not ex int8.
    """
    # add test si autre type de données int...
    
    # check input type, isintance    
    # if type(ma) is not type(matlab.double()):
    if 'mlarray' not in str(type(ma)):
        raise TypeError('Expected matlab.mlarray. Got %s' % type(ma))
    
    """
    for real:    
    ma._data.itemsize # get item size
    ma._data.typecode # get item type
    ma._python_type
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.dtype.html#numpy.dtype
    https://docs.python.org/3/library/array.html
    
    numpy kind 
    
        A character code (one of ‘biufcmMOSUV’) identifying the general kind of data.
        b 	boolean
        i 	signed integer
        u 	unsigned integer
        f 	floating-point
        c 	complex floating-point
        m 	timedelta
        M 	datetime
        O 	object
        S 	(byte-)string
        U 	Unicode
        V 	void
    """    
    
    
    # conversion using FROM BUFFER, need to be carefull with type
    if ma._is_complex==True:  
        nptype='f8'
        npa = np.frombuffer(ma._real,dtype=nptype).reshape(ma.size) + 1j*np.frombuffer(ma._imag,dtype=nptype).reshape(ma.size)
    else:
        # tuple that define type
        mltype=(ma._data.typecode,ma._data.itemsize)
        # use test to define the type, few are missing like uint!!
        if mltype==('d',8): # double
            nptype = 'f8'
        elif mltype==('B',1): # logical
            nptype = 'b'
        elif mltype==('b',1): #int8
            nptype = 'i1'
        elif mltype==('i',4):
            nptype = 'i4'
        elif mltype==('i',8): # int64
            nptype = 'i8'
        elif mltype==('l',8): # int64
            nptype = 'i8'        
        else:
            nptype = 'f8'
            
            
        # no copy with the buffer
        npa = np.frombuffer(ma._data,dtype=nptype).reshape(ma.size)
        
    return npa



# ============================================================================
#  M A I N
# ============================================================================

if __name__ == "__main__":
    """ 
    Test of the module    
    """
   
    n=np.ones(3)
    
    try:
        eng
    except NameError:
         print('Run matlab engine...')
         eng=matlab.engine.start_matlab()
    else:
         print('Matlab engine is already runnig...')
       
    
    # create matlab data
    mf = eng.rand(3)
    mc = eng.cos(matlab.double([1+1j, 0.3, 1j],is_complex=True))
    mi64 = matlab.int64([1,2,3])
    mi8 = matlab.int8([1,2,3])
    mb = matlab.logical([True,True,False])
    
    # Test conversion from matlab to numpy
    npf= mlarray2np(mf)
    npc = mlarray2np(mc)
    npi64= mlarray2np(mi64)
    npi8= mlarray2np(mi8)
    npb = mlarray2np(mb)
    
    # Test conversion from numpy to matlab 
    
    # Test for speed