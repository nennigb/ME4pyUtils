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
        npa = np.frombuffer(ma._real,dtype=nptype).reshape(ma.size,order='F') + 1j*np.frombuffer(ma._imag,dtype=nptype).reshape(ma.size,order='F')
    else:
        # tuple that define type
        mltype=(ma._data.typecode,ma._data.itemsize)
        # use test to define the type, few are missing like uint!!
        if mltype==('d',8): # double
            nptype = 'f8'
        elif mltype==('B',1): # logical is given as a int
            nptype = 'bool'
        elif mltype==('b',1): #int8
            nptype = 'i1'
        elif mltype==('i',4):
            nptype = 'i4'
        elif mltype==('i',8): # int64
            nptype = 'i8'
        elif mltype==('l',8): # int64
            nptype = 'i8'        
        else:
            nptype = 'f8' #default
            
            
        # no copy with the buffer
        npa = np.frombuffer(ma._data,dtype=nptype).reshape(ma.size,order='F')
        
    return npa



def np2mlarray(npa):
    """
    Convert  numpy array to matlab
    npa : must be a numpy ndarray and return (without copy for real case) a matlab mlarray
    For now only double, double complex and int64 are
    """

    # check input type
    if 'ndarray' not in str(type(npa)):
        raise TypeError('Expected  numpy.ndarray. Got %s' % type(ma))
    
   
    
    # complex case
    # =========================================================================
    # create empty matlab.mlarray
    if npa.dtype == np.complex128:
         # convert to array
         arr=np.asarray(npa.flatten('F').real)
         ari=np.asarray(npa.flatten('F').imag)
         ma= matlab.double(initializer=None, size=(1,len(arr)), is_complex=True)
         # associate the data
         ma._real=arr
         ma._imag=ari
         # reshape
         ma.reshape(npa.shape)
    
    else:
        
        # real case
        # =========================================================================
        # convert to array
        ar=np.asarray(npa.flatten('F'))
        # create empty matlab.mlarray
        if npa.dtype == np.float64:        
            ma= matlab.double(initializer=None, size=(1,len(ar)), is_complex=False)
        elif npa.dtype == np.int64:
            ma= matlab.int64(initializer=None, size=(1,len(ar)))
        elif npa.dtype == np.bool:
            ma= matlab.logical(initializer=None, size=(1,len(ar)))
        else:
            raise TypeError('Type is missing')
        
        # assocaite data
        ma._data=ar
        # reshape
        ma.reshape(npa.shape)
    
    return ma

# ============================================================================
#  M A I N
# ============================================================================

if __name__ == "__main__":
    """ 
    Test of the module    
    """
    import timeit 
    
    
    

    
    # Connect to matlab
    try:
        eng
    except NameError:
         print('Run matlab engine...')
         if len(matlab.engine.find_matlab())==0:
             #si aucune session share, run
             eng=matlab.engine.start_matlab()
         else:
             # connect to a session
             eng=matlab.engine.connect_matlab(matlab.engine.find_matlab()[0])
             print('connected...')
    else:
         print('Matlab engine is already runnig...')
       
    
    # create matlab data
    # ------------------------------------------------------------------------
    mf = eng.rand(3)
    mc = matlab.double([[1+1j, 0.3, 1j],[1.2j-1,0,1+1j]],is_complex=True)
    mi64 = matlab.int64([1,2,3])
    mi8 = matlab.int8([1,2,3])
    mb = matlab.logical([True,True,False])
    
    # Test conversion from matlab to numpy
    # ------------------------------------------------------------------------    
    npf= mlarray2np(mf)
    npc = mlarray2np(mc)
    npi64= mlarray2np(mi64)
    npi8= mlarray2np(mi8)
    npb = mlarray2np(mb)
    
    # Test conversion from numpy to matlab 
    # ------------------------------------------------------------------------
    npi=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],dtype=np.int64)
    mc2 = np2mlarray(npc)
    mf2 = np2mlarray(npf)
    mi64_2 = np2mlarray(npi)   
    mb2 = np2mlarray(npb)

    # test orientation in the matlab workspace    
    # ------------------------------------------------------------------------
    eng.workspace['mi']=mi64_2
    eng.workspace['mc2']=mc2

    # Test for speed
    # ------------------------------------------------------------------------
    print "Compare perf vs matlab.double(np_a.tolist())"
    setup_np2mat = (
    "import numpy as np\n"
    "import matlab\n"
    "import ME4pyUtils\n"
    "import array\n"
    "np_a=np.random.uniform(size=(1000))\n")
    np_a=np.random.uniform(size=(1000))
    tstd = timeit.timeit('mat_a = matlab.double(np_a.tolist())',setup=setup_np2mat,  number=1000)
    tnew = timeit.timeit('mat_a = ME4pyUtils.np2mlarray(np_a)',setup=setup_np2mat,  number=1000)
    print(' >  matlab.double(np_a.tolist()) =' + str(tstd) + ' s')
    print(' >  ME4pyUtils.np2mlarray(np_a)=' + str(tnew) + ' s')
    
    