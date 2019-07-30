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

# import order as its importance with matlab lib...
from __future__ import print_function
import matlab
import matlab.engine
import numpy as np
import scipy.sparse as sparse


def mlarray2np(ma):
    """ Convert matlab mlarray to numpy array

    The conversion is realised without copy for real data thank to the frombuffer
    protocol. The np ndarray type depends on the type of matlab data.
    
    Paramerters
    -----------
    ma : mlarray
        the matlab array to convert
    
    Returns 
    -------
    npa : numpy array
        the converted numpy array 
        
    References
    ----------
    https://stackoverflow.com/questions/34155829/how-to-efficiently-convert-matlab-engine-arrays-to-numpy-ndarray/34155926
        
    """   
    
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
        #npa = np.frombuffer(ma._real,dtype=nptype).reshape(ma.size,order='F') + 1j*np.frombuffer(ma._imag,dtype=nptype).reshape(ma.size,order='F')        
        # New version avoid some computation, but still copy, due to complex contigous array ?
        npa = np.empty(ma.size, dtype=complex)
        npa.real = np.frombuffer(ma._real,dtype=nptype).reshape(ma.size,order='F')
        npa.imag = np.frombuffer(ma._imag,dtype=nptype).reshape(ma.size,order='F')


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
    """ Conversion of a numpy array to matlab mlarray
    
    The conversion is realised without copy. First an empty initialization is realized.
    Then the numpy array is affected to the _data field. Thus the data field is not really an 
    array.array but a numpy array. Matlab doesn't see anything...
    With 'F' order input , there is still a copy due to ravel().
    
    Paramerters
    -----------
    npa : numpy array
        the array to convert
    Returns 
    -------
    ma : mlarray
        the converted array that can be passe to matlab

    Examples
    --------
    >>>npc=np.array([[1.0,1.1+1j,12],[1+1j,2,100+110j]],dtype=np.complex) # assume C
    
    References
    -----------
    https://scipy-lectures.org/advanced/advanced_numpy/ (strides)
    
    """
    # check numpy
    if 'ndarray' not in str(type(npa)):
        raise TypeError('Expect  numpy.ndarray. Got %s' % type(npa))
       
    # TODO: with 'F' order, there is still a copy due to ravel(); no need to swap the strides...
    # there is a copy if F order...

    
    # get input size
    # TODO: not hard to generalize to nd>2
    if len(npa.shape)==1:
        nr,nc= 1, npa.shape[0]
    elif len(npa.shape)==2:
        nr,nc = npa.shape
    elif len(npa.shape)>2: 
        raise  TypeError('Expect numpy.ndarray of dimension <=2, got %s' % len(npa.shape))
    
    
    # complex case    
    if npa.dtype in (np.complex128,np.complex):
         #  create empty matlab.mlarray    
         ma= matlab.double(initializer=None,  size=(1,nc*nr), is_complex=True)
         # associate the data
         ma._real=npa.ravel() # allow to map real and imaginary part continuously!
       
    # real case  
    else:                     
        # create empty matlab.mlarray
        if npa.dtype == np.float64:        
            ma= matlab.double(initializer=None, size=(1,nc*nr), is_complex=False)
        elif npa.dtype == np.int64:
            ma= matlab.int64(initializer=None, size=(1,nc*nr))
        elif npa.dtype == np.bool:
            ma= matlab.logical(initializer=None, size=(1,nc*nr))
        else:
            raise TypeError('Type %s is missing' % npa.dtype)
        
        # associate the data
        ma._data=npa.ravel() 
        print(ma._data.flags,ma._data,'\n')
    # back to original shape    
    ma.reshape((nr,nc)) 
    # array strides are in number of cell (numpy strides are in bytes)
    ma._strides=[nc,1] # change stride (matlab use 'F' order)
       

    return ma
    
    
    
    
    
def dict2sparse(K):
    """Create a scipy sparse CSR matrix from dictionnary
    
    Paramerters
    -----------
    K : dictionnary K['i'], K['j'] and K['s']  
        The sparse matrix in the coo format. K['i'], K['j'] constains the row 
        and column index (int64) and the values K['s']  (double or complex) 
            
    Returns 
    -------
    Ksp :  sparse.csr_matrix
        The converted sparse matrix. csr is faster for computation.

    """    
    # get shape
    shape=tuple(K['shape']._data)
    # -1 because matlab index start to 1
    if len(K['i']._data)>1:
        Ksp = sparse.coo_matrix( ( mlarray2np(K['s']).ravel(),
                                   (mlarray2np(K['i']).ravel() -1 ,
                                    mlarray2np(K['j']).ravel() -1  ) 
                                  ), shape=shape).tocsr()
    else:
        raise TypeError('The sparse matrix contains just one element and matlab returns just scalar...')
        
                       
    return Ksp
    
    
# ============================================================================
#  M A I N
# ============================================================================

if __name__ == "__main__":
    """ 
    Test of the module    
    """
    import timeit 
    import scipy as sp
    speedtest=False # set to True or False to avoid speed test
    

    
    # Connect to matlab
    # move into a function
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
    npf= mlarray2np(mf)  # no copy, if mf is changed, npf change!
    npc = mlarray2np(mc) # still copy for complex (only)
    npi64= mlarray2np(mi64)
    npi8= mlarray2np(mi8)
    npb = mlarray2np(mb)
    
    # Test conversion from numpy to matlab 
    # ------------------------------------------------------------------------
    npi=np.array([[1,2,3],[4,5,6],[7,8,9],[10,11,12]],dtype=np.int64,order='C')
    mc2 = np2mlarray(npc)
    mf2 = np2mlarray(npf) # copy, because npf has 'F' order (comes from mlarray)
    mi64_2 = np2mlarray(npi)   
    mb2 = np2mlarray(npb)

    # test orientation in the matlab workspace    
    # ------------------------------------------------------------------------
    eng.workspace['mi']=mi64_2
    eng.workspace['mc2']=mc2
    
    # no copy check
    # ------------------------------------------------------------------------
    # complex
    npcc =np.array([[1.0,1.1+1j],[1.12+0.13j,22.1,]],dtype=np.complex) # assume C
    mcc = np2mlarray(npcc)
    npcc[0,0]=0.25    
    print("Are the data reuse ?", ", OWNDATA =", mcc._real.flags.owndata, 
          "same base =", mcc._real.base is npcc, 
          ', If one is modified, the other is modified =', mcc._real[0]==npcc[0,0])
    
    # check results
    # ------------------------------------------------------------------------
    npcc_inv = sp.linalg.inv(npcc)
    mcc_inv=eng.inv(mcc)
    print('Are the inverse of matrix equal ?')
    print(mcc_inv)
    print(npcc_inv)

    
    # test sparse matrix requiert Recast4py.m
    K1,K2=eng.sptest(3.,nargout=2)
    Ksp1=dict2sparse(K1)
    Ksp2=dict2sparse(K2)
    
    # Test for speed
    # ------------------------------------------------------------------------
    if speedtest:
        print( "\nCompare Numpy to matlab conversion strategy : (a bit long with several matlab.engine opening)")
        setup_np2mat = (
            "import numpy as np\n"
            "import matlab\n"
            "import ME4pyUtils\n"
            "import array\n"
            "np_a=np.random.uniform(size=(10000))*(.5+0.1236*1j) \n")
        print(' >  matlab.double(np_a.tolist()) =' +
            str( timeit.timeit('mat_a = matlab.double(np_a.real.tolist())',setup=setup_np2mat,  number=100)) + ' s')
        print(' >  ME4pyUtils.np2mlarray(np_a)=' + 
           str(timeit.timeit('mat_a = ME4pyUtils.np2mlarray(np_a)',setup=setup_np2mat,  number=100))+' s')
        
        
        
        
        print ("\nCompare matlab to numpy conversition strategy :")
        setup_tolist = (
            "import numpy as np\n"
            "import matlab\n"
            "import ME4pyUtils\n"
            "eng = matlab.engine.start_matlab()\n"
            "mrd = eng.rand(matlab.int64([1,10000]),nargout=1)\n")
        print (' > From np.array :' +
            str( timeit.timeit('nprd = np.array(mrd,dtype = float) ',setup=setup_tolist,  number=100)) +
            ' s')
    
        print (' > From np.asarray [use _data] :' +
            str( timeit.timeit('nprd = np.asarray(mrd._data,dtype = float) ',setup=setup_tolist,  number=100)) +
            ' s')
        
        print (' > From ME4pyUtils.mlarray2np [use _data buffer]:' +
            str(timeit.timeit('nprd = ME4pyUtils.mlarray2np(mrd) ',setup=setup_tolist,  number=100)) +
            ' s')
        setup_tolist_cpx = (
            "import numpy as np\n"
            "import matlab\n"
            "import ME4pyUtils\n"
            "eng = matlab.engine.start_matlab()\n"
            "mrd = eng.log(eng.linspace(-1.,1.,10000.))\n")
        print (' > From ME4pyUtils.mlarray2np [use _data buffer complex]:' +
            str(timeit.timeit('nprd = ME4pyUtils.mlarray2np(mrd) ',setup=setup_tolist_cpx,  number=100)) +
            ' s')