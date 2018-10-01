`ME4PyUtils : MATLAB Engine API for Python utils`
============


## Aim

This set of pyton module and matlab function aim to ease some data type conversion between **matlab** and **numpy** python module.
MATLAB(R) Engine API for Python (ME4P) provides a package for Python to call MATLAB as a computational engine. 
MPE4P provides nativelly constructors to create arrays in Python (matlab.mlarray.X). This type is built on the **array** python module. 
Hower it not possible to create and pass numpy.ndarray. This package propose some **tricks** that can be used to 
	- convert such mlarray into numpy.ndarray using frombuffer
	- convert numpy.ndarray into matlab.mlarray.X using prealloc and np.asarray
	- pass easily sparse matrix as python dict
	- avoid to copy data 
	

## Basic Usage

```
import ME4PyUtils
import numpy as np
eng = matlab.engine.start_matlab()

# conversion to matlab
np_a=np.random.uniform(size=(1000))
m_a=np2mlarray(np_a)
# conversion to numpy
m_b = eng.rand(matlab.int64([1,1000]))
np_b=mlarray2np(m_b)
...
```

## Installation
install ME4P (see matlab documentation)
Just download or clone the project. Eventually unzip the file and add the folder to matlab and python path.

## Test and compare speedup with other possibility
python ME4Putils
