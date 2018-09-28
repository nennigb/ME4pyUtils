`ME4PyUtils : MATLAB Engine API for Python utils`
============


## Aim

This set of pyton module and matlab function aim to ease some data type conversion between **matlab** and **numpy** python module.
MATLAB(R) Engine API for Python (ME4P) provides a package for Python to call MATLAB as a computational engine. 
MPE provides constructors to create arrays in Python (matlab.mlarray.double). This type is built on the array python module. 
This package propose **tricks** that can be used to 
	- convert such array into numpy
	- convert numpy array into matlab.mlarray.double
	- pass easily sparse matrix
	- avoid data copy
	
Note that python list can be cast by matlab.double() 

## Basic Usage

```
import ME4PyUtils
eng = matlab.engine.start_matlab()
import numpy as np
np_a=np.random.uniform(size=(1000))
...
```

## Installation
install ME4P (see matlab documentation)
Just download or clone the project. Eventually unzip the file and add the folder to matlab and python path.

## Test
python ME4Putils
