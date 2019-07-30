`ME4PyUtils : MATLAB Engine API for Python utils`
============


## Aim

This set of python module and matlab functions aim to ease some data type conversion between **matlab** and **numpy** python module.
The MATLAB(R) Engine API for Python (ME4P) provides a package for Python to call MATLAB as a computational engine. 
MPE4P provides nativelly constructors to create arrays in Python (matlab.mlarray.X). This type is built on the **array** python module. 
Hower it not possible to create and pass numpy.ndarray. This module proposes some **tricks** that can be used to 
  - convert such mlarray into numpy.ndarray using frombuffer
  - convert numpy.ndarray into matlab.mlarray.X using prealloc and strides
  - pass easily sparse matrix as python dict
  - avoid to copy data (as long as possible)
	
>The conversion to numpy is fast, the bottleneck comes from the data transfert between python and matlab through the MATLAB Engine API for Python when the data become significant.

>Because numpy standard order is 'C' and mlarray use 'F', the strides may change.

## Basic Usage

```
import ME4PyUtils as ME
import numpy as np

# start matlab
eng = matlab.engine.start_matlab()
# conversion to matlab
np_a = np.random.uniform(size=(1000))
m_a = ME.np2mlarray(np_a)
# conversion to numpy
m_b = eng.rand(matlab.int64([1,1000]))
np_b = ME.mlarray2np(m_b)

```
if copy are needed, it can be done using `copy` module or `np.copy`.

Not tested on nd-array with nd>2...

## Installation
 1. install ME4P (see matlab documentation)
 2. Just download or clone the project. Eventually unzip the file and add the folder to matlab and python path.

## Test and compare speedup with other possibility
python ME4Putils

## Frequent issues with MATLAB Engine API

MATLAB Engine API for Python dislikes :
  - The calls of a matlab function with integer parameters instead of float. These parameters cast to long integer and will mislead matlab with int64
```
eng.sqrt(1)  # will crash
eng.sqrt(1.) # works
eng.mod(5,2) # works because integer are expected from matlab

```
  - returning more than one output without explicitly set `nargout`
```
s,id = eng.sort(np2mlarray(np.array([1.1,3.300,2.01])),nargout=2)  # works, id (index) is a double array...
s,id = eng.sort(np2mlarray(np.array([1.1,3.300,2.01])))  # crash
```


## Alternatives
The propose approach use MATLAB Engine API for Python, but other projects have been proposed, and use numpy nativelly.

[Transplant](https://github.com/bastibe/transplant) (socket communication)
[Oct2py](https://github.com/blink1073/oct2py) (mat-file communication, possible to speed up with RAM drive (tmpfs) by seting `temp_dir` properly)
[mlab](https://github.com/ewiger/mlab)
