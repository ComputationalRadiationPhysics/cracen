# Usage

The program needs at least one device with compute capability 3.5 and can be run with ./DSP <Filename>.

# Defining a model function

To define a new model function you need to implement the interface FitFunctor from "src/FitFunctor.hpp". The simplest
way is to derive a new class from the interface. 

- paramCount has to be set to the amount of parameters the fit needs. It must be set at compile time.
- The model must be in the form f(x,y,params)=0. the modelFunction implements f.
- the partial derivations of the modelFunction by each parameter must be implemented in the derivations function.



In FitFunctions.hpp are examples for a polynom, a polynom that only uses a window and the gauss function.

# Scaling

The program is able to use multiple GPUs for computation. The maximum number of devices that should be used can be changed in Constants.h.
To use pipelining for the kernel calls, the pipeline Depth can also be adjusted.
 
# File Input


# File Output
