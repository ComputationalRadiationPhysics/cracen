# Cracen
===

Communication pipeline for Reconfigurable Accelerated Computing in scientific ENvironments

Cracen is a framework for distributed dataflow programming. It is primarily made for applications with
high bandwith datastreams with independent chunks.

## Download and Install


Cracen is a header only library. It can be downloaded from:

```bash
  git clone https://github.com/ComputationalRadiationPhysics/cracen.git
```

## Dependencies

- gcc/5.3.0 or clang 3.8
- graybat - https://github.com/ComputationalRadiationPhysics/graybat

Since graybat is a header only library too and does make use of c++14 features and boost::hana, it also
raises the dependencies for cracen too. If c++14 is not available on the target system, this can be circumvented,
by using the dev-c++11 branch of graybat. It is not recomended to do so, because the this branch is no longer
maintained.


## Build and Test

Cracen comes with a suite of examples and tests. The test suite is additionally dependent on boost 1.55.0 or higher.
The examples can be build by the following commands:

TODO:

The test suite can be build and executed this way:

TODO:
