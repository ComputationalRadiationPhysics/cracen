# DSP
===

A set of applications to fit modelfunctions und digital signals. The whole application is splitted in programs for data sources, compute nodes and data sinks.
Data sources can either be dump files from previous experiments or live measurement data. The data sinks write the fit data back into files in different files formats. 
At the moment dump files in json format are supported, root files will be added soon. The compute node is a fixed application, however the fit function can be 
customized. The applications can be split over a whole network in order to use dedicated hardware.

Fits model function to a set of waveforms.

## Download

```bash
  git clone https://github.com/ComputationalRadiationPhysics/DSP.git
```

## Dependencies

- gcc/5.1 or higher
- Cuda 7.5
- pthread
- boost 1.59 or higher
- gtk3
- gtk3mm
- staticTTY 
  - https://github.com/ComputationalRadiationPhysics/staticTTY

Optional:
- propriatary acqiris library for the scopeReader
- root 6 for root file output

## Build

```bash
  cd DSP/build
  cmake ..
  make
```
In addition to these executables, the zmq signaling server must be compiled. How this can be done is documented in the ZMQ repository.

## Run

```bash
  # in DSP
  ./zmq_signaling

  # in DSP/build
  # Read values from a file (source)
  ./FileReader
  
  # Read values from acqiris Scope (source)
  ./ScopeReader

  # Compute fit parameter (compute)
  ./Fitter
  
  # Write to file (json) (sink)
  ./FileWriter
  
  # Write to file (root) (sink) (not yet implemented)
  ./RootWriter
```
At least one source, compute and sink node must be started in parallel to establish a working programchain. Execution of all programs can start, after all
participating programs have been started. Hotpluging of any program is not implemented at the moment.
To get a list of all programoptions, each individual executable has an own help page. 

```bash
   ./<program> --help
```
