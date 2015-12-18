DSP
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

## Build

```bash
  cd DSP/src
  make
```

## Run

```bash
  # in DSP/src
  make run
  # or
  ./DSP <FILENAME>
```

## Checking results

You can display the fit and orginal data in one diagram with the `Viewer`

```bash
   cd src/viewer/
   make clean
   make run
```
