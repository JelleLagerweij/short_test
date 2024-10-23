# Postprocessing super short simulation

1. folder `1-simulation` holds the simulation.
1. folder `2-postprocessing` holds the standards postprocessing code.
1. folder `3-shift` translates the .xyz file appropriatly for testing.
1. folder `4-GATeWAY` is there for running the GATeWAY postprocessing.
1. folder `5-compare` creates the final comparison plots.

## Setting up the python environment
The python environment might have some difficulties setting up. Especially mpi4py can have install issues. In my case, it worked like this:

```ssh
pip -m venv .venv
source .venv/bin/activate
pip install numpy scipy h5py pandas ase tabulate matplotlib # standard packages for scientific computing, storing and writing results
```

Step 1 would require a clean vasp-6.4.0+ & hdf5 install and step 2 a conda environment with working mpi4py. This is not easy to set up with pip.

The steps 3-5 can be executed at once using the `test_all.sh` shell script. However the individual gateway executable has to be set correctly in folder `4-GATeWAY`.

```shell

bash test_all.sh
```
