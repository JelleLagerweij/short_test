# Postprocessing vasp to get properties out
first the output of the vaspmodel is copied over to this folder. After which the Postprocessing is started in MPI mode.
As the simulation only contains 10 timesteps, executing on single core is perfectly fine.

```shell
cp ../1-simulation/vaspout.h5
mpiexec -np 1 python Prot_Hop_mpi.py
```
