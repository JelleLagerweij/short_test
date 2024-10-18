# Postprocessing vasp to get properties out
first the output of the vaspmodel is copied over to this folder. After which the Postprocessing is started in MPI mode.

```shell
cp ../1-simulation/vaspout.h5
mpiexec -np 4 python Prot_Hop_mpi.py
```
