# runing GATeWAY automatically

The run_gateway.sh file executes GATeWAY on all copy_# subdirectories. Note that path to the exectutable should be changed to suit your setup, this can be done in line 3. The box size is an input to the bash file:

```ssh
bash run_gateway.sh <lbox>
bash run_gateway.sh 14.8997209999999995
``` 

We would expect that not only the hydration of the OH- is the same for all copies, but also all hydration bonding of the waters should be the same on average. The output of the bash script can also be used as a simple indication using the line Average of Hbonds per atom (except hydrogen atoms) = ...`

To investigate without pbc, you can use very large box size. Note that this might give some dreadnaut warnings. I believe that this has to do with the : character. We will have some incomplete waters (and hydrogens) without pbc as some of our waters are origionally on the boundary itself, and thus get split in half now.

```ssh
bash run_gateway.sh 1000
``` 