# runing GATeWAY automatically

The run_gateway.sh file executes GATeWAY on all copy_# subdirectories. Note that path to the exectutable should be changed to suit your setup, this can be done in line 3. The box size is an input to the bash file:

```ssh
bash run_gateway.sh <lbox>
bash run_gateway.sh 14.8997209999999995
``` 

To investigate without pbc, you can use very large box size. Note that this might give some dreadnaut warnings. I believe that this has to do with the : character. We will have some incomplete waters (and hydrogens) without pbc as some of our waters are origionally on the boundary itself, and thus get split in half now.