# runing GATeWAY automatically

The run_gateway.sh file executes GATeWAY on all copy_# subdirectories. Note that path to the exectutable should be changed to suit your setup, this can be done in line 3. The box size is an input to the bash file:

```ssh
bash run_gateway.sh <lbox>
bash run_gateway.sh 14.8997209999999995
``` 

We would expect that not only the hydration of the OH- is the same for all copies, but also all hydration bonding of the waters should be the same on average. The output of the bash script can also be used as a simple indication using the line Average of Hbonds per atom (except hydrogen atoms) = ...`

To investigate without pbc, you can use very large box size. Now, we do see that the hydrogen bonding number per atom stays the same.

```ssh
bash run_gateway.sh 500.0
``` 