# Here we will shift the xyz around for testing purposes

The python file will automatically overwrite the copy_# folders in the folder 4-GATeWAY. Copy 0 is the origional.

Run the program with the following method

```ssh
python shift_save.py <lbox> <input .xyz file> <num copies to generate>
python shift_save.py 14.8997209999999995 traj.xyz 5
```

Additonal clearification can be recieved using `python shift_save -h`. The correct box-size can be found in 1-simuation/POSCAR.
To avoid applying periodic boundary conditions, set the boxsize to very large.

```ssh
python shift_save.py 1000 traj.xyz 5
```