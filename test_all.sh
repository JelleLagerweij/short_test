#!/bin/bash

# this script loop automatically through all steps 3-4-5. However, note that some path shall be changed before using it.

printf "First try with pbc enabled"
cd 3-shift
python shift_save.py 14.8997209999999995 traj.xyz 7
cd ../4-GATeWAY
bash run_gateway.sh 14.8997209999999995
cd ../5-compare
python compare.py
cd ..


# printf "Second try with pbc disabled"
# cd 3-shift
# python shift_save.py 500.0 traj.xyz 7
# cd ../4-GATeWAY
# bash run_gateway.sh 500.0
# cd ../5-compare
# python compare.py