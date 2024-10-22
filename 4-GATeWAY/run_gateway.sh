#!/bin/sh

if [ "$#" -ne 1 ] || [ "$1" = "-h" ]; then
    echo "Usage: $0 <lbox>"
    exit 1
fi

lbox=$1
gatewayexec=~//GATEwAYs/GATEwAY_2/bin/interface

for dir in copy_*; do
    if [ -d "$dir" ]; then
        echo "Processing $dir"
        cd $dir 
        $gatewayexec -w traj.xyz -x $lbox -y $lbox -z $lbox -d 7
        cd ..
    fi
done