"""
This python code contains a simple comparrison class to compare two GATeWAY output folders.
"""

import os
import shutil
import tempfile
import subprocess
from typing import List
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from tabulate import tabulate

class Gateway:
    """
    Gateway class to filter GATeWAY output and only get the for Jelle relevant parts
    """
    def __init__(self, name : str):
        # read the gateway output files
        self.oh_list = self.grep_fast_filter(name + "_OH_stats.txt", r"\s1\s*$", cols=[0, 2])
        self.hb_list = self.grep_fast_filter(name + "_HBs_stats.txt", r"\bwp\b", cols=[0, 1, 2])
        self.check_timsteps()

    def grep_fast_filter(self, filename: str, pattern: str, cols: List[int]) -> np.ndarray:
        """ 
        Runs the grep of ripgrep commands to filter files efficiently.
        Utelyzes system memory to be fast
        """
        # Check if 'rg' is availab
        rg_available = shutil.which('rg') is not None

        # Construct the command based on availability
        if rg_available:
            command = ['rg', '-N', pattern, os.path.join(filename)]
        else:
            command = ['grep', pattern, os.path.join(filename)]

        try:
            result = subprocess.run(command, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            # Handle the error here
            print(f"Command failed with return code {e.returncode}")
            print(f"Error output: {e.stderr}")
            raise RuntimeError(f"Error running {' '.join(command)}") from e

        max_size = 1024 * 1024 * 1024  # 1GB in bytes
        with tempfile.SpooledTemporaryFile(mode='w+', max_size=max_size) as temp_file:
            # Write the result to the temporary file
            temp_file.write(result.stdout)
            temp_file.seek(0)  # Rewind the file to the beginning
            # Read the tempfile into a pandas DataFrame
            df = pd.read_csv(temp_file, sep=r'\s+', header=None, usecols=cols)
            # Convert DataFrame to NumPy array
            array = df.to_numpy()
        return array

    def check_timsteps(self):
        """
        Checks every timestep what the number of hbonds is and
        mustates the class with the updated result.
        """
        unique_timesteps, start_indices = np.unique(self.oh_list[:, 0], return_index=True)
        end_indices = np.r_[start_indices[1:], len(self.oh_list)]
        n_max = unique_timesteps.shape[0]

        self.n_hb = np.zeros(n_max, dtype=int)  # number of hydrogen bonds
        self.i_oh = np.zeros_like(self.n_hb) # index of OH-
        self.n_oh = np.zeros_like(self.n_hb) # number of OH-

        last_oh = None
        for t in unique_timesteps:
            oh_indices = self.oh_list[start_indices[t]:end_indices[t], 1]

            self.n_oh[t] = oh_indices.shape[0]
            # determine OH-
            if self.n_oh[t] == 1:
                # only 1 oh to deal with
                self.i_oh[t] = oh_indices[0]
            else:
                # if more then 1
                if (last_oh in oh_indices) or (self.n_oh[t] == 0):
                    # take same as last one
                    self.i_oh[t] = last_oh
                else:
                    # or take just the first of the list
                    self.i_oh[t] = oh_indices[0]

            last_oh = self.i_oh[t]
            # Create mask to count hydrogen bonds

            self.n_hb[t] = np.count_nonzero(self.hb_list[:, 0] == t)

def read_hbs(filename: str) -> np.ndarray:
    """
    Reads the output.h5 file and returns the hydrogen bonds.
    """
    input = h5py.File(filename, 'r')
    hbs = input["transient/current_OH_hbs"][()]
    hbs_com = hbs[:, 3, 0] + hbs[:, 3, 1]
    return hbs_com
    

if __name__ == "__main__":
    versions = []

    # Find the output.h5 file in 2-postprocessing
    jelle_post_loc = os.path.normpath("../2-postprocessing/output.h5")
    versions.append(read_hbs(jelle_post_loc))
    
    # find all subdirectories in ../4-GATeWAY and store their paths in a list
    gateway_dir = '../4-GATeWAY'
    subdirectories = [os.path.join(gateway_dir, d) for d in os.listdir(gateway_dir) if os.path.isdir(os.path.join(gateway_dir, d))]
    for subdirectory in subdirectories:
        print(f"Processing {subdirectory}")
        versions.append(Gateway(os.path.join(subdirectory, "traj/traj")).n_hb)
    
    # plot the results to compare
    plt.figure()
    plt.plot(versions[0], label='Jelle on origional')
    plt.plot(versions[1], label='Sana on origional')
    for i in range(2, len(versions)):
        plt.plot(versions[i], label=f'Sana on copy {i-1}')
    plt.savefig('compare.png')

    # Print results to compare
    header = ["timestep", "Jelle Orig", "Sana Orig"] + [f"Sana copy {i-1}" for i in range(2, len(versions))]
    data = []

    max_timesteps = max(len(v) for v in versions)
    for t in range(max_timesteps):
        row = [str(t)]
        for version in versions:
            if t < len(version):
                row.append(str(version[t]))
            else:
                row.append("N/A")
        data.append(row)

    print(tabulate(data, headers=header, tablefmt="grid"))
