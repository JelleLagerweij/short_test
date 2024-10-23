#!/mnt/c/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Progress_meeting_23/New folder/origional/sana_test/.venv/bin/python
"""
This code 

"""
import argparse
import os
import shutil
from typing import List
from ase import io, Atoms
import numpy as np


def validate_filename(filename : str) -> str:
    """Check if filename ends with .xyz"""
    if not filename.endswith('.xyz'):
        raise argparse.ArgumentTypeError(f"file '{filename}' has to have a .xyz extension.")
    return filename


def read_file(inputfile : str) -> List[Atoms]:
    """Reads the .xyz file to extract types and coordinates"""
    try:
        frames = io.read(inputfile, format='xyz', index=':')
    except Exception as e:
        raise RuntimeError(f"Failed to read the file '{inputfile}'") from e
    return frames


def shift_positions(lbox: float, frames: List[Atoms]) -> List[Atoms]:
    """Shifts the box in such a way that atom n is the one shifted back
    to 0, 0, 0"""
    lmax = 20
    shift = (np.random.rand(1, 3)*lmax)  # random shift in the box between 0 and 20 A
    for frame in frames:
        frame.positions += shift  # occurs on all indexes of the array
        frame.positions = frame.positions % lbox  # wraps all back to fit in the pbc box
    return frames


def save_file(outputfile : str, frames : List[Atoms]):
    """Writes outputfile from by half the boxsize shifted inputfile"""
    try:
        io.write(outputfile, frames, format='xyz')
    except Exception as e:
        raise RuntimeError(f"Failed to save the file '{outputfile}'") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Processes an xyz file by shifting it and "+
                                     "saves the output file. It uses the boxsize, input" +
                                     "filename and output file name")

    # Add parser inputs
    parser.add_argument('lbox', type=float, help='boxsize in Angstrom.')
    parser.add_argument('inputfile', type=validate_filename, help='Input file name.')
    parser.add_argument('num_copies', type=int, help='Number of copied and shifted boxes to generate.')

    args = parser.parse_args()
    
    gateway_folder = '../4-GATeWAY/'
    
    # Clean the gateway folder by removing all subdirectories
    for item in os.listdir(gateway_folder):
        item_path = os.path.join(gateway_folder, item)
        if os.path.isdir(item_path):
            shutil.rmtree(item_path)
    
    configuration = read_file(args.inputfile)
    for copy in range(args.num_copies):
        if copy != 0:
            configuration = shift_positions(args.lbox, configuration)
        
        # Save the file, first create a subdirectory in the gateway folder
        subdirectory = os.path.join(gateway_folder, f"copy_{copy}")
        os.makedirs(subdirectory, exist_ok=True)
        
        # Then write the file to that subdirectory
        outputfile = os.path.join(subdirectory, os.path.basename(args.inputfile))
        save_file(outputfile, configuration)

    # Indicate successful completion of the processing
    print(f"Processing completed successfully, results availlable in 4-GATeWAY folder.")
