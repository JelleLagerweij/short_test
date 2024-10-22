#!python
import numpy as np
import glob
import matplotlib.pyplot as plt
import freud
import scipy.constants as co
from scipy import integrate
from mpi4py import MPI
import time
import os
import h5py
from typing import Optional
from ase import io, Atoms

class Prot_Hop:
    """MPI supporting postprocesses class for NVT VASP simulations of aqueous KOH."""

    def __init__(self, 
                 folder: str, 
                 r_hb_old: float = 2.9, 
                 cheap: bool = True, 
                 verbose: bool = False, 
                 xyz_out: bool | str = False, 
                 serial_check: bool = False, 
                 center: bool = False):
        """
        Postprocesses class for NVT VASP simulations of aqueous KOH.

        It can compute the total system viscosity and the diffusion coefficient
        of the K+ and the OH-. The viscosity is computed using Green-Kubo from
        the pressure correlations and the diffusion coefficient using Einstein
        relations from the positions. The reactive OH- trajactory is computed
        using minimal and maximal bond lengths.

        Args:
            folder (string): path to hdf5 output file
            r_hb_old (float or int): maximum hydrogen bond distance
            cheap (bool): flag for cheapened calculation
            verbose (bool): flag for verbose output
            xyz_out (bool or string): output format for xyz, can be False, "xyz" or "pdb"
            serial_check (bool): flag for serial check
            center (bool): flag for centering
        """
        if not isinstance(folder, str):
            raise ValueError("folder must be a string")
        if not isinstance(r_hb_old, (float, int)):
            raise ValueError("r_hb_old must be a float or int")
        if not isinstance(cheap, bool):
            raise ValueError("cheap must be a boolean")
        if not isinstance(verbose, bool):
            raise ValueError("verbose must be a boolean")
        if not (xyz_out is False or xyz_out in ["xyz", "pdb"]):
            raise ValueError('xyz_out must be False, "xyz", or "pdb"')
        if not isinstance(serial_check, bool):
            raise ValueError("serial_check must be a boolean")
        if not isinstance(center, bool):
            raise ValueError("center must be a boolean")

        self.tstart = time.time()
        self.species = ['H', 'O', 'K']

        self.comm = MPI.COMM_WORLD

        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.error_flag = 0
        self.cheap = cheap
        self.folder = os.path.normpath(folder)
        self.verbose = verbose
        self.xyz_out = xyz_out
        self.serial_check = serial_check
        self.r_max_count = r_hb_old
        self.center = center  # has to be a boolean

        # Normal Startup Behaviour
        self.setting_properties_all()  # all cores
        
        # Real postprocessing the trajectory
        self.loop_timesteps_all()  # identify OH-
        self.stitching_together_all()
        self.loop_timesteps_next_OH_all()  # loop back and also find hbond of water becomming OH-

        # afterwards on single core additional properties
        if self.rank == 0:
            self.compute_MSD_pos()
            self.compute_MSD_pres()
        
        # Saving everything
        self.save_results_all()

        if self.rank == 0:
            if self.verbose is True:
                print('time to completion',  time.time() - self.tstart)

            if serial_check is True:
                self.test_combining_main()
        exit()
        
    def setting_properties_main(self):
        """
        Load the output file into dedicated arrays

        This function should only be executed on the main core

        Args:
            folder (string): path to hdf5 output file
        """
        try:
            self.folder = os.path.normpath(self.folder)
            if (os.name == 'posix') and ('WSL_DISTRO_NAME' in os.environ):
                # Convert Windows path to WSL path
                self.folder = os.path.normpath(os.getcwd(), self.folder)
            elif (os.name == 'posix') and ('delftblue' == os.environ['CMD_WLM_CLUSTER_NAME']):
                # Convert to fully relative paths
                self.folder = os.path.join(os.getcwd(), self.folder)
            elif (os.name == 'nt') and ('vlagerweij == os.getlogin()'):
                # use standard windows file path
                self.folder = self.folder
        except:
            print('no automatic filepath conversion available use relative path')
            self.folder = os.path.normpath(os.getcwd() + "/" + self.folder)

        # finds all vaspout*.h5 files in folder and orders them alphabetically
        subsimulations = sorted(glob.glob(os.path.normpath(self.folder + "/vaspout*.h5")))
        if len(subsimulations) == 0:
            print(f'FATAL ERROR, no such vaspout*.h5 file availlable at {self.folder}')
            self.comm.Abort()

        # loop over subsimulations
        for i in range(len(subsimulations)):
            self.df = h5py.File(subsimulations[i])
            if i == 0:
                try:
                    skips =  self.df['/input/incar/ML_OUTBLOCK'][()]
                except:
                    skips = 1
                self.pos_all = self.df['intermediate/ion_dynamics/position_ions'][()]
                self.force = self.df['intermediate/ion_dynamics/forces'][()]
                self.stress = self.df['intermediate/ion_dynamics/stress'][()]
                self.energy = self.df['intermediate/ion_dynamics/energies'][()]
                
                
                # Load initial structure properties and arrays out of first simulation
                self.N = self.df['results/positions/number_ion_types'][()]
                self.L = self.df['results/positions/lattice_vectors'][()][0, 0]  # Boxsize
                self.pos_all *= self.L
                
                # load initial force properties and arrays out of first simulation                
                # Read custom data out of HDF5 file (I do not know how to get it out of py4vasp)
                self.dt = self.df['/input/incar/POTIM'][()]
                self.dt *= skips
                self.T_set = float(self.df['/input/incar/TEBEG'][()])

            else:
                pos = self.df['intermediate/ion_dynamics/position_ions'][()]*self.L
                force = self.df['intermediate/ion_dynamics/forces'][()]
                stress = self.df['intermediate/ion_dynamics/stress'][()]
                energy = self.df['intermediate/ion_dynamics/energies'][()]
                               
                # load new positions, but apply pbc unwrapping (# ARGHGH VASP)
                dis_data = self.pos_all[-1, :, :] - pos[0, :, :]
                dis_real = ((self.pos_all[-1, :, :] - pos[0, :, :] + self.L/2) % self.L - self.L/2)
                pos -= (dis_real - dis_data)

                # now matching together
                self.pos_all = np.concatenate((self.pos_all, pos), axis=0)
                
                # load new forces, stresses and energies and add to old array
                self.force = np.concatenate((self.force, force), axis=0)
                self.stress = np.concatenate((self.stress, stress), axis=0)
                self.energy = np.concatenate((self.energy, energy), axis=0)
            self.df.close()

        self.t = np.arange(self.pos_all.shape[0])*self.dt
        
        # After putting multiple simulations together
        self.pos_all_split = np.array_split(self.pos_all, self.size, axis=0)
        self.force_split = np.array_split(self.force, self.size, axis=0)
        self.t_split = np.array_split(self.t, self.size)
        self.T_split = np.array_split(self.energy[:, 3], self.size, axis=0)

        # calculate chunk sizes for communication
        self.chunks = [int]*self.size
        self.steps_split = [None]*self.size
        sta = 0
        for i in range(self.size):
            self.chunks[i] = len(self.pos_all_split[i][:, 0])
            sto = sta + len(self.pos_all_split[i][:, 0])
            self.steps_split[i] = np.arange(start=sta, stop=sto)
            sta = sto + 1

        for i in range(1, self.size):  # only send to specific cores in chunks
            self.comm.Send([self.pos_all_split[i], MPI.DOUBLE], dest=i)
            self.comm.Send([self.force_split[i], MPI.DOUBLE], dest=i)

        self.pos = self.pos_all_split[0]
        self.force = self.force_split[0]

    def setting_properties_all(self):
        """
        Initializes system for all cores

        This function initializes the variables that need to be available to all cores
        """
        if self.rank == 0:
            self.setting_properties_main()  # Initializes on main cores
        else:
            self.chunks = [None]*self.size
            self.L = float
            self.N = np.empty(3, dtype=int)
            self.dt = float
            self.T_set = float
            self.center = bool
            self.verbose = bool
        
        self.chunks = self.comm.bcast(self.chunks, root=0)
        self.L = self.comm.bcast(self.L, root=0)
        self.N = self.comm.bcast(self.N, root=0)
        self.dt = self.comm.bcast(self.dt, root=0)
        self.T_set =  self.comm.bcast(self.T_set, root=0)
        self.center = self.comm.bcast(self.center, root=0)
        self.verbose = self.comm.bcast(self.verbose, root=0)
        
        # print('self.dt is', self.dt, 'self.T_set is', self.T_set, 'rank is', self.rank)
        # Asses the number of Hydrogens
        self.N_tot = np.sum(self.N)
        self.N_H = self.N[self.species.index('H')]
        self.N_O = self.N[self.species.index('O')]
        self.N_K = self.N[self.species.index('K')]

        self.H_i = np.arange(self.N_H)
        self.O_i  = np.arange(self.N_H, self.N_H + self.N_O)
        self.K_i = np.arange(self.N_H + self.N_O, self.N_H + self.N_O + self.N_K)

        # Calculate and initiate the OH- tracking
        self.N_OH = self.N_K  # intended number of OH- from input file
        self.N_H2O = self.N_O - self.N_OH
        
        if self.rank != 0:
            self.pos = np.empty((self.chunks[self.rank], self.N_tot, 3))  # create empty dumy on all cores
            self.force = np.empty((self.chunks[self.rank], self.N_tot, 3))  # create empty dumy on all cores
            self.steps_split = [np.empty(self.chunks[i]) for i in range(self.size)]
            self.t_split = [np.empty(self.chunks[i]) for i in range(self.size)]
            self.T_split = [np.empty(self.chunks[i]) for i in range(self.size)]

        # Import the correct split position arrays
        if self.rank != 0:
            self.comm.Recv([self.pos, MPI.DOUBLE], source=0)
            self.comm.Recv([self.force, MPI.DOUBLE], source=0)
        self.steps = self.comm.scatter(self.steps_split, root=0)
        self.t = self.comm.scatter(self.t_split, root=0)
        self.T_trans = self.comm.scatter(self.T_split, root=0)
        
        self.n_max = len(self.pos[:, 0, 0])  # number of timestep on core
        self.n_max_all = self.comm.allreduce(self.n_max, op=MPI.SUM)
        
        # communicate if cheapened calculation (less interaction types included)
        if self.rank == 0:
            self.cheap = self.cheap
        else:
            self.cheap = False
        self.cheap = self.comm.bcast(self.cheap, root=0)
                    
        if self.rank == 0 and self.verbose is True:
            print('Time after communication', time.time() - self.tstart, flush=True)

        if self.verbose:
            print('Communication done on rank', self.rank, 'size', self.pos.shape, flush=True)

    def recognize_molecules_all(self, n: int) -> None:
        """
        Find the index of the Oxygen beloging to the OH- or H2O.

        This function searches for the index of the Oxygen belonging to the OH-
        particles. It automatically creates a neighbor list as well as an array
        which holds the unwraped location of the real OH particle.

        Args:
            n (integer): timestep of the assesment of the hydoxide recognition
        """
        if n == 0:  # Do a startup procedure
            self.OH = np.zeros((self.n_max, self.N_OH, 3))  # prepair the OH- position storage array
            self.OH_i = np.zeros((self.n_max, self.N_OH), dtype='int')  # prepair for the OH- O index storage array
            self.n_OH = np.zeros(self.n_max, dtype='int')  # prepair for real number of OH-
            self.OH_shift = np.zeros((self.N_OH, 3), dtype='int')  # prepair history shift list for pbc crossings

            self.H2O = np.zeros((self.n_max, self.N_H2O, 3))  # prepair the H2O position storage array
            self.H2O_i = np.zeros((self.n_max, self.N_H2O), dtype='int')  # prepair for the H2O O index storage array
            self.n_H2O = np.zeros(self.n_max, dtype='int')  # prepair for real number of H2O
            self.H2O_shift = np.zeros((self.N_O-self.N_OH, 3), dtype='int')  # prepair history shift list for pbc crossings
            
            self.n_H3O = np.zeros(self.n_max, dtype='int')  # prepair for real number of H3O+ (should be 0)
        self.O_per_H = np.argmin(self.d_HO.reshape((self.N_H, self.N_O)), axis=1)
        counter_H_per_O = np.bincount(self.O_per_H)  # Get number of H per oxygen in molecule
        
        # Identify and count all real OH-, H2O and H3O+
        OH_i = np.where(counter_H_per_O == 1)[0]
        self.n_OH[n] = OH_i.shape[0]
        H2O_i = np.where(counter_H_per_O == 2)[0]
        self.n_H2O[n] = H2O_i.shape[0]
        H3O_i = np.where(counter_H_per_O == 3)[0]
        self.n_H3O[n] = H3O_i.shape[0]
        
        # Now start matching the correct molecules together
        if n == 0:
            # No matching needed, just filling the arrays correctly
            self.OH_i[n, :] = OH_i
            self.OH[n, :, :] = self.pos_O[n, self.OH_i[n, :], :]
            
            self.H2O_i[n, :] = H2O_i
            self.H2O[n, :, :] = self.pos_O[n, self.H2O_i[n, :], :]
            
            self.OH_i_s = OH_i  # set this in the first timestep
        else:
            if self.N_OH != self.n_OH[n] or self.n_H3O[n] > 0:  # Exemption for H3O+ cases
                print(f"Strange behaviour found, H3O+ created. Rank", self.rank, 'timestep ', self.t[n])
                print(f"N_H3O+", self.n_H3O[n], flush=True)
                
                # issues = np.setdiff1d(OH_i, self.OH_i[n])
                # for i in issues:
                self.OH_i[n, :] = self.OH_i[n-1, :]  # use origional sorting by using last version as nothing changed
                self.OH[n, :, :] = self.pos_O[n, self.OH_i[n, :], :] + self.L*self.OH_shift
                
                self.H2O_i[n, :] = self.H2O_i[n-1, :]  # use origional sorting by using last version as nothing changed
                self.H2O[n, :, :] = self.pos_O[n, self.H2O_i[n, :], :] + self.L*self.H2O_shift

            elif (OH_i == self.OH_i_s).all():  # No reaction occured only check PBC
                self.OH_i[n, :] = self.OH_i[n-1, :]  # use origional sorting by using last version as nothing changed
                self.OH[n, :, :] = self.pos_O[n, self.OH_i[n, :], :] + self.L*self.OH_shift
                
                self.H2O_i[n, :] = self.H2O_i[n-1, :]  # use origional sorting by using last version as nothing changed
                self.H2O[n, :, :] = self.pos_O[n, self.H2O_i[n, :], :] + self.L*self.H2O_shift

            else:  # Reaction cases
                # find which OH- belongs to which OH-. This is difficult because of sorting differences.
                diff_new = np.setdiff1d(OH_i, self.OH_i[n-1, :], assume_unique=True)
                diff_old = np.setdiff1d(self.OH_i[n-1, :], OH_i, assume_unique=True)
                
                self.OH_i[n, :] = self.OH_i[n-1, :]
                self.H2O_i[n, :] = self.H2O_i[n-1, :]
                for i in range(len(diff_old)):
                    ## HYDROXIDE PART
                    # Check every closest old version to every unmatching new one and replace the correct OH_i index
                    r_OO = (self.pos_O[n, diff_new, :] - self.pos_O[n, diff_old[i], :] + self.L/2) % self.L - self.L/2               
                    d2 = np.sum(r_OO**2, axis=1)
                    i_n = np.argmin(d2)  # find new OH- index

                    if d2[i_n] > 9:  # excape when jump too large
                        print("Strange behaviour found, reaction jump too far, d = ", np.sqrt(d2[i_n]), 'Angstrom',
                              "CPU rank=", self.rank, 'timestep', self.steps[n], flush=True)

                    idx_OH_i = np.where(self.OH_i[n-1, :] == diff_old[i]) # get index in OH_i storage list
                    self.OH_i[n, idx_OH_i] = diff_new[i_n]  # update OH_i storage list

                    # Adjust for different PBC shifts in reaction
                    dis = (self.pos_O[n, self.OH_i[n, idx_OH_i], :] - self.pos_O[n-1, self.OH_i[n-1, idx_OH_i], :] + self.L/2) % self.L - self.L/2   # displacement vs old location
                    real_loc = self.pos_O[n-1, self.OH_i[n-1, idx_OH_i], :] + dis
                    self.OH_shift[idx_OH_i, :] += np.round((real_loc - self.pos_O[n, self.OH_i[n, idx_OH_i], :])/self.L).astype(int)  # Correct shifting update for index
                
                    ## WATER PART
                    # We already know the exchange of atomic indexes, now find the array location in the water list (logic is reversed)
                    idx_H2O_i = np.where(self.H2O_i[n-1, :] == diff_new[i_n])  # New OH- was old H2O
                    self.H2O_i[n, idx_H2O_i] = diff_old[i]  # Therefore new H2O is old OH-
                    
                    # Adjust for different PBC shifts in reaction
                    dis = (self.pos_O[n, self.H2O_i[n, idx_H2O_i], :] - self.pos_O[n-1, self.H2O_i[n-1, idx_H2O_i], :] + self.L/2) % self.L - self.L/2   # displacement vs old location
                    real_loc =self.pos_O[n-1, self.H2O_i[n-1, idx_H2O_i], :] + dis
                    self.H2O_shift[idx_H2O_i, :] += np.round((real_loc - self.pos_O[n, self.H2O_i[n, idx_H2O_i], :])/self.L).astype(int)  # Correct shifting update for index
                    
                
                # Update all the positions
                self.OH[n, :, :] = self.pos_O[n, self.OH_i[n, :], :] + self.L*self.OH_shift
                self.H2O[n, :, :] = self.pos_O[n, self.H2O_i[n, :], :]+ self.L*self.H2O_shift
                self.OH_i_s = OH_i  # always sort after reaction or initiation to have a cheap check lateron.

    def hydrogen_bonds(self, idx_Oc: int, n: int) -> np.ndarray:
        """
        Calculate hydrogen bonds for a given oxygen index and time step.

        Args:
            idx_Oc (int): Index of the oxygen atom.
            n (int): Time step.

        Returns:
            np.ndarray: Array of hydrogen bond information.
        """
        hbs = np.zeros((5, 2), dtype=int)
        
        # Get all oxygen-oxygen interactions close to idx_Oc
        Oc_Other = ((np.isin(self.idx_OO[0], idx_Oc)) | (np.isin(self.idx_OO[1], idx_Oc)))  # all other Oxygen Oxygen interactions
        O_close = (Oc_Other & (self.d_OO < 3.5))
        idx_O_close = np.concatenate([self.idx_OO[0][O_close & (self.idx_OO[1] == idx_Oc)],
                                      self.idx_OO[1][O_close & (self.idx_OO[0] == idx_Oc)]], axis=0)
        
        # List relevant oxygen-oxygen interactions
        r_O_O_close = self.r_OO[O_close]  # N by 3, the vector
        d_O_O_close = self.d_OO[O_close]  # N by 1, the distance
        
        # get bond bevtor within nextOH (intern)
        idx_H_int = np.where(self.O_per_H == idx_Oc)  # it shall result in 2 hydrogens as this is still water
        b_don_vec = self.r_HO[np.isin(self.idx_HO[0], idx_H_int) & np.isin(self.idx_HO[1], idx_Oc)]  # 2 by 3 the vector
        b_don_len = self.d_HO[np.isin(self.idx_HO[0], idx_H_int) & np.isin(self.idx_HO[1], idx_Oc)]  # 2 by 1 the length
        b_don_norm = b_don_vec/b_don_len[:, np.newaxis]
        
        hbs[0, 0] = np.sum(d_O_O_close < self.r_max_count)  # most simple oxygen distance counter
        for i, idx_O in enumerate(idx_O_close):
            # Collect details of relevant vectors
            # Step 1, slice correct O-O distance vector
            flip_sign = np.where(idx_O > idx_Oc, -1, 1)  # flip sign if the index is smaller than the next OH
            R_vec = r_O_O_close[i]  # 1 by 3, the vector
            R_len = d_O_O_close[i]  # 1 by 1, the length
            R_norm = R_vec*flip_sign/R_len  # 1 by 3, the normalized vector
            
            # Step 2, find H in molecule idx_O_close and bond vectors (extern)
            idx_H_ext = np.where(self.O_per_H == idx_O)  # This shall be 2 values as there are 2 Hydrogen per oxygen
            b_acc_vec = self.r_HO[np.isin(self.idx_HO[0], idx_H_ext) & np.isin(self.idx_HO[1], idx_O)]  # 2 (or 1) by 3, the vector
            b_acc_len = self.d_HO[np.isin(self.idx_HO[0], idx_H_ext) & np.isin(self.idx_HO[1], idx_O)]  # 2 (or 1) by 1, the distance
            b_acc_norm = b_acc_vec/b_acc_len[:, np.newaxis]  # 2 (or 1) by 3, the normalized vector
            
            # Step 3, find vector Hydrogen-Oxygen r in Kumar; J. R. Schmidt; J. L. Skinner 2007
            r_don_vec = self.r_HO[np.isin(self.idx_HO[0], idx_H_int) & np.isin(self.idx_HO[1], idx_O)]  # 2 by 3, the vector
            r_don_len = self.d_HO[np.isin(self.idx_HO[0], idx_H_int) & np.isin(self.idx_HO[1], idx_O)]  # 2 by 1, the distance
            r_don_norm = r_don_vec/r_don_len[:, np.newaxis]  # 2 by 3, the normalized vector
            
            # Step 4, find vector Hydrogen-Oxygen r in Kumar; J. R. Schmidt; J. L. Skinner 2007
            # We should find 2 vectors
            r_acc_vec = self.r_HO[np.isin(self.idx_HO[0], idx_H_ext) & np.isin(self.idx_HO[1], idx_Oc)]  # 2 (or 1) by 3, the vector
            r_acc_len = self.d_HO[np.isin(self.idx_HO[0], idx_H_ext) & np.isin(self.idx_HO[1], idx_Oc)]  # 2 (or 1) by 1, the distance
            r_acc_norm = r_acc_vec/r_acc_len[:, np.newaxis]  # 2 (or 1) by 3, the normalized vector
            
            # Part 1: Donor testing
            beta_don = np.arccos(b_don_norm[:, 0]*R_norm[0] +
                                 b_don_norm[:, 1]*R_norm[1] +
                                 b_don_norm[:, 2]*R_norm[2]) 
            theta_don = np.arccos(b_don_norm[:, 0]*r_don_norm[:, 0] +
                                  b_don_norm[:, 1]*r_don_norm[:, 1] +
                                  b_don_norm[:, 2]*r_don_norm[:, 2])
            
            # Cheat python: True will be treated as 1, False as 0, so you can use this in equations.
            hbs[1, 0] += np.sum(beta_don < 0.5235987755982988)  # Chandler and Luzar (1996) 30 degree beta limit
            hbs[2, 0] += np.sum(R_len + 1.4444347940051674*beta_don**2 < 3.3)  # Wernet Nordlund Bermann (2004)
            hbs[3, 0] += np.sum((theta_don > 2.443460952792061) & (R_len < 3.2))  # The way Sana does it (Giulia Galli & Francois Gygi 2000) 140 degree limit
            hbs[4, 0] += np.sum((r_don_len < 2.27) & (theta_don > 2.443460952792061))  # Kuo and Mundy (2004) !!
            
            # Part 2: Acceptor testing
            beta_acc = np.arccos(-(b_acc_norm[:, 0]*R_norm[0]
                                 + b_acc_norm[:, 1]*R_norm[1]
                                 + b_acc_norm[:, 2]*R_norm[2])) 
            theta_acc = np.arccos(b_acc_norm[:, 0]*r_acc_norm[:, 0]
                                  + b_acc_norm[:, 1]*r_acc_norm[:, 1]
                                  + b_acc_norm[:, 2]*r_acc_norm[:, 2])
            
            # Cheat python: True will be treated as 1, False as 0, so you can use this in equations.
            hbs[1, 1] += np.sum(beta_acc < 0.5235987755982988)  # Chandler and Luzar (1996) 30 degree beta limit
            hbs[2, 1] += np.sum(R_len + 1.4444347940051674*beta_acc**2 < 3.3)  # Wernet Nordlund Bermann (2004)
            hbs[3, 1] += np.sum((theta_acc > 2.443460952792061) & (R_len < 3.2))  # The way Sana does it (Giulia Galli & Francois Gygi 2000) 140 degree limit
            hbs[4, 1] += np.sum((r_acc_len < 2.27) & (theta_acc > 2.443460952792061))  # Kuo and Mundy (2004) !! TODO maybe adjust minimal length to 1.1A
        
        return hbs

    def loop_timesteps_all(self, n_samples: int = 10) -> None:
        """
        Loop over all timesteps and track time-dependent properties.

        This function calls the molecule recognition function and the RDF functions when needed.

        Args:
            n_samples (int, optional): Time between sampling RDFs. Defaults to 10.
        """
        # split the arrays up to per species description
        self.pos_H = self.pos[:, self.H_i, :]
        self.pos_O = self.pos[:, self.O_i, :]
        self.pos_K = self.pos[:, self.K_i, :]

        self.force_H = self.force[:, self.H_i, :]
        self.force_O = self.force[:, self.O_i, :]
        self.force_K = self.force[:, self.K_i, :]

        # Create per species-species interactions indexing arrays
        self.idx_KK = np.triu_indices(self.N_K, k=1)
        self.idx_KO = np.mgrid[0:self.N_K, 0:self.N_O].reshape(2, self.N_K*self.N_O)
        self.idx_HO = np.mgrid[0:self.N_H, 0:self.N_O].reshape(2, self.N_H*self.N_O)
        self.idx_OO = np.triu_indices(self.N_O, k=1)
        
        self.current_OH_hbs = np.zeros((self.n_max, 5, 2), dtype=int)
        
        if self.cheap == False:  # Exclude yes usefull interactions especially the H-H interactions take long
            self.idx_HH = np.triu_indices(self.N_H, k=1)
            self.idx_HK = np.mgrid[0:self.N_H, 0:self.N_K].reshape(2, self.N_H*self.N_K)

        for n in range(self.n_max):  # Loop over all timesteps
            if (n % 10000 == 0) and (self.verbose is True) and (n > 0):
                print("time is:", self.t[n], " in pass 1, rank is:", self.rank, flush=True)
            # Calculate only OH distances for OH- recognition
            self.r_HO = (self.pos_O[n, self.idx_HO[1], :] - self.pos_H[n, self.idx_HO[0], :] + self.L/2) % self.L - self.L/2
            self.d_HO = np.sqrt(np.sum(self.r_HO**2, axis=1))
            
            self.recognize_molecules_all(n)
        
            # Oxygen-Oxygen
            # already compted for Hydrogen bonding numbers
            self.r_OO = (self.pos_O[n, self.idx_OO[1], :] - self.pos_O[n, self.idx_OO[0], :] + self.L/2) % self.L - self.L/2
            self.d_OO = np.sqrt(np.sum(self.r_OO**2, axis=1))
            self.F_OO = self.force_O[n, self.idx_OO[1], :] - self.force_O[n, self.idx_OO[0], :]
            OHH2O = (((np.isin(self.idx_OO[0], self.H2O_i[n])) & (np.isin(self.idx_OO[1], self.OH_i[n])))) | (((np.isin(self.idx_OO[0], self.OH_i[n])) & (np.isin(self.idx_OO[1], self.H2O_i[n]))))  # ((a and b) or (b and a)) conditional
            # Create index array for Oxygen differencing
            H2OH2O = ((np.isin(self.idx_OO[0], self.H2O_i[n])) & (np.isin(self.idx_OO[1], self.H2O_i[n])))

            self.d_OHH2O = self.d_OO[OHH2O]
            
            self.r_H2OH2O = self.r_OO[H2OH2O]
            self.d_H2OH2O = self.d_OO[H2OH2O]
            self.F_H2OH2O = self.F_OO[H2OH2O]

            self.r_OHH2O = self.r_OO[OHH2O]
            self.F_OHH2O = self.F_OO[OHH2O]
            
            self.current_OH_hbs[n] = self.hydrogen_bonds(self.OH_i[n], n)
            
            if n % n_samples == 0:
                KOH = np.isin(self.idx_KO[1], self.OH_i[n])
                KH2O = np.isin(self.idx_KO[1], self.H2O_i[n])
                
                r_KO = (self.pos_O[n, self.idx_KO[1], :] - self.pos_K[n, self.idx_KO[0], :] + self.L/2) % self.L - self.L/2
                d_KO = np.sqrt(np.sum(r_KO**2, axis=1))
                F_KO = self.force_O[n, self.idx_KO[1], :] - self.force_K[n, self.idx_KO[0], :]
                
                KOH = np.isin(self.idx_KO[1], self.OH_i[n])
                KH2O = np.isin(self.idx_KO[1], self.H2O_i[n])
                self.r_KOH = r_KO[KOH]  # selecting only OH from array
                self.d_KOH = d_KO[KOH]  
                self.F_KOH = F_KO[KOH]
                
                self.r_KH2O = r_KO[KH2O]  # selecting only H2O from array
                self.d_KH2O = d_KO[KH2O]
                self.F_KH2O = F_KO[KH2O]

                if self.N_K > 1:  # only ion-ion self interactions if more than 1 is there
                    OHOH = ((np.isin(self.idx_OO[0], self.OH_i[n])) & (np.isin(self.idx_OO[1], self.OH_i[n])))
                    self.r_OHOH = self.r_OO[OHOH]
                    self.d_OHOH = self.d_OO[OHOH]
                    self.F_OHOH = self.F_OO[OHOH]
                    
                    self.r_KK = (self.pos_K[n, self.idx_KK[1], :] - self.pos_K[n, self.idx_KK[0], :] + self.L/2) % self.L - self.L/2
                    self.d_KK = np.sqrt(np.sum(self.r_KK**2, axis=1))
                    self.F_KK = self.force_K[n, self.idx_KK[1], :] -  self.force_K[n, self.idx_KK[0], :]

                if self.cheap == False:  # Exclude yes usefull interactions especially the H-H interactions take long
                    HOH = np.isin(self.idx_HO[1], self.OH_i[n])
                    HH2O = np.isin(self.idx_HO[1], self.H2O_i[n])
                    
                    F_HO = self.force_O[n, self.idx_HO[1], :] -  self.force_H[n, self.idx_HO[0], :]
                    
                    self.r_HOH = self.r_HO[HOH]
                    self.d_HOH = self.d_HO[HOH]
                    self.F_HOH = F_HO[HOH]
                    
                    self.r_HH2O = self.r_HO[HH2O]
                    self.d_HH2O = self.d_HO[HH2O]
                    self.F_HH2O = F_HO[HH2O]

                    self.r_HH = (self.pos_H[n, self.idx_HH[1], :] - self.pos_H[n, self.idx_HH[0], :] + self.L/2) % self.L - self.L/2
                    self.d_HH = np.sqrt(np.sum(self.r_HH**2, axis=1))
                    self.F_HH = self.force_H[n, self.idx_HH[1], :] -  self.force_H[n, self.idx_HH[0], :]
                    
                    # r_HK = (self.pos_K[n, self.idx_HK[1], :] - self.pos_H[n, self.idx_HK[0], :] + self.L/2) % self.L - self.L/2
                    # self.d_HK = np.sqrt(np.sum(r_HK**2, axis=1))
                    
                    self.d_KO_all = d_KO
                    self.d_OO_all = self.d_OO

                # Now compute RDF results
                self.rdf_compute_all(n)
                self.rdf_force_compute_all(n)            


        if self.rank == 0 and self.verbose is True:
            print('Time calculating distances', time.time() - self.tstart)

    def rdf_compute_all(self, n: int, nb: int = 32, r_max: Optional[float] = None) -> None:
        """
        Compute the radial distribution functions (RDF) for various pairs of particles.
        Parameters:
        -----------
        n : int
            The current step or iteration number. If n == 0, initializes the RDF arrays.
        nb : int, optional
            The number of bins to use for the histogram. Default is 32.
        r_max : float, optional
            The maximum distance for the RDF calculation. If None, it is set to half the box length.
        Returns:
        --------
        None
        """
        # RDF startup scheme
        if n == 0:
            # set standard maximum rdf value
            if r_max == None:
                r_max = self.L/2  # np.sqrt(3*self.L**2/4)  # set to default half box diagonal distance

            # set basic properties
            self.r = np.histogram(self.d_H2OH2O, bins=nb + 1, range=(0, r_max))[1] # array with outer edges
            
            # Standard rdf pairs
            self.rdf_H2OH2O = np.zeros(self.r.size - 1, dtype=float)
            self.rdf_OHH2O = np.zeros_like(self.rdf_H2OH2O)
            self.rdf_KOH = np.zeros_like(self.rdf_H2OH2O)
            self.rdf_KH2O = np.zeros_like(self.rdf_H2OH2O)
            if self.N_K > 1:  # only ion-ion self interactions if more than 1 is there
                self.rdf_OHOH = np.zeros_like(self.rdf_H2OH2O)
                self.rdf_KK = np.zeros_like(self.rdf_H2OH2O)
            if self.cheap is False: # Also execute Hydrogen interaction distances (long lists)
                self.rdf_HOH = np.zeros_like(self.rdf_H2OH2O)
                self.rdf_HH2O = np.zeros_like(self.rdf_H2OH2O)
                # self.rdf_HK = np.zeros_like(self.rdf_H2OH2O)
                self.rdf_HH = np.zeros_like(self.rdf_H2OH2O)
                self.rdf_KO_all = np.zeros_like(self.rdf_H2OH2O)
                self.rdf_OO_all = np.zeros_like(self.rdf_H2OH2O)

        # Now calculate all rdf's (without rescaling them, will be done later)
        self.rdf_H2OH2O += np.histogram(self.d_H2OH2O, bins=self.r)[0]
        self.rdf_OHH2O += np.histogram(self.d_OHH2O, bins=self.r)[0]
        self.rdf_KOH += np.histogram(self.d_KOH, bins=self.r)[0]
        self.rdf_KH2O += np.histogram(self.d_KH2O, bins=self.r)[0]

        if self.N_K > 1:  # only ion-ion self interactions if more than 1 is there
            self.rdf_OHOH += np.histogram(self.d_OHOH, bins=self.r)[0]
            self.rdf_KK += np.histogram(self.d_KK, bins=self.r)[0]
    
        if self.cheap is False: # Also execute Hydrogen interaction distances (long lists)
                self.rdf_HOH += np.histogram(self.d_HOH, bins=self.r)[0]
                self.rdf_HH2O += np.histogram(self.d_HH2O, bins=self.r)[0]
                # self.rdf_HK += np.histogram(self.d_HK, bins=self.r)[0]
                self.rdf_HH += np.histogram(self.d_HH, bins=self.r)[0]
                self.rdf_KO_all += np.histogram(self.d_KO_all, bins=self.r)[0]
                self.rdf_OO_all += np.histogram(self.d_OO_all, bins=self.r)[0]

    def rdf_force_compute_all(self, n: int, nb: int = 2048, r_max: Optional[float] = None) -> None:
        """
        Compute the force radial distribution functions (RDF) for various pairs of particles.

        Args:
            n (int): The current step or iteration number.

        Returns:
            None
        """
        # RDF startup scheme
        if n == 0:
            self.rdf_sample_counter = 0
            
            # set standard maximum rdf value
            if r_max == None:
                r_max = self.L/2  # set to default half box diagonal distance
  
            # set basic properties
            self.rdf_F_bins = np.arange(0, r_max, (r_max)/nb) # array with outer edges
            
            # Standard rdf pairs
            self.store_F_H2OH2O = []  # linked list, we will append and later reallocate to np.array
            self.rdf_F_H2OH2O = np.zeros(self.rdf_F_bins.shape[0], dtype=np.float64)  # test 128 bit floats for accuracy
            self.store_F_OHH2O = []
            self.rdf_F_OHH2O = np.zeros_like(self.rdf_F_H2OH2O)  # copy sizing from H2OH2O array
            self.store_F_KOH = []
            self.rdf_F_KOH = np.zeros_like(self.rdf_F_H2OH2O)
            self.store_F_KH2O = []
            self.rdf_F_KH2O = np.zeros_like(self.rdf_F_H2OH2O)
            if self.N_K > 1:  # only ion-ion self interactions if more than 1 is there
                self.store_F_OHOH = []
                self.rdf_F_OHOH = np.zeros_like(self.rdf_F_H2OH2O)
                self.store_F_KK = []
                self.rdf_F_KK = np.zeros_like(self.rdf_F_H2OH2O)
            if self.cheap is False: # Also execute Hydrogen interaction distances (long lists)
                self.store_F_HOH = []
                self.rdf_F_HOH = np.zeros_like(self.rdf_F_H2OH2O)
                self.store_F_HH2O = []
                self.rdf_F_HH2O = np.zeros_like(self.rdf_F_H2OH2O)
            #     self.store_F_HK = []
            #     self.rdf_F_HK = np.zeros_like(self.rdf_F_H2OH2O)
                self.store_F_HH = []
                self.rdf_F_HH = np.zeros_like(self.rdf_F_H2OH2O)
            
            self.rdf_F_T = []
            #     self.store_F_KO_all = []
            #     self.rdf_F_KO = np.zeros_like(self.rdf_F_H2OH2O)
            #     self.store_F_OO_all = []
            #     self.rdf_F_OO = np.zeros_like(self.rdf_F_H2OH2O)
        
        this_F_rdf = self.rdf_force_state_all(self.r_H2OH2O, self.d_H2OH2O, self.F_H2OH2O)
        self.store_F_H2OH2O.append(this_F_rdf)
        self.rdf_F_H2OH2O += this_F_rdf
        
        this_F_rdf = self.rdf_force_state_all(self.r_OHH2O, self.d_OHH2O, self.F_OHH2O)
        self.store_F_OHH2O.append(this_F_rdf)
        self.rdf_F_OHH2O += this_F_rdf
        
        this_F_rdf = self.rdf_force_state_all(self.r_KOH, self.d_KOH, self.F_KOH)
        self.store_F_KOH.append(this_F_rdf)
        self.rdf_F_KOH += this_F_rdf
        
        this_F_rdf = self.rdf_force_state_all(self.r_KH2O, self.d_KH2O, self.F_KH2O)
        self.store_F_KH2O.append(this_F_rdf)
        self.rdf_F_KH2O += this_F_rdf
        
        if self.N_K > 1:
            this_F_rdf = self.rdf_force_state_all(self.r_OHOH, self.d_OHOH, self.F_OHOH)
            self.store_F_OHOH.append(this_F_rdf)
            self.rdf_F_OHOH += this_F_rdf
            
            this_F_rdf = self.rdf_force_state_all(self.r_KK, self.d_KK, self.F_KK)
            self.store_F_KK.append(this_F_rdf)
            self.rdf_F_KK += this_F_rdf
        
        if self.cheap == False:
            this_F_rdf = self.rdf_force_state_all(self.r_HOH, self.d_HOH, self.F_HOH)
            self.store_F_HOH.append(this_F_rdf)
            self.rdf_F_HOH += this_F_rdf
            
            this_F_rdf = self.rdf_force_state_all(self.r_HH2O, self.d_HH2O, self.F_HH2O)
            self.store_F_HH2O.append(this_F_rdf)
            self.rdf_F_HH2O += this_F_rdf
            
            this_F_rdf = self.rdf_force_state_all(self.r_HH, self.d_HH, self.F_HH)
            self.store_F_HH.append(this_F_rdf)
            self.rdf_F_HH += this_F_rdf
        
        # Store the temperature of this snapshot
        self.rdf_F_T.append(self.T_trans[n])
        self.rdf_sample_counter += 1

    def rdf_force_state_all(self, r:np.ndarray, d:np.ndarray, F:np.ndarray) -> np.ndarray:
        """
        Calculate the radial distribution function (RDF) force state for all particles.
        This function computes the RDF force state for all particles based on their positions,
        distances, and forces. It returns an array representing the RDF force state.
        Parameters:
        -----------
        r : np.ndarray
            An array of shape (N, 3) representing the positions of the particles.
        d : np.ndarray
            An array of shape (N,) representing the distances between particles.
        F : np.ndarray
            An array of shape (N, 3) representing the forces acting on the particles.
        Returns:
        --------
        np.ndarray
            An array representing the RDF force state for all particles.
        """
        storage_array=np.zeros(np.size(self.rdf_F_bins), dtype=np.float64)

        F_dot_r = np.sum(F*r, axis=1)/d  # F dot rxyz/r (strength of F in the direction of r)
        dp = F_dot_r/d**2  # (F dot r_vec)/r_rad^3
        dp[(r[:, 0]>self.L/2)+(r[:, 1]>self.L/2)+(r[:, 2]>self.L/2)]=0  # filter
        
        # the next part is from revelsmd package, adjusted for my code
        digtized_array = np.digitize(d, self.rdf_F_bins)-1
        dp[digtized_array==np.size(self.rdf_F_bins)-1] = 0

        storage_array[(np.size(self.rdf_F_bins)-1)]= np.sum(dp[(digtized_array==np.size(self.rdf_F_bins)-1)]) #conduct heaviside for our first bin
        for l in range(np.size(self.rdf_F_bins)-2,-1,-1):
            storage_array[l]= np.sum(dp[(digtized_array==l)])#conduct subsequent heavisides with a rolling sum
        return storage_array
    
    def rdf_force_rescale_all(self, store:list, rdf:np.ndarray, interactions:int) -> np.ndarray:
        """
        Rescales the radial distribution function (rdf) and force data for all interactions.
        Parameters:
            store (list): A list containing the force data to be rescaled.
            rdf (np.ndarray): An array containing the radial distribution function data.
            interactions (int): The number of interactions to consider for rescaling.
        Returns:
            np.ndarray: A 2D array containing the rescaled rdf, rdf_zero, and rdf_inf values.
        """
        T = np.array(self.rdf_F_T)  # Local temperatures
        rescale_geo = (8*np.pi*(co.k/co.eV)*T).reshape(T.shape[0], 1)
        prefactor = self.L**3/(interactions)
        store = np.array(store)*prefactor/rescale_geo
        store_zero = np.array(np.cumsum(store, axis=1))[:,:-1]
        store_inf = np.array(1-np.cumsum(store[:,::-1], axis=1)[:,::-1][:,1:])
        store_delta = store_inf - store_zero
        
        # total parts
        rescale_geo *= len(store)
        rdf *= prefactor/rescale_geo.mean()
        rdf_zero = np.array(np.cumsum(rdf)[:-1])
        rdf_inf = np.array(1-np.cumsum(rdf[::-1])[::-1][1:])
        rdf_delta = rdf_inf - rdf_zero

        # varience part
        var_del=np.mean((store_delta-rdf_delta)**2,axis=0)
        cov_inf=np.mean((store_delta-rdf_delta)*(store_inf-rdf_inf),axis=0)
        weights = cov_inf/var_del
        rdf = np.mean(store_inf*(1-weights)+(store_zero*weights), axis=0)
        return np.array([rdf.astype(np.float64), rdf_zero.astype(np.float64), rdf_inf.astype(np.float64)])
            
    def stitching_together_all(self) -> None:
        """
        Gathers and reduces various simulation data across all MPI processes to the root process.
        This method performs the following steps:
        1. Gathers OH-, H2O, H3O+, K, and H positional and count data from all MPI processes to the root process.
        2. Gathers time arrays and hydrogen bonding arrays from all MPI processes to the root process.
        3. Rescales and reduces radial distribution functions (RDFs) for various particle interactions.
        4. Rescales and reduces force RDFs for various particle interactions.
        5. Re-centers RDF bins.
        6. Calls `stitching_together_main` on the root process to finalize the stitching of data.
        Note:
            - The method uses MPI gather and reduce operations to collect and aggregate data.
            - Rescaling of RDFs is based on the geometry and sample counters.
            - The method handles both cheap and non-cheap RDF calculations.
            - Only ion-ion self-interactions are considered if more than one ion is present.
        Returns:
            None
        """
        # prepair gethering on all cores 1) All OH- stuff
        self.OH_i = self.comm.gather(self.OH_i, root=0)
        self.OH = self.comm.gather(self.OH, root=0)
        self.n_OH = self.comm.gather(self.n_OH, root=0)
        self.OH_shift = self.comm.gather(self.OH_shift, root=0)
        
        # prepair gethering on all cores 1) All H2O stuff
        self.H2O_i = self.comm.gather(self.H2O_i, root=0)
        self.H2O = self.comm.gather(self.H2O, root=0)
        self.n_H2O = self.comm.gather(self.n_H2O, root=0)
        self.H2O_shift = self.comm.gather(self.H2O_shift, root=0)
        
        # prepair gethering on all cores 1) All H3O+ stuff
        self.n_H3O = self.comm.gather(self.n_H3O, root=0)
        
        self.K = self.comm.gather(self.pos_K, root=0)
        self.H = self.comm.gather(self.pos_H, root=0)

        # gather the time arrays as well
        self.t_full= np.copy(self.t)
        self.t_full = self.comm.gather(self.t_full, root=0)
        
        # Gather hydrogen bonding array
        self.current_OH_hbs = self.comm.gather(self.current_OH_hbs, root=0)
        
        # RDF's
        # First rescale all RDFs accordingly also prepaire for averaging using mpi.sum
        self.r_cent = (self.r[:-1] + self.r[1:])/2  # central point of rdf bins
        rdf_sample_counter_all = self.comm.allreduce(self.rdf_sample_counter, op=MPI.SUM)
        rescale_geometry = (4*np.pi*self.r_cent**2)*(self.r[1] - self.r[0])  # 4*pi*r^2*dr
        rescale_counters = (self.L**3)/(rdf_sample_counter_all)
        rescale = rescale_counters/rescale_geometry

        # TODO: Make sure to do Na*Na/2 (not Na*(Na-1)/2)
        self.rdf_H2OH2O *= rescale/(self.N_H2O*(self.N_H2O - 1)*0.5)  # rdf*L_box^3/(n_sample*n_interactions*geometry_rescale/n_cores)
        self.rdf_OHH2O *= rescale/(self.N_OH*self.N_H2O)
        self.rdf_KOH *= rescale/(self.N_OH*self.N_K)
        self.rdf_KH2O *= rescale/(self.N_H2O*self.N_K)
        if self.N_K > 1:  # only ion-ion self interactions if more than 1 is there
            self.rdf_OHOH *= rescale/(self.N_OH*(self.N_OH - 1)*0.5)  
            self.rdf_KK *= rescale/(self.N_K*(self.N_K - 1)*0.5)
        if self.cheap is False:
            self.rdf_HOH *= rescale/(self.N_H*self.N_OH)
            self.rdf_HH2O *= rescale/(self.N_H*self.N_H2O)
            # self.rdf_HK *= rescale/(self.N_H*self.N_K)
            self.rdf_HH *= rescale/(self.N_H*(self.N_H - 1)*0.5)
            self.rdf_KO_all *= rescale/(self.N_K*self.N_O)
            self.rdf_OO_all *= rescale/(self.N_O*(self.N_O - 1)*0.5)

        # Then communicate these to main core.
        self.rdf_H2OH2O = self.comm.reduce(self.rdf_H2OH2O, op=MPI.SUM, root=0)
        self.rdf_OHH2O = self.comm.reduce(self.rdf_OHH2O, op=MPI.SUM, root=0)       
        self.rdf_KOH = self.comm.reduce(self.rdf_KOH, op=MPI.SUM, root=0)
        self.rdf_KH2O = self.comm.reduce(self.rdf_KH2O, op=MPI.SUM, root=0)

        if self.N_K > 1:  # only ion-ion self interactions if more than 1 is there
            self.rdf_OHOH = self.comm.reduce(self.rdf_OHOH, op=MPI.SUM, root=0)
            self.rdf_KK = self.comm.reduce(self.rdf_KK, op=MPI.SUM, root=0)
        
        if self.cheap is False:
            self.rdf_HOH = self.comm.reduce(self.rdf_HOH, op=MPI.SUM, root=0)
            self.rdf_HH2O = self.comm.reduce(self.rdf_HH2O, op=MPI.SUM, root=0)
            # self.rdf_HK = self.comm.reduce(self.rdf_HK, op=MPI.SUM, root=0)
            self.rdf_HH = self.comm.reduce(self.rdf_HH, op=MPI.SUM, root=0)
            self.rdf_KO_all = self.comm.reduce(self.rdf_KO_all, op=MPI.SUM, root=0)
            self.rdf_OO_all = self.comm.reduce(self.rdf_OO_all, op=MPI.SUM, root=0)

        # The Force RDF
        # Rescale
        rdf_F_H2OH2O = self.rdf_force_rescale_all(self.store_F_H2OH2O, self.rdf_F_H2OH2O, self.N_H2O*(self.N_H2O-1)/2)/self.size
        rdf_F_OHH2O = self.rdf_force_rescale_all(self.store_F_OHH2O, self.rdf_F_OHH2O, self.N_H2O*self.N_OH)/self.size
        rdf_F_KOH = self.rdf_force_rescale_all(self.store_F_KOH, self.rdf_F_KOH, self.N_K*self.N_OH)/self.size
        rdf_F_KH2O = self.rdf_force_rescale_all(self.store_F_KH2O, self.rdf_F_KH2O, self.N_K*self.N_H2O)/self.size
        if self.N_K > 1:
            rdf_F_OHOH = self.rdf_force_rescale_all(self.store_F_OHOH, self.rdf_F_OHOH, self.N_OH*(self.N_OH-1)/2)/self.size
            rdf_F_KK = self.rdf_force_rescale_all(self.store_F_KK, self.rdf_F_KK, self.N_K*(self.N_K-1)/2)/self.size
        if self.cheap is False:
            rdf_F_HOH = self.rdf_force_rescale_all(self.store_F_HOH, self.rdf_F_HOH, self.N_H*self.N_OH)/self.size
            rdf_F_HH2O = self.rdf_force_rescale_all(self.store_F_HH2O, self.rdf_F_HH2O, self.N_H*self.N_H2O)/self.size
            rdf_F_HH = self.rdf_force_rescale_all(self.store_F_HH, self.rdf_F_HH, self.N_H*(self.N_H-1)/2)/self.size
        # recenter bins
        self.rdf_F_bins = (self.rdf_F_bins[1:] + self.rdf_F_bins[:-1])/2

        # Then communicate these to main core.
        self.rdf_F_H2OH2O = self.comm.reduce(rdf_F_H2OH2O, op=MPI.SUM, root=0)
        self.rdf_F_OHH2O = self.comm.reduce(rdf_F_OHH2O, op=MPI.SUM, root=0)
        self.rdf_F_KOH = self.comm.reduce(rdf_F_KOH, op=MPI.SUM, root=0)
        self.rdf_F_KH2O = self.comm.reduce(rdf_F_KH2O, op=MPI.SUM, root=0)
        if self.N_K > 1:
            self.rdf_F_OHOH = self.comm.reduce(rdf_F_OHOH, op=MPI.SUM, root=0)
            self.rdf_F_KK = self.comm.reduce(rdf_F_KK, op=MPI.SUM, root=0)
        if self.cheap is False:
            self.rdf_F_HOH = self.comm.reduce(rdf_F_HOH, op=MPI.SUM, root=0)
            self.rdf_F_HH2O = self.comm.reduce(rdf_F_HH2O, op=MPI.SUM, root=0)
            self.rdf_F_HH = self.comm.reduce(rdf_F_HH, op=MPI.SUM, root=0)
        # Stich together correctly on main cores
        if self.rank == 0:
            self.stitching_together_main()
      
    def stitching_together_main(self) -> None:
        """
        This method stitches together simulation data across multiple segments, ensuring continuity and consistency
        for OH- and H2O molecules. It handles mismatches at the boundaries of segments, swaps columns to align data,
        and adjusts for periodic boundary conditions (PBC). The method also combines the adjusted arrays back into
        their original shapes.
        Steps:
        1. For each segment boundary, check for mismatches in OH- and H2O indices.
        2. Swap columns in the data arrays to align mismatched indices.
        3. Handle cases where reactions occur at the stitching location.
        4. Adjust for periodic boundary conditions in the shift columns.
        5. Combine the adjusted arrays back into their original shapes.
        Returns:
            None
        """
        # NOW ADD 1 reaction recognition and 2 reordering
        # for every end of 1 section check with start next one
        for n in range(self.size-1):
            # OH-
            OH_not_found = np.empty(0)
            mismatch_indices = np.where(self.OH_i[n][-1, :] != self.OH_i[n+1][0, :])[0]
            # Swap columns in C_modified based on the mismatch
            for index in mismatch_indices:
                # Find the index in B corresponding to the value in A
                b_index = np.where(self.OH_i[n+1][0, :] == self.OH_i[n][-1, index])[0]
                if b_index.size == 0:
                    OH_not_found = np.append(OH_not_found, self.OH_i[n][-1, index])
                    if self.verbose is True:
                        print('CHECK: OH part Reaction occured on stitching back together', flush=True)

                else:
                    self.OH_i[n+1][:, [index, b_index[0]]] = self.OH_i[n+1][:, [b_index[0], index]]
                    self.OH[n+1][:, [index, b_index[0]], :] = self.OH[n+1][:, [b_index[0], index], :]
                    self.OH_shift[n+1][[index, b_index[0]], :] = self.OH_shift[n+1][[b_index[0], index], :]
            
            # H2O
            H2O_not_found = np.empty(0)
            mismatch_indices = np.where(self.H2O_i[n][-1, :] != self.H2O_i[n+1][0, :])[0]
            # Swap columns in C_modified based on the mismatch
            for index in mismatch_indices:
                # Find the index in B corresponding to the value in A
                b_index = np.where(self.H2O_i[n+1][0, :] == self.H2O_i[n][-1, index])[0]
                if b_index.size == 0:
                    H2O_not_found = np.append(H2O_not_found, self.H2O_i[n][-1, index])
                    if self.verbose is True:
                        print('CHECK: H2O part Reaction occured on stitching back together', flush=True)
                else:
                    self.H2O_i[n+1][:, [index, b_index[0]]] = self.H2O_i[n+1][:, [b_index[0], index]]
                    self.H2O[n+1][:, [index, b_index[0]], :] = self.H2O[n+1][:, [b_index[0], index], :]
                    self.H2O_shift[n+1][[index, b_index[0]], :] = self.H2O_shift[n+1][[b_index[0], index], :]
            
            #### these two for loops only activate if there are reactions at stichting location
            idx_H2O = np.zeros_like(H2O_not_found, dtype=int)
            for i, H2O_i in enumerate(H2O_not_found):
                idx_H2O[i] = np.where(self.H2O_i[n][-1, :] == H2O_i)[0][0]

            for OH_old in OH_not_found:
                # OH part
                idx_OH_old = np.where(self.OH_i[n][-1, :] == OH_old)[0][0]

                # find closest H2O for reaction in the index_H2O lists
                r_OO = (self.OH[n][-1, idx_OH_old] - self.H2O[n][-1, idx_H2O] + self.L/2) % self.L - self.L/2
                d2 = np.sum(r_OO**2, axis=1)
                i = np.argmin(d2) # index in the H2O not found array
                if d2[i] > 9:  # excape when jump too large
                        print("Strange behaviour found, reaction jump too far, d = ", np.sqrt(d2[i]), 'Angstrom',
                                        "during stitching back together", flush=True)
                H2O_old = H2O_not_found[i]
                idx_H2O_old = idx_H2O[i]
                
                OH_new = H2O_old
                H2O_new = OH_old
                
                idx_OH_new = np.where(self.OH_i[n+1][0, :] == OH_new)[0][0]
                idx_H2O_new = np.where(self.H2O_i[n+1][0, :] == H2O_new)[0][0]
                # check if index of OH_new is already the right one
                
                if self.verbose is True:
                    print('matching up OH', OH_old, OH_new, idx_OH_old, idx_OH_new)
                    print('matching up H2O', H2O_old, H2O_new, idx_H2O_old, idx_H2O_new)
                if idx_OH_new != idx_OH_old: ## if not, swap around
                    self.OH_i[n+1][:, [idx_OH_old, idx_OH_new]] = self.OH_i[n+1][:, [idx_OH_new, idx_OH_old]]
                    self.OH[n+1][:, [idx_OH_old, idx_OH_new], :] = self.OH[n+1][:, [idx_OH_new, idx_OH_old], :]
                    self.OH_shift[n+1][[idx_OH_old, idx_OH_new], :] = self.OH_shift[n+1][[idx_OH_new, idx_OH_old], :]
                
                if idx_H2O_new != idx_H2O_old: ## if not, swap around
                    self.H2O_i[n+1][:, [idx_H2O_new, idx_H2O_new]] = self.H2O_i[n+1][:, [idx_H2O_new, idx_H2O_old]]
                    self.H2O[n+1][:, [idx_H2O_old, idx_H2O_new], :] = self.H2O[n+1][:, [idx_H2O_new, idx_H2O_old], :]
                    self.H2O_shift[n+1][[idx_H2O_old, idx_H2O_new], :] = self.H2O_shift[n+1][[idx_H2O_new, idx_H2O_old], :]
                # so now we now for sure that idx_H2O_old=idx_H2O_new and index_OH_old=index_OH_new
                # and that H2O_old=OH_new and that OH_old=H2O_new
                
                # Only check for pbc during a reaction for the OH
                dis = (self.OH[n+1][0, idx_OH_old, :] - self.OH[n][-1, idx_OH_old, :] + self.L/2) % self.L - self.L/2
                real_loc = self.OH[n][-1, idx_OH_old, :] + dis
                self.OH_shift[n][idx_OH_old, :] = np.round((real_loc - self.OH[n+1][0, idx_OH_old, :])/self.L).astype(int)
                
                # Only check for pbc during a reaction for the H2O
                dis = (self.H2O[n+1][0, idx_H2O_old, :] - self.H2O[n][-1, idx_H2O_old, :] + self.L/2) % self.L - self.L/2
                real_loc = self.H2O[n][-1, idx_H2O_old, :] + dis
                self.H2O_shift[n][idx_H2O_old, :] = np.round((real_loc - self.H2O[n+1][0, idx_H2O_old, :])/self.L).astype(int)
            #### Till here

            # Now adjust for passing PBC in the shift collumns. OH-
            self.OH[n+1][:, :, :] += self.L*self.OH_shift[n][:, :]
            self.OH_shift[n+1][:, :] += self.OH_shift[n][:, :]
            
            # Now adjust for passing PBC in the shift collumns. H2O
            self.H2O[n+1][:, :, :] += self.L*self.H2O_shift[n][:, :]
            self.H2O_shift[n+1][:, :] += self.H2O_shift[n][:, :]
            
        # Combining adjusted arrays back to the right shape
        # OH-
        self.OH = np.concatenate(self.OH, axis=0)
        self.OH_i = np.concatenate(self.OH_i, axis=0)
        self.n_OH = np.concatenate(self.n_OH, axis=0)
        
        # H2O
        self.H2O = np.concatenate(self.H2O, axis=0)
        self.H2O_i = np.concatenate(self.H2O_i, axis=0)
        self.n_H2O = np.concatenate(self.n_H2O, axis=0)
        
        # H3O+
        self.n_H3O = np.concatenate(self.n_H3O, axis=0)
        
        # K+
        self.K = np.concatenate(self.K, axis=0)
        
        # H
        self.H = np.concatenate(self.H,  axis=0)
        
        # Combine hbonding array
        self.current_OH_hbs = np.concatenate(self.current_OH_hbs, axis=0)

        # RDF functionality has no need to do anything
        
        # Getting a time array
        self.t_full = np.concatenate(self.t_full, axis=0)

    def calculate_next_OH_main(self) -> np.ndarray:
        """
        Calculate the next OH- molecule index for each time step.

        This function creates an array that identifies the next OH- molecule index
        for each time step. If no new index is found, the next index is set to 0.

        Returns:
            numpy.ndarray: An array with the next OH- molecule index for each time step.
        """
        next_OH_i = np.zeros_like(self.OH_i)
        for i in range(self.OH_i.shape[1]):
            current_indices = self.OH_i[:, i]
            change_indices = np.where(current_indices[:-1] != current_indices[1:])[0] + 1
            old_change_index = 0
            for new_change_index in change_indices:
                next_OH_i[old_change_index:new_change_index, i] = self.OH_i[new_change_index, i]
                old_change_index = new_change_index
        return next_OH_i               

    def loop_timesteps_next_OH_all(self) -> None:
        """
        Perform a loop over all timesteps to calculate intermolecular distances and hydrogen bonds for OH- ions 
        using MPI for parallel processing.
        This method performs the following steps:
        1. Preparation steps with MPI communication to distribute the workload among different ranks.
        2. Calculate intermolecular distances for OH- ions and hydrogen bonds at each timestep.
        3. Gather the results from all ranks and concatenate them on the root rank.
        Returns:
            None
        """
        # Preperation steps with communication
        if self.rank == 0:
            # Calculate the next OH- index
            self.next_OH_i = self.calculate_next_OH_main()
            
            # prep to split the array appropriately
            self.next_OH_i_split = np.array_split(self.next_OH_i, self.size, axis=0)
        else:
            # all other ranks
            self.next_OH_i_split = None  # create empty dummy on all cores
        
        self.next_OH_i = self.comm.scatter(self.next_OH_i_split, root=0)  # scatter the array to all cores

        # All the communication done. Now calculate intermolecular distances again.
        self.next_OH_hbs = np.zeros((self.n_max, 5, 2), dtype=int)
        for n in range(self.n_max):  # Loop over all timesteps
            if (n % 10000 == 0) and (self.verbose is True) and (n > 0):
                print("time is:", self.t[n], " in pass 2, rank is:", self.rank, flush=True)
            # Calculate only OH dist seances for OH- recognition
            self.r_HO = (self.pos_O[n, self.idx_HO[1], :] - self.pos_H[n, self.idx_HO[0], :] + self.L/2) % self.L - self.L/2
            self.d_HO = np.sqrt(np.sum(self.r_HO**2, axis=1))
            self.O_per_H = np.argmin(self.d_HO.reshape((self.N_H, self.N_O)), axis=1)
        
            # Oxygen-Oxygen
            # already compted for Hydrogen bonding numbers
            self.r_OO = (self.pos_O[n, self.idx_OO[1], :] - self.pos_O[n, self.idx_OO[0], :] + self.L/2) % self.L - self.L/2
            self.d_OO = np.sqrt(np.sum(self.r_OO**2, axis=1))

            self.next_OH_hbs[n] = self.hydrogen_bonds(self.next_OH_i[n], n)

        # Gather all the results
        # Gather hydrogen bonding array
        self.next_OH_hbs = self.comm.gather(self.next_OH_hbs, root=0)
        self.next_OH_i = self.comm.gather(self.next_OH_i, root=0)
        if self.rank == 0 and self.verbose is True:
            self.next_OH_hbs = np.concatenate(self.next_OH_hbs, axis=0)
            self.next_OH_i = np.concatenate(self.next_OH_i, axis=0)
            print('Time calculating distances pass', time.time() - self.tstart, flush=True)
    
    def compute_MSD_pos(self) -> None:
        """
        Compute the Mean Squared Displacement (MSD) for different particle types.
        This method calculates the MSD for OH, H2O, and K particles using the 
        freud library's windowed MSD calculation mode. The results are stored 
        in the instance variables `msd_OH`, `msd_H2O`, and `msd_K`.
        Returns:
            None
        """
        # prepaire windowed MSD calculation mode with freud
        msd = freud.msd.MSD(mode='window')
        
        self.msd_OH = msd.compute(self.OH).msd
        self.msd_H2O = msd.compute(self.H2O).msd
        self.msd_K = msd.compute(self.K).msd
    
    def compute_MSD_pres(self) -> None:
        """
        Compute the Mean Squared Displacement (MSD) of the pressure tensor components.
        This method calculates the MSD of the pressure tensor components by first loading
        the relevant components of the stress tensor into an array, then computing the 
        running integrals for each component using Simpson's rule, and finally calculating 
        the mean squared displacement.
        The pressure tensor components considered are:
        - xy component of the stress tensor
        - xz component of the stress tensor
        - yz component of the stress tensor
        - 0.5 * (xx - yy) component of the stress tensor
        - 0.5 * (yy - zz) component of the stress tensor
        The stress tensor components are scaled by 1e8 to convert units appropriately.
        The running integrals are computed using the cumulative Simpson's rule with a 
        time step of `self.dt * 1e-15`.
        The resulting MSD is stored in the `self.msd_P` attribute.
        Returns:
            None
        """
        # Load all pressure states to array.
        p_ab = np.array([
            self.stress[:, 0, 1],  # xy
            self.stress[:, 0, 2],  # xz
            self.stress[:, 1, 2],  # yz
            (self.stress[:, 0, 0] - self.stress[:, 1, 1]) / 2,  # 0.5 xx-yy
            (self.stress[:, 1, 1] - self.stress[:, 2, 2]) / 2   # 0.5 yy-zz
            ]) * 1e8
        
        # Compute the running integrals for each component
        p_ab_int = integrate.cumulative_simpson(p_ab, dx=self.dt*1e-15, axis=-1, initial=None)
        self.msd_P = integral =(p_ab_int**2).mean(axis=0)

    def save_results_all(self) -> None:
        """
        Save the results of the simulation.
        This method handles the saving of results for both single-core and multi-core
        simulations. It performs the following tasks:
        - Determines the appropriate folder path based on the number of cores.
        - Creates the output HDF5 file on the main core.
        - Optionally saves position output as .xyz files on separate cores.
        - Communicates arrays between cores using MPI.
        Attributes:
            size (int): The number of cores used in the simulation.
            folder (str): The folder path where results are saved.
            rank (int): The rank of the current core.
            xyz_out (bool): Flag indicating whether to save position output as .xyz files.
            pos_all (np.ndarray): Array containing all positions.
            pos_pro (np.ndarray): Array containing processed positions.
            OH (np.ndarray): Array containing OH positions.
            H2O (np.ndarray): Array containing H2O positions.
            K (np.ndarray): Array containing K positions.
            H (np.ndarray): Array containing H positions.
            comm (MPI.Comm): MPI communicator for broadcasting data between cores.
            center (bool): Flag indicating whether to center the positions in the .xyz files.
        Returns:
            None
        """
        # separate single core or multi core folders
        if self.size == 1:
            # path = os.path.normpath(self.folder + r"/single_core/")
            path = self.folder
        else:
            path = self.folder
        # self.save_numpy_files_main(path)

        # create the output.h5 file always on main core
        if self.rank == 0:
            self.create_dataframe_main(path)

        # save position output if needed as .xyz (4 seperate files) on separate cores
        # communicate arrays
        if self.xyz_out is not False:
            if self.rank == 0:
                shape = self.pos_all.shape
                self.pos_pro = np.concatenate((self.OH, self.H2O,
                                               self.K, self.H), axis=1)
            else:
                shape = None

            shape = self.comm.bcast(shape, root=0)

            if self.rank != 0:
                self.pos_all = np.empty(shape, dtype=np.float64)
                self.pos_pro = np.empty(shape, dtype=np.float64)

            self.comm.Bcast(self.pos_pro, root=0)
            self.comm.Bcast(self.pos_all, root=0)
            for i in range(4):
                if (i+1) % self.size == self.rank:
                    self.write_to_xyz_all(i, center=self.center)

    def create_dataframe_main(self, path: str) -> None:
        """
        Creates an HDF5 file at the specified path and populates it with various datasets.
        Parameters:
        path (str): The directory path where the output HDF5 file will be created.
        The function performs the following tasks:
        - Creates an HDF5 file named 'output.h5' in the specified directory.
        - Populates the file with system properties, radial distribution functions (rdfs), 
          force rdfs, mean squared displacements (msds), and transient properties over time.
        - Includes additional datasets based on the values of `self.N_K` and `self.cheap`.
        Datasets created:
        - System properties: Lbox, N_OH, N_K, N_H2O.
        - Radial distribution functions (rdfs): r, g_H2OH2O(r), g_OHH2O(r), g_KH2O(r), g_KOH(r), 
          g_OHOH(r), g_KK(r), g_HOH(r), g_HH2O(r), g_HH(r), g_KO(r), g_OO(r).
        - Force rdfs: r, g_H2OH2O(r), g_OHH2O(r), g_KOH(r), g_KH2O(r), g_OHOH(r), g_KK(r), 
          g_HOH(r), g_HH2O(r), g_HH(r).
        - Mean squared displacements (msds): OH, H2O, K, P.
        - Transient properties over time: time, index_OH, index_next_OH, index_H2O, index_K, 
          pos_OH, pos_H2O, pos_K, stresses, energies, current_OH_hbs, next_OH_hbs.
        If `self.verbose` is True, prints messages indicating the progress of file preparation 
        and writing.
        """
        file = {os.path.normpath(path + '/output.h5')}
        if self.verbose is True:
            print(f'Prepairing outputfile {file} completed on rank: {self.rank}')

        # create large dataframe with output
        df = h5py.File(os.path.normpath(path + '/output.h5'), "w")

        # System properties
        df.create_dataset("system/Lbox", data=self.L)
        df.create_dataset("system/N_OH", data=self.N_OH)
        df.create_dataset("system/N_K", data=self.N_K)
        df.create_dataset("system/N_H2O", data=self.N_H2O)

        # rdfs
        df.create_dataset("rdf/r", data=self.r_cent)
        df.create_dataset("rdf/g_H2OH2O(r)", data=self.rdf_H2OH2O)
        df.create_dataset("rdf/g_OHH2O(r)", data=self.rdf_OHH2O)
        df.create_dataset("rdf/g_KH2O(r)", data=self.rdf_KH2O)
        df.create_dataset("rdf/g_KOH(r)", data=self.rdf_KOH)

        if self.N_K > 1:
            df.create_dataset("rdf/g_OHOH(r)", data=self.rdf_OHOH)
            df.create_dataset("rdf/g_KK(r)", data=self.rdf_KK)

        if self.cheap is False:
            df.create_dataset("rdf/g_HOH(r)", data=self.rdf_HOH)
            df.create_dataset("rdf/g_HH2O(r)", data=self.rdf_HH2O)
            # df.create_dataset("rdf/g_HK(r)", data=self.rdf_HK)
            df.create_dataset("rdf/g_HH(r)", data=self.rdf_HH)
            df.create_dataset("rdf/g_KO(r)", data=self.rdf_KO_all)
            df.create_dataset("rdf/g_OO(r)", data=self.rdf_OO_all)

        # force rdfs
        df.create_dataset("rdf_F/r", data=self.rdf_F_bins)
        df.create_dataset("rdf_F/g_H2OH2O(r)", data=self.rdf_F_H2OH2O)
        df.create_dataset("rdf_F/g_OHH2O(r)", data=self.rdf_F_OHH2O)
        df.create_dataset("rdf_F/g_KOH(r)", data=self.rdf_F_KOH)
        df.create_dataset("rdf_F/g_KH2O(r)", data=self.rdf_F_KH2O)
        if self.N_K > 1:
            df.create_dataset("rdf_F/g_OHOH(r)", data=self.rdf_F_OHOH)
            df.create_dataset("rdf_F/g_KK(r)", data=self.rdf_F_KK)
        if self.cheap is False:
            df.create_dataset("rdf_F/g_HOH(r)", data=self.rdf_F_HOH)
            df.create_dataset("rdf_F/g_HH2O(r)", data=self.rdf_F_HH2O)
            df.create_dataset("rdf_F/g_HH(r)", data=self.rdf_F_HH)

        # msds
        df.create_dataset("msd/OH", data=self.msd_OH)
        df.create_dataset("msd/H2O", data=self.msd_H2O)
        df.create_dataset("msd/K", data=self.msd_K)
        df.create_dataset("msd/P", data=self.msd_P)
        
        # properties over time
        df.create_dataset("transient/time", data=self.t_full)
        df.create_dataset("transient/index_OH", data=self.OH_i)
        df.create_dataset("transient/index_next_OH", data=self.next_OH_i)
        df.create_dataset("transient/index_H2O", data=self.H2O_i)
        df.create_dataset("transient/index_K", data=self.K_i)
        df.create_dataset("transient/pos_OH", data=self.OH)
        df.create_dataset("transient/pos_H2O", data=self.H2O)
        df.create_dataset("transient/pos_K", data=self.K)
        df.create_dataset("transient/stresses", data=self.stress)
        df.create_dataset("transient/energies", data=self.energy)
        df.create_dataset("transient/current_OH_hbs", data=self.current_OH_hbs)
        df.create_dataset("transient/next_OH_hbs", data=self.next_OH_hbs)
        df.close()
        
        if self.verbose is True:
            print(f'writing outputfile {file} completed on rank: {self.rank}')

    def write_to_xyz_all(self, type, center=False) -> None:
        """
        Writes the atomic positions to an XYZ or PDB file.
        Parameters:
        -----------
        type : int
            Specifies the type of positions to write:
            - 0: Unwrapped unprocessed positions.
            - 1: Wrapped unprocessed positions.
            - 2: Unwrapped processed positions.
            - 3: Wrapped processed positions.
        center : bool, optional
            If True, centers the positions before wrapping (default is False).
        Raises:
        -------
        ValueError
            If an unsupported molecular file output format is specified.
        Notes:
        ------
        - The method uses the `ase` library to write the atomic configurations.
        - The output format is determined by the `self.xyz_out` attribute, which can be 'xyz' or 'pdb'.
        - The positions and types of atoms are determined based on the `type` parameter.
        - If `self.verbose` is True, progress messages are printed.
        """
        # assesing tasks correctly
        
        # I do have positions of the OH- at every timestep
        if type == 0:
            ## unwrapped unprocessed postitions
            types = ['H']*self.N_H + ['O']*self.N_O + ['K']*self.N_K
            pos = self.pos_all
            name = '/traj_unprocessed_unwrapped'
        if type == 1:
            ## wrapped processed postitions
            types = ['H']*self.N_H + ['O']*self.N_O + ['K']*self.N_K
            if center:
                pos = self.pos_all - self.pos_pro[:, 0, np.newaxis, :] + self.L / 2
            else:
                pos = self.pos_all
            pos = pos % self.L
            name = '/traj_unprocessed_wrapped'
        if type == 2:
            ## unwrapped unprocessed postitions
            types = ['F']*self.N_OH + ['O']*self.N_H2O + ['K']*self.N_K + ['H']*self.N_H
            pos = self.pos_pro
            name = '/traj_processed_unwrapped'
        if type == 3:
            ## wrapped processed postitions
            types = ['F']*self.N_OH + ['O']*self.N_H2O + ['K']*self.N_K + ['H']*self.N_H
            
            if center:
                pos = self.pos_pro - self.pos_pro[:, 0, np.newaxis, :] + self.L / 2
            else:
                pos = self.pos_pro

            pos = pos % self.L
            name = '/traj_processed_wrapped'

        if self.verbose is True:
            print(f'prepare for writing {name} started on rank: {self.rank}')

        if self.xyz_out == 'xyz':
            file_ext = '.xyz'
            format = 'xyz'
        elif self.xyz_out == 'pdb':
            file_ext = '.pdb'
            format = 'proteindatabank'
        else:
            raise ValueError(f"Unsupported molecular file output format: {format}. Either disable output with xyz=False, or set 'xyz' or 'pdb'.")
            
        configs = [None]*pos.shape[0]
        for i in range(self.pos_all.shape[0]):
            configs[i] = Atoms(types, positions=pos[i, :, :], cell=[self.L, self.L, self.L], pbc=True)
        io.write(os.path.normpath(self.folder+name+file_ext), configs, format=format, parallel=False)

        if self.verbose is True:
            print(f'writing {self.folder+name} completed on rank: {self.rank}')

    def save_numpy_files_main(self, path) -> None:
        """
        Save simulation data to a compressed NumPy file.

        Parameters:
        path (str): The directory path where the output file will be saved.

        The method creates the specified directory if it does not exist. It then saves various simulation data arrays
        into a compressed .npz file named 'output.npz' in the specified directory. The data saved includes tracking 
        information for OH- and H2O, as well as radial distribution function (rdf) data.

        If the 'cheap' attribute of the class instance is False, the method saves additional rdf data.

        Raises:
        OSError: If the directory creation fails.
        """
        try:
            os.makedirs(path, exist_ok=True)
        except OSError as error:
            # print("os.mkdir threw error, but continued with except:", error)
            error = 1
        if self.cheap is False:
            np.savez(os.path.normpath(path + "/output.npz"),
                    OH_i=self.OH_i, OH=self.OH, H2O_i=self.H2O_i, H2O=self.H2O,  # tracking OH-
                    r_rdf=self.r_cent, rdf_H2OH2O=self.rdf_H2OH2O, rdf_OHH2O=self.rdf_OHH2O, rdf_KH2O=self.rdf_KH2O)
                    # rdf_HOH=self.rdf_HOH, rdf_HK=self.rdf_HK, rdf_HH=self.rdf_HH, rdf_KO_all=self.rdf_KO_all, rdf_OO_all=self.rdf_OO_all)  # sensing the rdf
        else:
            np.savez(os.path.normpath(path + r"/output.npz"),
                    OH_i=self.OH_i, OH=self.OH, H2O_i=self.H2O_i, H2O=self.H2O,  # tracking OH-
                    r_rdf=self.r_cent, rdf_H2OH2O=self.rdf_H2OH2O, rdf_OHH2O=self.rdf_OHH2O, rdf_KH2O=self.rdf_KH2O)  # sensing the rdf


# Example usage

### TEST LOCATIONS ###
# Traj = Prot_Hop(r"/mnt/c/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/RPBE_Production/AIMD/10ps/i_1/")
# Traj = Prot_Hop(r"/mnt/c/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/RPBE_Production/MLMD/100ps_Exp_Density/i_1")
# Traj = Prot_Hop(r"/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/RPBE_Production/MLMD/100ps_Exp_Density/i_1")
# Traj = Prot_Hop(r"/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/RPBE_Production/AIMD/10ps/i_1/")
# Traj = Prot_Hop(r"/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/Quantum_to_Transport/post-processing scripts/KOH systems/test_output/", verbose=True)
# Traj1 = Prot_Hop("/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/Quantum_to_Transport/post-processing scripts/KOH systems/test_output/combined_simulation/", cheap=False, xyz_out=False, verbose=True)
# Traj2 = Prot_Hop(r"/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/Quantum_to_Transport/post-processing scripts/KOH systems/test_output/longest_up_till_now/", cheap=True, xyz_out=True, verbose=True)
# Traj3 = Prot_Hop(r"/Users/vlagerweij/Documents/TU jaar 6/Project KOH(aq)/Repros/Quantum_to_Transport/post-processing scripts/KOH systems/test_output/", cheap=False, xyz_out=True, verbose=True)

Traj = Prot_Hop(r"./", cheap=False, xyz_out='xyz', verbose=True, center=True)
