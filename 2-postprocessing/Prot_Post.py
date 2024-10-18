import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.constants as co
from scipy import integrate
import os
import re
import h5py
from typing import Tuple
import scipy as sp

class Prot_Post:
    def __init__(self, folder, verbose=True):
        self.folder = os.path.normpath(folder)
        try:
            if (os.name == 'posix') and ('WSL_DISTRO_NAME' in os.environ):
                # Convert Windows path to WSL path
                self.folder = os.path.normpath('/mnt/c'+self.folder)
            elif (os.name == 'posix') and ('delftblue' == os.environ['CMD_WLM_CLUSTER_NAME']):
                # Convert to fully relative paths
                self.folder = os.path.join(os.getcwd(), self.folder)
            elif (os.name == 'nt') and ('vlagerweij == os.getlogin()'):
                # use standard windows file path
                self.folder = self.folder
        except:
            if verbose:
                print('no automatic filepath conversiona evailable use relative path')
            self.folder = os.path.normpath(os.getcwd()+'/'+self.folder)
        self.load_properties()

    def load_properties(self):
        input = h5py.File(os.path.normpath(self.folder+'/output.h5'), 'r')

        # System properties
        self.L = input["system/Lbox"][()]
        self.N_OH = input["system/N_OH"][()]
        self.N_K = input["system/N_K"][()]
        self.N_H2O = input["system/N_H2O"][()]
        
        # Retrieving the msd properties
        self.msd_OH = input["msd/OH"][()]
        self.msd_H2O = input["msd/H2O"][()]
        self.msd_K = input["msd/K"][()]
        self.msd_P = input["msd/P"][()]
        
        # retrieving the transient properties
        self.t = input["transient/time"][()]
        self.stress = input["transient/stresses"][()]
        self.energy = input["transient/energies"][()]
        
        self.index_OH = input["transient/index_OH"][()]
        self.index_K = input["transient/index_K"][()]
        self.index_H2O = input["transient/index_H2O"][()]
        
        self.OH = input["transient/pos_OH"][()]
        self.K = input["transient/pos_K"][()]
        self.H2O = input["transient/pos_H2O"][()]
        
        self.hbs = input["transient/hydrongbonding"][()]
        
        self.N_OH = self.OH.shape[1]
        self.N_K = self.K.shape[1]
        self.N_H2O = self.H2O.shape[1]
        self.N_m = np.array([self.N_OH, self.N_K, self.N_H2O], dtype=int)  # count of molecule types OH- K+ H2O
        self.N_m_n = np.array(['OH', 'K', 'H2O'], dtype=str)
        self.N = np.array([self.N_OH + 2*self.N_H2O, self.N_OH + self.N_H2O, self.N_K], dtype=int)  # count of atom types, H, O, K
        self.N_n = np.array(['H', 'K', 'K'], dtype=str)
        
         # load rdfs
        self.rdf_r = input["rdf/r"][()]
        self.rdf_H2OH2O = input["rdf/g_H2OH2O(r)"][()]
        self.rdf_OHH2O = input["rdf/g_OHH2O(r)"][()]
        self.rdf_KH2O = input["rdf/g_KH2O(r)"][()]
        self.rdf_KOH = input["rdf/g_KOH(r)"][()]
        
        if self. N_K > 1:
            self.rdf_OHOH = input["rdf/g_OHOH(r)"][()]        
            self.rdf_KK = input["rdf/g_KK(r)"][()]
    
        try:
            self.rdf_HOH = input["rdf/g_HOH(r)"][()]
            self.cheap = False
        except:
            self.cheap = True
        
        if self.cheap is False:
            self.rdf_HH2O = input["rdf/g_HH2O(r)"][()]
            # self.rdf_HK = input["rdf/g_HK(r)"][()]
            self.rdf_HH = input["rdf/g_HH(r)"][()]
            self.rdf_KO_all = input["rdf/g_KO(r)"][()]
            self.rdf_OO_all = input["rdf/g_OO(r)"][()]
        
        # retrieve the force rdfs
        self.rdf_F_r = input["rdf_F/r"][()]
        self.rdf_F_H2OH2O = input["rdf_F/g_H2OH2O(r)"][()]
        self.rdf_F_OHH2O = input["rdf_F/g_OHH2O(r)"][()]
        self.rdf_F_KOH = input["rdf_F/g_KOH(r)"][()]
        self.rdf_F_KH2O = input["rdf_F/g_KH2O(r)"][()]
        if self. N_K > 1:
            self.rdf_F_OHOH = input["rdf_F/g_OHOH(r)"][()]        
            self.rdf_F_KK = input["rdf_F/g_KK(r)"][()]
        if self.cheap is False:
            self.rdf_F_HOH = input["rdf_F/g_HOH(r)"][()]
            self.rdf_F_HH2O = input["rdf_F/g_HH2O(r)"][()]
            self.rdf_F_HH = input["rdf_F/g_HH(r)"][()]
            
        # compare with sana's results if there.
        if "sana" in input:
            self.sana = True
            self.sana_index_OH = input["sana/i_oh"][()]
            self.sana_n_HB = input["sana/n_hb"][()]
        input.close()

    
    def diffusion(self, specie, linear=False, t_min=5e2, t_max=4e5, steps=2000, m=False, plotting=False):
        # Settings for the margins and fit method.
        er_max = 0.1  # maximum allowed error
        
        try:
            MSD_in = getattr(self, f"msd_{specie}")
        except:
            ValueError("Input variable indicates non existing specie")

        if linear is True:
            # Linear Settings
            # Settings for the margins and fit method.
            Minc = 125  # minimum number of points included in the fit
            Mmax = 250  # maximum number of points included in the fit
            
            t = self.t
            step = (t.shape[0] ) // steps
            t = t[::step]
            
            MSD_in = MSD_in
            MSD_in = MSD_in[::step]
            
            start = np.argmin(abs(t - t_min))
            end = np.argmin(abs(t - t_max))
            
            t = t[start:end]*1e-15
            MSD_in = MSD_in[start:end]
        else:
            # Log spacing (like OCTP)
            Minc = 8
            Mmax = 20
            steps = int(steps/20)
            
            n = np.logspace(np.log10(t_min), np.log10(t_max), steps, dtype=int)
            n = n[:np.where(n > self.t.shape)[0][0]]  # make very sure to not overflow time array
            MSD_in = MSD_in[n]
            t = self.t[n]
            
            start = np.argmin(abs(t - t_min))
            end = np.argmin(abs(t - t_max))
            
            t = t[start:end]*1e-15
            MSD_in = MSD_in[start:end]
        
        t_log = np.log10(t)
        MSD_log_in = np.log10(np.abs(MSD_in))
        ibest = 'failed'
        jbest = 'failed'
        mbest = 0

        for i in range(t_log.shape[0]-Minc):
            for j in range(Minc, min(Mmax, t_log.shape[0]-i)):
                if (t[i] != t[i+1]):
                    p, res, aa, aa1, aa3 = np.polyfit(t_log[i:i+j],
                                                    MSD_log_in[i:i+j], 1,
                                                    full=True)
                    mlog = p[0]
                    if (mlog > (1-er_max) and mlog < (1+er_max) and abs(mbest-1) > abs(mlog-1)):
                        mbest = mlog
                        jbest = j
                        ibest = i

        # Make sure to return NaN (not included in np.nanmean() for averaging).
        if ibest == 'failed':
            D = np.nan
            t_fit = t[0]
            fit = MSD_in[0]

        else:
            D, b = np.polyfit(t[ibest:ibest+jbest],
                            MSD_in[ibest:ibest+jbest], 1)

            # Test box size to displacement comparison.
            if np.abs(MSD_in[ibest+jbest]-MSD_in[ibest]) < m**2 and type(m) is not bool:
                print('MSD fit is smaller than simulation box',
                    MSD_in[ibest+jbest]-MSD_in[ibest], 'versus', m**2)

            t_fit = t[ibest:ibest+jbest]
            fit = D*t_fit + b

        if plotting is True:
            plt.figure('Diffusion fitting' + specie)
            plt.title(specie)
            plt.loglog(t, MSD_in, 'o', label='data')
            plt.loglog(t_fit, fit, '-.', label='fit')
            plt.grid()
            plt.legend()

        fact = (1e-20)/(6)
        return D*fact
    
    def viscosity(self, linear=False, t_min=5e3, t_max=4e5, steps=2000, m=False, plotting=False):
        # Settings for the margins and fit method.
        er_max = 0.1  # maximum allowed error
        MSD_in = self.msd_P

        if linear is True:
            # Linear Settings
            # Settings for the margins and fit method.
            Minc = 125  # minimum number of points included in the fit
            Mmax = 250  # maximum number of points included in the fit
            
            t = self.t
            step = (t.shape[0] ) // steps
            t = t[::step]
            
            MSD_in = MSD_in
            MSD_in = MSD_in[::step]
            
            start = np.argmin(abs(t - t_min))
            end = np.argmin(abs(t - t_max))
            
            t = t[start:end]*1e-15
            MSD_in = MSD_in[start:end]
        else:
            # Log spacing (like OCTP)
            Minc = 5
            Mmax = 15
            steps = int(steps/20)
            
            n = np.logspace(np.log10(t_min), np.log10(t_max), steps, dtype=int)
            n = n[:np.where(n > self.t.shape)[0][0]]  # make very sure to not overflow time array
            MSD_in = MSD_in[n]
            t = self.t[n]
            
            start = np.argmin(abs(t - t_min))
            end = np.argmin(abs(t - t_max))
            
            t = t[start:end]*1e-15
            MSD_in = MSD_in[start:end]
        
        t_log = np.log10(t)
        MSD_log_in = np.log10(np.abs(MSD_in))
        ibest = 'failed'
        jbest = 'failed'
        mbest = 0

        for i in range(t_log.shape[0]-Minc):
            for j in range(Minc, min(Mmax, t_log.shape[0]-i)):
                if (t[i] != t[i+1]):
                    p, res, aa, aa1, aa3 = np.polyfit(t_log[i:i+j],
                                                    MSD_log_in[i:i+j], 1,
                                                    full=True)
                    mlog = p[0]
                    if (mlog > (1-er_max) and mlog < (1+er_max) and abs(mbest-1) > abs(mlog-1)):
                        mbest = mlog
                        jbest = j
                        ibest = i

        # Make sure to return NaN (not included in np.nanmean() for averaging).
        if ibest == 'failed':
            D = np.nan
            t_fit = t[0]
            fit = MSD_in[0]

        else:
            D, b = np.polyfit(t[ibest:ibest+jbest],
                            MSD_in[ibest:ibest+jbest], 1)
            print(ibest, jbest)
            t_fit = t[ibest:ibest+jbest]
            fit = D*t_fit + b

        if plotting is True:
            plt.figure('Viscosity fitting')
            plt.title('viscosity')
            plt.loglog(t, MSD_in, 'o', label='data')
            plt.loglog(t_fit, fit, '-.', label='fit')
            plt.grid()
            plt.legend()

        fact = 1000*(1e-30*self.L**3)/(2*co.k*self.energy[:, 3].mean())
        return D*fact
    
    def viscosity2(self, plotting=False):
        T = self.energy[:, 3].mean()
        viscosity = self.msd_P* ( self.L**3 * 10**(-30) / (2 * co.k*T * self.t[1:] * 10**(-15)) )*1000
        if plotting is True:
            plt.figure('Viscosity')
            plt.title('viscosity')
            # plt.loglog(t, MSD_in, 'o', label='data')
            # plt.loglog(t_fit, fit, '-.', label='fit')
            plt.plot(self.t[1:], viscosity)
            plt.xlabel('time in fs')
            plt.ylabel('viscosity in cP (mPa s)')
            plt.grid()
            plt.legend()
        return viscosity
    
    def water_shape(self, force_rdf=True) -> Tuple[float, float]:
        """Retrieve the shape of the average water molecule from the rdf

        Args:
            force_rdf (bool, optional): Switch between force rdf or traditional rdf. Defaults to True=force rdf.

        Returns:
            Tuple:
            r_HO (float): average distance between oxygen and hydrogen in water molecule in Angstrom
            angle (float): average angle HOH /_  in degrees
        """
        if force_rdf is True:
            r_HO = self.rdf_F_r[self.rdf_F_HH2O[0, :].argmax()]
            r_HH = self.rdf_F_r[self.rdf_F_HH[0, :].argmax()]
        else:
            r_HO = self.rdf_r[self.rdf_HH2O[0, :].argmax()]
            r_HH = self.rdf_r[self.rdf_HH[0, :].argmax()]
        # Calculate the cosine of the angle
        cos_gamma = 1 - (r_HH**2) / (2 * r_HO**2)
        
        # Calculate the angle in radians
        gamma_rad = np.arccos(cos_gamma)
        
        # Convert the angle to degrees
        gamma_deg = np.degrees(gamma_rad)
        return r_HO, gamma_deg

        
    def cordination_N(self, specie1, specie2, force_rdf=True, F_idx=1):
        if force_rdf is True:
            try:
                rdf = getattr(self, f"rdf_F_{specie1+specie2}")
                Na = getattr(self, f"N_{specie2}")
            except:
                try:
                    rdf = getattr(self, f"rdf_F_{specie2+specie1}")
                    Na = getattr(self, f"N_{specie1}")
                except:
                    print(f"No rdf_F_{specie1+specie2} or rdf_F_{specie2+specie1}")
            
            r = self.rdf_F_r
        else:
            try:
                rdf = getattr(self, f"rdf_{specie1+specie2}")
                Na = getattr(self, f"N_{specie2}")
            except:
                try:
                    rdf = getattr(self, f"rdf_{specie2+specie1}")
                    Na = getattr(self, f"N_{specie1}")
                except:
                    print(f"No rdf_{specie1+specie2} or rdf_{specie2+specie1}")
            
            r = self.rdf_r
        
        n = (4*np.pi*Na)/(self.L**3)*integrate.cumulative_simpson((r**2)*rdf[F_idx], x=r)
        return n
    
    def sana_compare(self):
        if self.sana is False:
            ValueError("Data unavailable: sana is False.")
            
        else:
            self.sana_index_OH = np.asanyarray(self.sana_index_OH)
            self.sana_n_HB = np.asanyarray(self.sana_n_HB)
            return self.t/1000, self.index_OH, self.hbs, self.sana_index_OH, self.sana_n_HB
            # return self.t[:self.sana_index_OH.shape[0]]/1000, self.index_OH[:self.sana_index_OH.shape[0]], self.hbs[:self.sana_index_OH.shape[0], :, :], self.sana_index_OH, self.sana_n_HB
            
    
    # def react_time(self, plotting=False, bin_width=200, bin_max=5000):
    #     x = self.index_OH
    #     # ensure array
    #     x = np.asanyarray(x)
    #     if x.shape[-1] == 1:
    #         x =x.reshape(x.shape[0])
    #     else:
    #         raise ValueError('only 1D array supported')
    #     n = x.shape[0]


    #     # find run starts
    #     index_shift = np.where(x[1:] != x[:-1])[0] + 1
    #     index_shift = np.concatenate(([0], index_shift, [len(x)]))
    #     time_between = np.diff(index_shift)
        
    #     bin_edges = np.arange(0, bin_max, step=bin_width, dtype=np.float64)
    #     hist, _ = np.histogram(time_between, bins=bin_edges, density=True)
        
    #     # unbias the bins
    #     hist = hist.astype(np.float64)
    #     # hist /= bin_edges[1:] - bin_edges[:-1]
    #     bins = (bin_edges[:-1] + bin_edges[1:])/2  # get the centre of the bins

    #     if plotting is True:
    #         # set labels of plot
    #         i = re.findall(r'\d+', self.folder)[-1]
    #         string = ' run ' + i

    #         plt.figure('reaction_spacing')
    #         plt.plot(bins, hist, label=string)
            
    #         plt.figure('reaction_spacing log')
    #         plt.plot(bins, hist, label=string)
    #         plt.yscale('log')
    #         plt.xscale('linear')
    #     return bins, hist, time_between
    
    def react_time(self, n_bins: int, bin_max: int, lin=True, plotting=False):
        x = self.index_OH
        # ensure array
        x = np.asanyarray(x)
        if x.shape[-1] == 1:
            x =x.reshape(x.shape[0])
        else:
            raise ValueError('only 1D array supported')
        n = x.shape[0]

        # find run starts
        index_shift = np.where(x[1:] != x[:-1])[0] + 1
        index_shift = np.concatenate(([0], index_shift, [len(x)]))
        time_between = np.diff(index_shift)*(self.t[1]-self.t[0])
        
        if lin:
            # Use linear spaced bins
            bin_edges = np.linspace(0, bin_max, num=n_bins, dtype=np.float64)
            hist, _ = np.histogram(time_between, bins=bin_edges, density=True)
        
            # unbias the bins
            hist = hist.astype(np.float64)
        else:
            # Use log spaced bins
            bin_edges = np.logspace(np.log10(1), np.log10(bin_max/(self.t[1]-self.t[0])), num=n_bins)
            bin_edges = np.unique(np.ceil(bin_edges).astype(int))*(self.t[1]-self.t[0])
            hist, _ = np.histogram(time_between, bins=bin_edges, density=False)
            hist = hist/(bin_edges[1:] - bin_edges[:-1])
            hist = hist/time_between.shape[0]

        # Center the bins by taking the average of the outside parts
        bins = (bin_edges[:-1] + bin_edges[1:])/2  # get the centre of the bins

        if plotting is True:
            # set labels of plot
            i = re.findall(r'\d+', self.folder)[-1]
            string = ' run ' + i

            plt.figure('reaction_spacing')
            plt.plot(bins, hist, label=string)
            
            plt.figure('reaction_spacing log')
            plt.plot(bins, hist, label=string)
            plt.yscale('log')
            plt.xscale('linear')
        return bins, hist, time_between

def averages(data):
    mean = data.mean(axis=0)
    std = np.sqrt(data.var(axis=0)/(data.shape[0]-1))
    return mean, std

def diffusion_outside_class(MSD_in, t, specie, linear=False, t_min=5e2, t_max=4e5, steps=2000, m=False, plotting=False):
        # Settings for the margins and fit method.
        er_max = 0.1  # maximum allowed error

        if linear is True:
            # Linear Settings
            # Settings for the margins and fit method.
            Minc = 125  # minimum number of points included in the fit
            Mmax = 250  # maximum number of points included in the fit
            
            step = (t.shape[0] ) // steps
            t = t[::step]
            
            MSD_in = MSD_in
            MSD_in = MSD_in[::step]
            
            start = np.argmin(abs(t - t_min))
            end = np.argmin(abs(t - t_max))
            
            t = t[start:end]*1e-15
            MSD_in = MSD_in[start:end]
        else:
            # Log spacing (like OCTP)
            Minc = 8
            Mmax = 20
            steps = int(steps/20)
            
            n = np.logspace(np.log10(t_min), np.log10(t_max), steps, dtype=int)
            n = n[:np.where(n > t.shape)[0][0]]  # make very sure to not overflow time array
            MSD_in = MSD_in[n]
            t = t[n]
            
            start = np.argmin(abs(t - t_min))
            end = np.argmin(abs(t - t_max))
            
            t = t[start:end]*1e-15
            MSD_in = MSD_in[start:end]
        
        t_log = np.log10(t)
        MSD_log_in = np.log10(np.abs(MSD_in))
        ibest = 'failed'
        jbest = 'failed'
        mbest = 0

        for i in range(t_log.shape[0]-Minc):
            for j in range(Minc, min(Mmax, t_log.shape[0]-i)):
                if (t[i] != t[i+1]):
                    p, res, aa, aa1, aa3 = np.polyfit(t_log[i:i+j],
                                                    MSD_log_in[i:i+j], 1,
                                                    full=True)
                    mlog = p[0]
                    if (mlog > (1-er_max) and mlog < (1+er_max) and abs(mbest-1) > abs(mlog-1)):
                        mbest = mlog
                        jbest = j
                        ibest = i

        # Make sure to return NaN (not included in np.nanmean() for averaging).
        if ibest == 'failed':
            D = np.nan
            t_fit = t[0]
            fit = MSD_in[0]

        else:
            D, b = np.polyfit(t[ibest:ibest+jbest],
                            MSD_in[ibest:ibest+jbest], 1)

            # Test box size to displacement comparison.
            if np.abs(MSD_in[ibest+jbest]-MSD_in[ibest]) < m**2 and type(m) is not bool:
                print('MSD fit is smaller than simulation box',
                    MSD_in[ibest+jbest]-MSD_in[ibest], 'versus', m**2)

            t_fit = t[ibest:ibest+jbest]
            fit = D*t_fit + b

        if plotting is True:
            plt.figure('Diffusion fitting' + specie)
            plt.title(specie)
            plt.loglog(t, MSD_in, 'o', label='data')
            plt.loglog(t_fit, fit, '-.', label='fit')
            plt.grid()
            plt.legend()

        fact = (1e-20)/(6)
        return D*fact

# molality to molarity calculation
def statepoint_Gilliam(t, n_koh=1, n_w=110):
    # convert to kelving
    t += co.zero_Celsius
    m = n_koh*55/n_w
    
    # Set basic masses
    u_H2O = 2*1.00797 + 1*15.9994  # mass H2O
    u_KOH = 1*1.00797 + 1*15.9994 + 1*39.0983  # mass of KOH

    # Density equation from Gilliam et al (2007)
    # Datapoints from Gilliam et al converted to Kelvin
    T = np.arange(0, 71, 5, dtype=float) + co.zero_Celsius   # Table 3
    A = np.array([1001.9, 1001.0, 1000.0, 999.06, 998.15,
                  997.03, 995.75, 994.05, 992.07, 990.16,
                  988.45, 985.66, 983.20, 980.66, 977.88])  # Table 3
    spline = sp.interpolate.CubicSpline(T, A)
    a = spline(t)  # retrieve constant at temperature intended
    
    w = m*u_KOH/(55*u_H2O + m*u_KOH)
    rho = a*np.exp(0.0086*100*w)
    m = (n_w*u_H2O + n_koh*u_KOH)/(1000*co.N_A)  # check the molality

    L = 1e10*np.power(m/rho, 1/3)  # (mass/density)^(1/3)
    r = (n_w + n_koh)*rho/(m*co.N_A*1e3)
    
    M = (n_koh/co.N_A)/((L*1e-9)**3)  # mol/liter instead (conversion from particles to mol and from angstroms to dm)
    return M, L, t

def conductivity_Gilliam(T, M):
    T_K = T + co.zero_Celsius
    # Fit params, gillam et al. Table 5
    a = -2.041
    b = -0.0028
    c = 0.005332
    d = 207.2
    e = 0.001043
    f = -0.0000003

    sig = a*M + b*M**2 + c*M*T_K + d*M/T_K + e*M**3 + f*(T_K*M)**2  # S/cm
    return sig*1e2  # S/m

def viscosity_Guo(T, M):
    # Fit params, guo et al. eq 3
    a = 0.4300
    b = -0.0251
    c = 1e-4
    d = 0.1307
    e = 0.2366 # only in here for concistency, no K2CrO4 in mixture
    T = T
    eta = np.exp(a + b*T + c*T**2 + d*M + e*0)  # mPa s
    return eta*1e-3  # Pa s

def fin_size_cor(d, vis, T, L):
    xi = 2.837298
    d = d+(co.k*T*xi)/(6*np.pi*(L*1e-10)*vis)
    return d

def electric_conductivity(D_OH:float, D_K:float, T:float, L:float, N_salt:int):
    rho_N = N_salt/((L*1e-10)**3)
    sigma = (D_OH + D_K)*(co.eV**2*rho_N)/(co.k*T)
    return sigma

def line_with_errors(y_ave, y_err):
    y_min = y_ave - y_err
    y_max = y_ave + y_err
    return y_ave, y_min, y_max


def set_plot_settings(svg=False, tex=False):
    if svg:
        matplotlib.use('svg')
        plt.rcParams['savefig.transparent'] = True
    else:
        matplotlib.use('qtagg')
        plt.rcParams['figure.dpi'] = 250
    
    plt.rcParams["figure.figsize"] = (16, 5)

    # Fonts
    if tex:
        plt.rcParams['text.usetex'] = True
        plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}\usepackage{helvet}\renewcommand{\familydefault}{\sfdefault}'
        plt.rcParams['font.family'] = 'DejaVuSansMono'
        plt.rcParams['mathtext.fontset'] = 'custom'
        plt.rcParams['mathtext.rm'] = 'DejaVu Sans'
        plt.rcParams['mathtext.it'] = 'DejaVu Sans:italic'
        plt.rcParams['mathtext.bf'] = 'DejaVu Sans:bold'
        plt.rcParams['font.size'] = 14
        plt.rcParams['legend.fontsize'] = 12
        # Use siunitx for formatting units
        plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'
        plt.rcParams['text.latex.preamble'] += r'\sisetup{detect-all}'
        # and mhchem for mathematics
        plt.rcParams['text.latex.preamble'] += r'\usepackage[version=3]{mhchem}'
    else:
        plt.rcParams['text.usetex'] = False
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["svg.fonttype"] = "none"
        plt.rcParams['font.size'] = 12
        plt.rcParams['legend.fontsize'] = 12
    
    # Sizes of specific parts
    plt.rcParams['axes.grid'] = False
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.linewidth'] = 2
    
    plt.rcParams['lines.markersize'] = 10
    plt.rcParams['lines.markeredgewidth'] = 2
    plt.rcParams['lines.linewidth'] = 2
    

    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 1.0
    plt.rcParams['legend.facecolor'] = 'none'
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.shadow'] = False
    plt.rcParams['legend.labelspacing'] = 0.3
    plt.rcParams['legend.edgecolor'] = 'k'

    # X
    plt.rcParams['xtick.labelsize'] = 12
    plt.rcParams['xtick.major.pad'] = 8
    plt.rcParams['xtick.major.size'] = 5
    plt.rcParams['xtick.major.width'] = 2
    # Y
    plt.rcParams['ytick.labelsize'] = 12
    plt.rcParams['ytick.major.pad'] = 5
    plt.rcParams['ytick.major.size'] = 5
    plt.rcParams['ytick.major.width'] = 2
