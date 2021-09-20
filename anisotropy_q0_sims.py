"""
Fits that are very similar to the _q0_JLA script that are applied for the forecasts for quadrupole fits
"""

import numpy as np 
import matplotlib.pyplot as plt
import glob
import pymultinest as pmn
import sys
import os
import pandas as pd

from astropy.coordinates import SkyCoord
from astropy.io import fits
from utils_func import vincenty_sep, dl_q0aniso, get_zcmb,dl_q0dip_h0quad
from time import time
from astropy.cosmology import FlatLambdaCDM




np.random.seed(1729) #did I have to do this just because of the Ramanujan story  
#the actual omega_M and h0 shouldnt matter since for the anisotropic case, we'll use "the same" (i.e. equivalent for the monopole)
flc = FlatLambdaCDM(70., 0.3)

# load SN data along with the fitres files for the associated variables
varnames = np.loadtxt('Pantheon/data_fitres/Ancillary_G10.FITRES', skiprows=6, dtype=str)[0]
data_SN = np.loadtxt('Pantheon/data_fitres/Ancillary_G10.FITRES', skiprows=7, dtype=str)
num_SN = len(data_SN)
sys_SN = np.loadtxt('Pantheon/data_fitres/sys_full_long_G10.txt', skiprows=1).reshape(num_SN, num_SN)
zhel_SN = np.loadtxt('Pantheon/lcparam_full_long_zhel.txt', dtype=str, skiprows=1)

zval = sys.argv[1]
zval_arr = ['cmb', 'hd', 'hel']

#reorganise the data to fit 
print(varnames)
mu_SN = data_SN[:,np.where(varnames == "MU")[0][0]].astype('float32')
muerr_hd = data_SN[:,np.where(varnames == "MUERR")[0][0]].astype('float32')

zCMB_SN = data_SN[:,np.where(varnames == "zCMB")[0][0]].astype('float32')
zhd = data_SN[:,np.where(varnames == "zHD")[0][0]].astype('float32')

RA_SN = data_SN[:,np.where(varnames == "RA")[0][0]].astype('float32')
Dec_SN = data_SN[:,np.where(varnames == "DECL")[0][0]].astype('float32')

nfac = int(sys.argv[1])
uniform_coord = bool(sys.argv[2])
quad_input = bool(sys.argv[3])


if uniform_coord:
    RA_SN = np.random.uniform(0, 360, len(zhd))
    Dec_SN = np.random.uniform(-90, 90, len(zhd))
 



#define the arrays for the concatenation and fill the first NSN values with the original data
zlow = zhd[zhd < 0.1]

zsim = np.zeros(len(zhd) + nfac * len(zlow))
rasim = np.zeros(len(zhd) + nfac * len(zlow))
decsim = np.zeros(len(zhd) + nfac * len(zlow))
muerr_sim = np.zeros(len(zhd) + nfac * len(zlow))

zsim[:len(zhd)] = zhd
rasim[:len(zhd)] = RA_SN
decsim[:len(zhd)] = Dec_SN
muerr_sim[:len(zhd)] = muerr_hd

#how many SNe to augment the low-z anchor by
for i in range(nfac):
    zsim[len(zhd) + i*len(zlow): len(zhd) + (i+1) * len(zlow)] = zlow
    rasim[len(zhd) + i*len(zlow): len(zhd) + (i+1) * len(zlow)] = RA_SN[zhd < 0.1]
    decsim[len(zhd) + i*len(zlow): len(zhd) + (i+1) * len(zlow)] = Dec_SN[zhd < 0.1]
    muerr_sim[len(zhd) + i*len(zlow): len(zhd) + (i+1) * len(zlow)] = muerr_hd[zhd < 0.1]

#muerr_hd[zhd < 0.1] /= np.sqrt(nfac)
print(len(zhd[zhd < 0.1]))

#define the simulated distances for the zsim array. this is with an LCDM cosmology
mu_synth = flc.distmod(zsim).value
mu_synth += np.random.normal(0., 0.15, len(zsim))

#fix to the PV corrected redshift 
z_SN = zsim
stat_only = True #this needs to be changed with the adequate systematics error covariance (use Pantheon 1? 2? some forecast?)

modelval = 'quad_exp_iso'

#take one sensible value of lambda1, lambda2
mu_quad = 5 * np.log10(dl_q0dip_h0quad(z_SN, [70., -0.55, 0.4, 0.12, 0.12, 0.03], rasim, decsim, model=modelval)) + 25
print(len(zsim), len(zhd))

if quad_input:
    mu_sims = mu_quad
else:
    mu_sims = mu_synth

def llhood(model_param, npar, ndim):
    if modelval == 'quad_exp_iso':
        h0, q0, j0, M, alpha, beta, dmass, lam1, lam2, S  = [model_param[i] for i in range(10)]
        theta = [h0, q0,  j0, lam1, lam2, S]


    C = np.diag(muerr_hd ** 2.)
    cinv_SN = np.linalg.inv(C) 
    dl_aniso = dl_q0dip_h0quad(z_SN, theta, rasim, decsim, model=modelval)

    mu_th = 5 * np.log10(dl_aniso) + 25.
    delta1 = mu_synth - mu_th + M 
    chisq = np.dot(delta1.T, np.dot(cinv_SN, delta1))
    return -0.5*chisq

#prior hypercube is determined the exact same way as for the data
def prior(cube, ndim, npar):
    cube[0] = cube[0] * 50. + 50.
    cube[1] = cube[1] * 8. - 4.
    cube[2] = cube[2] * 20. - 10.
    cube[3] = cube[3] * 20. - 10.
    cube[4] = cube[4] * 70. - 35.
    cube[5] = cube[5] * 1.
    cube[6] = cube[6] * 4.
    cube[7] = cube[7] * 0.5
    if modelval == 'exp':
        cube[8] = cube[8] * 4. - 2.
    elif modelval == "quad_aniso":
        cube[8] = cube[8] * 4. - 2.
        cube[9] = cube[9] * 4. - 2.
        cube[10] = cube[10] * 4. - 2.

def prior_iso(cube, ndim, npar):
    cube[0] = cube[0] * 50. + 50.
    cube[1] = cube[1] * 8. - 4.
    cube[2] = cube[2] * 20. - 10.
    cube[3] = cube[3] * 70. - 35. 
    cube[4] = cube[4] * 1. 
    cube[5] = cube[5] * 4. 
    cube[6] = cube[6] * 0.5
    if modelval == "quad_iso":
        cube[7] = cube[7] * 4. - 2.
        cube[8] = cube[8] * 4. - 2.
    if modelval == "quad_exp_iso":
        cube[7] = cube[7] * 4. - 2.
        cube[8] = cube[8] * 4. - 2.
        cube[9] = cube[9] * 2. - 1.

chainsdir = 'chains/'
if not os.path.exists(chainsdir):
    os.makedirs(chainsdir)

#a bit wordy but this is where I define the number of parameters and the hypercube for the parameter priors to be passed
#to the PyMultiNest run function
if modelval == 'const':
    npar = 8
    prior_cube = prior 
elif modelval == 'exp':
    npar = 9
    prior_cube = prior
elif modelval == 'iso':
    npar = 7
    prior_cube = prior_iso

elif modelval == 'quad_aniso':
    npar = 11
    prior_cube = prior 

elif modelval == 'quad_iso':
    npar = 9
    prior_cube = prior_iso

elif modelval == 'quad_exp_iso':
    npar = 10
    prior_cube = prior_iso

if stat_only:
    cov_str = 'stat_only'
else:
    cov_str = 'stat_sys'
    
if uniform_coord:
    coordval = "uniformCoord"
else:
    coordval  = "trueCoord"

if quad_input:
    inpval = "NonZeroQuad"
else:
    inpval = "LCDM"
nlp = 200
t1 = time()
pmn.run(llhood, prior_cube, npar, verbose=True, n_live_points=nlp, outputfiles_basename='chains/sims/q0aniso_sims_'+modelval+'_'+coordval+'_'+inpval+'_'+str(nfac)+"_"+cov_str+'_lp'+str(nlp)+'-')
t2 = time()
duration = (t2 - t1) / 60.
print("It took ", duration, " minutes")


