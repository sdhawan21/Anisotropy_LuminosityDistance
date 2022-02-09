"""
Same as the previous script for fitting the dipole of q0, but with JLA
This docstring is to keep a track of the different "cases" or input toggles for the chains in the output 

- Likelhood: constrained chisq (add other options or just compare the same case of other parameters with )
- Pecvel: with covariance but can turn it off
- BiasCorr: always applied unless specially subtracted off 
- Quadrupole on / off TO BE ADDED	THE QUADRUPOLE IS A MODELVAL OPTION e.g. CONST, EXP, ISO

NOTE: the zCMB includes a PV correction, do the boost yourself
Include the model with isotropic case as well


ATTENTION: also runs with a fixed alpha and beta

Redshift list
- zCMB -> I compute this by hand using l,b from Kogut+1996
- zHel
- zPV -> this is the "zcmb" in the JLA file

Model list
- const
- exponential
- isotropic
and two with quadrupole
- quad_aniso --> exp for q_d
- quad_iso --> iso for q0



Usage 

python anisotropy_q0_JLA.py <zval> <modelval> <pvcov> <biascor> 
e.g.
python anisotropy_q0_JLA.py cmb exp "" "" yes 
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

# load SN data
data_SN = pd.read_csv('JLA/jla_likelihood_v4/data/jla_lcparams.txt', delim_whitespace=True, header=None, usecols=(1,2,4,5,6,7,8,9,10,14,15,16,18,19,20), names=['zcmb', 'zhel', 'mb','dmb','x1','dx1','color','dcolor','mhost', 'cov_m_s', 'cov_m_c' ,'cov_s_c', 'ra', 'dec', 'biascor'], skiprows=1)

num_SN = len(data_SN)

zval = sys.argv[1]
modelval = sys.argv[2]
#zval_arr = ['cmb', 'hd']  #forget it, its got a whole bunch of different notations for the various redshifts 

#reorganise the data to fit 
#get the arrays separately
mb_SN = data_SN['mb']
x1_SN = data_SN['x1']
c_SN = data_SN['color']
hostmass_SN = data_SN['mhost']

#compute the boost to the CMB frame instead of using the zcmb in the JLA files since that will likely include a
# peculiar velocity correction which we cant test for
zCMB_SN = get_zcmb(data_SN['zhel'].values, data_SN['ra'].values, data_SN['dec'].values)
#print(zCMB_SN)
zPV_SN = data_SN['zcmb']
zHEL_SN = data_SN['zhel']
RA_SN = data_SN['ra']
Dec_SN = data_SN['dec']

if zval == 'cmb':
    z_SN = np.array(zCMB_SN)
elif zval == 'hel':
    z_SN = np.array(zHEL_SN)
elif zval == 'pvcorr':
    z_SN = np.array(zPV_SN)

pecvelCov = bool(sys.argv[3])
if pecvelCov:
    covmat_ls = glob.glob('JLA/jla_likelihood_v4/C*.fits')
    pv_str = 'withPVCovSigZ'
    pec_sig = True
    print("With peculiar velocity errors")

else:
    covmat_ls = glob.glob('JLA/jla_likelihood_v4/C*.fits')
    covmat_ls = [i for i in covmat_ls if 'pecvel' not in i]
    pv_str = 'NoPVCovNoSigZ'
    pec_sig = False
    print("No peculiar velocity errors")

biasCor = bool(sys.argv[4])



if biasCor:
    data_SN['mb'] -= data_SN['biascor']
    bstr='BiasSub'
    
else:
    bstr='FidNoBiasSub'


fixSlopes = bool(sys.argv[5])


def mu_cov(alpha, beta, sig_pecvel=True):
    """ Assemble the full covariance matrix of distance modulus

    See Betoule et al. (2014), Eq. 11-13 for reference
    """
    Ceta = sum([fits.getdata(mat) for mat in covmat_ls])
    
    Cmu = np.zeros_like(Ceta[::3,::3])
    for i, coef1 in enumerate([1., alpha, -beta]):
        for j, coef2 in enumerate([1., alpha, -beta]):
            Cmu += (coef1 * coef2) * Ceta[i::3,j::3]

    # Add diagonal term from Eq. 13
    sigma = np.loadtxt('JLA/jla_likelihood_v4/sigma_mu.txt')
    if sig_pecvel:
        sigma_pecvel = (5 * 150 / 3e5) / (np.log(10.) * sigma[:, 2])
    else:
        sigma_pecvel = 0
    Cmu[np.diag_indices_from(Cmu)] += sigma[:, 0] ** 2 + sigma[:, 1] ** 2 + sigma_pecvel ** 2
    
    return Cmu

#if im fitting with alpha, beta and delta fixed for a single case, this is where i should define that 
#define a separate likelihood for this case
alpha = 0.14
beta = 3.1
delta = 0.07
C = mu_cov(alpha, beta, sig_pecvel=pec_sig)
cinv_SN = np.linalg.inv(C)
print(np.shape(cinv_SN))
#sys.exit()


#have a likelihood with all models as options
def llhood(model_param, ndim, npar):
    if modelval == 'const':
            h0, q0, qd, j0, M, alpha, beta, dmass = [model_param[i] for i in range(8)]
         theta = [h0, q0, qd, j0]
    elif modelval == 'exp':
        h0, q0, qd, j0, M, alpha, beta, dmass, S  = [model_param[i] for i in range(9)]
        theta = [h0, q0, qd, j0, S]
    elif modelval == 'iso':
        h0, q0, j0, M, alpha, beta, dmass  = [model_param[i] for i in range(7)]
        theta = [h0, q0, j0]
    elif modelval == 'quad_aniso':
        h0, q0, qd, j0, M, alpha, beta, dmass, S, lam1, lam2  = [model_param[i] for i in range(11)]
        theta = [h0, q0, qd, j0, S, lam1, lam2]
    elif modelval == 'quad_iso': 
        h0, q0, j0, M, alpha, beta, dmass, lam1, lam2  = [model_param[i] for i in range(9)]
        theta = [h0, q0,  j0, lam1, lam2]
    elif modelval == 'quad_exp_iso':
        h0, q0, j0, M, alpha, beta, dmass, lam1, lam2, S  = [model_param[i] for i in range(10)]
        theta = [h0, q0,  j0, lam1, lam2, S]
    elif modelval == 'quad_exp_aniso':
        h0, q0, qd, j0, M, alpha, beta, dmass, lam1, lam2, Sd, Sq  = [model_param[i] for i in range(12)]
        theta = [h0, q0, qd, j0, lam1, lam2, Sd, Sq]
 
    C = mu_cov(alpha, beta, sig_pecvel=pec_sig)
    cinv_SN = np.linalg.inv(C)

    mu_SN = mb_SN + alpha * x1_SN - beta * c_SN 
    mu_SN[hostmass_SN >= 10.] += dmass
    #this part about fitting the quadrupole has modelval twice which is a little superfluous
    if "quad" in modelval:
        dl_aniso = dl_q0dip_h0quad(z_SN, theta, RA_SN, Dec_SN, model=modelval)
    else:
        dl_aniso = dl_q0aniso(z_SN, theta, RA_SN, Dec_SN, model=modelval)
    mu_th = 5 * np.log10(dl_aniso) + 25.
    delta1 = mu_SN - mu_th + M 
    chisq = np.dot(delta1.T, np.dot(cinv_SN, delta1))
    return -0.5*chisq

dmass = delta 
mu_SN = mb_SN + alpha * x1_SN - beta * c_SN 
mu_SN[hostmass_SN >= 10.] += delta
#have a likelihood with all models as options
def llhood_fixed(model_param, ndim, npar):
    if modelval == 'const':
         h0, q0, qd, j0, M = [model_param[i] for i in range(5)]
         theta = [h0, q0, qd, j0]
    elif modelval == 'exp':
        h0, q0, qd, j0, M, S  = [model_param[i] for i in range(6)]
        theta = [h0, q0, qd, j0, S]
    elif modelval == 'iso':
        h0, q0, j0, M  = [model_param[i] for i in range(4)]
        theta = [h0, q0, j0]
    elif modelval == 'quad_aniso':
        h0, q0, qd, j0, M, S, lam1, lam2  = [model_param[i] for i in range(8)]
        theta = [h0, q0, qd, j0, S, lam1, lam2]
    elif modelval == 'quad_iso': 
        h0, q0, j0, M, lam1, lam2  = [model_param[i] for i in range(6)]
        theta = [h0, q0,  j0, lam1, lam2]
    elif modelval == 'quad_exp_iso':
        h0, q0, j0, M,  lam1, lam2, S  = [model_param[i] for i in range(7)]
        theta = [h0, q0,  j0, lam1, lam2, S]
    elif modelval == 'quad_exp_aniso':
        h0, q0, qd, j0, M, lam1, lam2, Sd, Sq  = [model_param[i] for i in range(9)]
        theta = [h0, q0, qd, j0, lam1, lam2, Sd, Sq]
 
    

    

    #this part about fitting the quadrupole has modelval twice which is a little superfluous
    #not anymore?
    if "quad" in modelval:
        dl_aniso = dl_q0dip_h0quad(z_SN, theta, RA_SN, Dec_SN, model=modelval)
    else:
        dl_aniso = dl_q0aniso(z_SN, theta, RA_SN, Dec_SN, model=modelval)
    mu_th = 5 * np.log10(dl_aniso) + 25.
    delta1 = mu_SN - mu_th + M 
    chisq = np.dot(delta1.T, np.dot(cinv_SN, delta1))
    return -0.5*chisq

def prior(cube, ndim, npar):
    cube[0] = cube[0] * 50. + 50.
    cube[1] = cube[1] * 8. - 4.	#q0
    cube[2] = cube[2] * 20. - 10. #qd
    cube[3] = cube[3] * 20. - 10. #j0
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
    elif modelval == "quad_exp_aniso":
        cube[8] = cube[8] * 4. - 2.
        cube[9] = cube[9] * 4. - 2.
        cube[10] = cube[10] * 4. - 2.
        cube[11] = cube[11] * 4. - 2.
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

def prior_fixslopes(cube, ndim, npar):
    cube[0] = cube[0] * 50. + 50.
    cube[1] = cube[1] * 8. - 4. #q0
    cube[2] = cube[2] * 20. - 10. #qd
    cube[3] = cube[3] * 20. - 10. #j0
    cube[4] = cube[4] * 70. - 35. 
    if modelval == 'exp':
        cube[5] = cube[5] * 4. - 2.
    elif modelval == "quad_aniso":
        cube[5] = cube[5] * 4. - 2.
        cube[6] = cube[6] * 4. - 2.
        cube[7] = cube[7] * 4.# - 2.
    elif modelval == "quad_exp_aniso":
        cube[5] = cube[5] * 4. - 2.
        cube[6] = cube[6] * 4. - 2.
        cube[7] = cube[7] * 4. - 2.
        cube[8] = cube[8] * 4.# - 2.

def prior_exp_iso(cube, ndim, npar):
    cube[0] = cube[0] * 50. + 50.
    cube[1] = cube[1] * 8. - 4. #q0
    cube[2] = cube[2] * 20. - 10. #j0
    cube[3] = cube[3] * 70. - 35. 
    cube[4] = cube[4] * 4. - 2.
    cube[5] = cube[5] * 4. - 2. 
    cube[6] = cube[6] * 4.# - 2.

chainsdir = 'chains/'
if not os.path.exists(chainsdir):
    os.makedirs(chainsdir)

#a bit wordy but this is where I define the number of parameters and the hypercube for the parameter priors to be passed
#to the PyMultiNest run function
if modelval == 'const':
    npar = 5
    prior_cube = prior 
elif modelval == 'exp':
    npar = 6
    prior_cube = prior
elif modelval == 'iso':
    npar = 4
    prior_cube = prior_iso

elif modelval == 'quad_aniso':
    npar = 8
    prior_cube = prior 

elif modelval == 'quad_iso':
    npar = 6
    prior_cube = prior_iso

elif modelval == 'quad_exp_iso':
    npar = 7
    prior_cube = prior_exp_iso
elif modelval == 'quad_exp_aniso':
	npar = 9
	prior_cube = prior 

nlp = int(sys.argv[6])
t1 = time()

if fixSlopes:
    fix_str = "fixedSlope"
    if modelval == "quad_exp_iso":
        pmn.run(llhood_fixed, prior_cube, npar, verbose=True, n_live_points=nlp, outputfiles_basename='chains/q0aniso_test_JLA_'+zval+'_'+modelval+'_'+pv_str+'_'+bstr+'_'+fix_str+'_lp'+str(nlp)+'-')
    else:
        pmn.run(llhood_fixed, prior_fixslopes, npar, verbose=True, n_live_points=nlp, outputfiles_basename='chains/q0aniso_test_JLA_'+zval+'_'+modelval+'_'+pv_str+'_'+bstr+'_'+fix_str+'_lp'+str(nlp)+'-')
else:
    fix_str = "freeSlope"
    npar += 3
    pmn.run(llhood, prior_cube, npar, verbose=True, n_live_points=nlp, outputfiles_basename='chains/q0aniso_test_JLA_'+zval+'_'+modelval+'_'+pv_str+'_'+bstr+'_'+fix_str+'_lp'+str(nlp)+'-')
t2 = time()
duration = (t2 - t1) / 60.
print("It took ", duration, " minutes")

