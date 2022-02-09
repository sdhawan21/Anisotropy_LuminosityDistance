"""
Author: Suhail Dhawan

No "_" tag suggests that we are using Pantheon in this script to constrain the quadrupole and the dipole
 
"""

import numpy as np 
import matplotlib.pyplot as plt
import glob
import pymultinest as pmn
import sys
import os
import pandas as pd

from astropy.coordinates import SkyCoord
from utils_func import vincenty_sep, get_zcmb, dl_q0dip_h0quad
from time import time

# load SN data
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
sigma_SN = data_SN[:,np.where(varnames == "MUERR")[0][0]].astype('float32')

zCMB_SN = data_SN[:,np.where(varnames == "zCMB")[0][0]].astype('float32')
zHD_SN = data_SN[:,np.where(varnames == "zHD")[0][0]].astype('float32')

RA_SN = data_SN[:,np.where(varnames == "RA")[0][0]].astype('float32')
Dec_SN = data_SN[:,np.where(varnames == "DECL")[0][0]].astype('float32')

#define the covariance matrix as the 
c_SN = np.diag(sigma_SN**2.) + sys_SN
cinv_SN = np.linalg.inv(c_SN)

if zval == 'cmb':
    z_SN = zCMB_SN
elif zval == 'hd':
    z_SN = zHD_SN
elif zval == 'hel':
    z_SN = np.array([float(zhel_SN[zhel_SN[:,0] == i][0][2]) for i in data_SN[:,1]])

modelval = sys.argv[2]
c = 299792.458 #in km/s

fitSlopes = bool(sys.argv[3])

#this is where we should define the mb, x1, c for the arrays 
mB_SN = data_SN[:,np.where(varnames == "mB")[0][0]].astype('float32')
x1_SN = data_SN[:,np.where(varnames == "x1")[0][0]].astype('float32')
salt2c_SN = data_SN[:,np.where(varnames == "c")[0][0]].astype('float32')
muerr_SN = data_SN[:,np.where(varnames == "MUERR")[0][0]].astype('float32')
hostmass_SN = data_SN[:,np.where(varnames == "HOST_LOGMASS")[0][0]].astype('float32')


def dl_q0aniso(z, theta, ra, dec, lcmb = 264.021, bcmb = 48.523, model='const'):
    """
    Written as one massive be-all function with the (l,b) from the CMB fixed
    """    
    if model == 'const':
        h0, q0, qd, j0 = theta

    elif model == 'exp':
        h0, q0, qd, j0, S = theta

    coords = SkyCoord(ra, dec, frame='icrs', unit="deg")
    gal_coords = coords.galactic
    
    sep = vincenty_sep(gal_coords.l.deg * np.pi/180., gal_coords.b.deg * np.pi/180., lcmb * np.pi/180., bcmb * np.pi/180.)
    if model == 'const':
        q = q0 + qd * np.cos(sep) 
    elif model == 'exp':
        q = q0 + qd * np.cos(sep) * np.exp(- z / S)

    term = 1 + (1 - q) * z / 2. - (1 - q - 3*q**2. + j0)  * z ** 2. / 6.
    
    dl = (c * z ) * term/h0
    return dl

def llhood(model_param, ndim, npar):
    #fitting different quadrupole models to the data
    if modelval == 'const':
         h0, q0, qd, j0, M = [model_param[i] for i in range(5)]
         theta = [h0, q0, qd, j0]
    elif modelval == 'exp':
        h0, q0, qd, j0, M, S = [model_param[i] for i in range(6)]
        theta = [h0, q0, qd, j0, S]
    elif modelval == 'quad_exp_aniso': 
        h0, q0, j0, qd, S, M, lam1, lam2, Sq = [model_param[i] for i in range(9)]
        theta = [h0, q0, qd, j0, lam1, lam2, Sq]
    elif modelval == 'quad_exp_iso':
        h0, q0, j0, M, lam1, lam2, Sq = [model_param[i] for i in range(7)]
        theta = [h0, q0, j0, lam1, lam2, Sq]

    #define the function for the distance expression
    #for models with a quadrupole, using the quad distance expression
    if modelval == 'quad_exp_aniso' or modelval == 'quad_exp_iso':
        dl_aniso = dl_q0dip_h0quad(z_SN, theta, RA_SN, Dec_SN, model=modelval)
        mu_th = 5 * np.log10(dl_aniso) + 25.
    else:
        dl_aniso = dl_q0aniso(z_SN, theta, RA_SN, Dec_SN, model=modelval)
        mu_th = 5*np.log10(dl_aniso) + 25.
    delta1 = mu_SN - mu_th + M 
    chisq = np.dot(delta1.T, np.dot(cinv_SN, delta1))
    return -0.5*chisq

def llhood_fitSlopes(model_param, ndim, npar):
    if modelval == 'const':
         h0, q0, qd, j0, M, alpha, beta, delta = [model_param[i] for i in range(8)]
         theta = [h0, q0, qd, j0]
    elif modelval == 'exp':
        h0, q0, j0, qd, M, alpha, beta, delta, S = [model_param[i] for i in range(9)]
        theta = [h0, q0, qd, j0, S]
    #apply the Tripp relation
    mu_OBS = mB_SN + alpha * x1_SN - beta * salt2c_SN
    mu_OBS[hostmass_SN >= 10.] += delta
    dl_aniso = dl_q0aniso(z_SN, theta, RA_SN, Dec_SN, model=modelval)
    mu_th = 5*np.log10(dl_aniso) + 25.
    delta1 = mu_OBS - mu_th + M 
    chisq = np.dot(delta1.T, np.dot(cinv_SN, delta1))
    return -0.5*chisq

def prior(cube, ndim, npar):
    cube[0] = cube[0] * 50. + 50.
    cube[1] = cube[1] * 8. - 4.
    cube[2] = cube[2] * 20. - 10.
    if modelval == 'exp':
        cube[3] = cube[3] * 20. - 10.
        cube[4] = cube[4] * 4. - 2.
        cube[5] = cube[5] * 4. - 2.
    elif modelval == 'quad_exp_aniso':
        cube[3] = cube[3] * 20. - 10.
        cube[4] = cube[4] * 4. - 2.
        cube[5] = cube[5] * 4. - 2.
        cube[6] = cube[6] * 4. - 2.
        cube[7] = cube[7] * 4. - 2.
        cube[8] = cube[8] * 2. - 1.
    elif modelval == 'quad_exp_iso':
        cube[3] = cube[3] * 4. - 2.
        cube[4] = cube[4] * 4. - 2.
        cube[5] = cube[5] * 4. - 2.
        cube[6] = cube[6] * 2. - 1. 

def prior_fitSlopes(cube, ndim, npar):
    cube[0] = cube[0] * 50. + 50.
    cube[1] = cube[1] * 8. - 4.
    cube[2] = cube[2] * 20. - 10.
    cube[3] = cube[3] * 20. - 10.
    cube[4] = cube[4] * 70. - 35. 	#this is not on a mu scale but an mB scale
    cube[5] = cube[5] * 1.
    cube[6] = cube[6] * 4. 
    cube[7] = cube[7] * 0.5
    if modelval == 'exp':
        cube[8] = cube[8] * 4. - 2.

chainsdir = 'chains/'
if not os.path.exists(chainsdir):
    os.makedirs(chainsdir)

if modelval == 'const':
    if fitSlopes:
        npar = 8 
    else:
        npar = 5
elif modelval == 'exp':
    if fitSlopes:
        npar = 9
    else:
        npar = 6
elif modelval == 'quad_exp_aniso':
    npar = 9
elif modelval   == 'quad_exp_iso':
    npar = 7
nlp = int(sys.argv[4])
start = time()

if fitSlopes:
    pmn.run(llhood_fitSlopes, prior_fitSlopes, npar, verbose=True, n_live_points=nlp, outputfiles_basename='chains/q0aniso_test_'+zval+'_'+modelval+'fitSlopes'+'_lp'+str(nlp)+'-')
else:
    pmn.run(llhood, prior, npar, verbose=True, n_live_points=nlp, outputfiles_basename='chains/q0aniso_test_'+zval+'_'+modelval+'_lp'+str(nlp)+'-')
end = time()
duration = (end - start)/60.
print("it took ", duration," minutes")
