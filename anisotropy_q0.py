import numpy as np 
import matplotlib.pyplot as plt
import glob
import pymultinest as pmn
import sys
import os
import pandas as pd

from astropy.coordinates import SkyCoord
from utils_func import vincenty_sep


# load SN data
varnames = np.loadtxt('Pantheon/data_fitres/Ancillary_G10.FITRES', skiprows=6, dtype=str)[0]
data_SN = np.loadtxt('Pantheon/data_fitres/Ancillary_G10.FITRES', skiprows=7, dtype=str)
num_SN = len(data_SN)
sys_SN = np.loadtxt('Pantheon/data_fitres/sys_full_long_G10.txt', skiprows=1).reshape(num_SN, num_SN)
zhel_SN = np.loadtxt('Pantheon/lcparam_full_long_zhel.txt', dtype=str, skiprows=1)

zval = sys.argv[1]
zval_arr = ['cmb', 'hd']

#reorganise the data to fit 
print(varnames)
mu_SN = data_SN[:,np.where(varnames == "MU")[0][0]].astype('float32')
sigma_SN = data_SN[:,np.where(varnames == "MUERR")[0][0]].astype('float32')

zCMB_SN = data_SN[:,np.where(varnames == "zCMB")[0][0]].astype('float32')
zHD_SN = data_SN[:,np.where(varnames == "zHD")[0][0]].astype('float32')

RA_SN = data_SN[:,np.where(varnames == "RA")[0][0]].astype('float32')
Dec_SN = data_SN[:,np.where(varnames == "DECL")[0][0]].astype('float32')

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
    if modelval == 'const':
         h0, q0, qd, j0, M = [model_param[i] for i in range(5)]
         theta = [h0, q0, qd, j0]
    elif modelval == 'exp':
        h0, q0, qd, j0, S, M = [model_param[i] for i in range(6)]
        theta = [h0, q0, qd, j0, S]

    dl_aniso = dl_q0aniso(z_SN, theta, RA_SN, Dec_SN, model=modelval)
    mu_th = 5*np.log10(dl_aniso) + 25.
    delta1 = mu_SN - mu_th + M 
    chisq = np.dot(delta1.T, np.dot(cinv_SN, delta1))
    return -0.5*chisq

def prior(cube, ndim, npar):
    cube[0] = cube[0] * 50. + 50.
    cube[1] = cube[1] * 8. - 4.
    cube[2] = cube[2] * 20. - 10.
    cube[3] = cube[3] * 20. - 10.
    cube[4] = cube[4] * 4. - 2.
    if modelval == 'exp':
        cube[5] = cube[5] * 4. - 2.

chainsdir = 'chains/'
if not os.path.exists(chainsdir):
    os.makedirs(chainsdir)

if modelval == 'const':
    npar = 5
elif modelval == 'exp':
    npar = 6


pmn.run(llhood, prior, npar, verbose=True, n_live_points=150, outputfiles_basename='chains/q0aniso_test_'+zval+'_'+modelval+'-')
