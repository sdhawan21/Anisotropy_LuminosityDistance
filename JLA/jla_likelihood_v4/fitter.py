#Using the complete JLA to fit standard cosmology
import numpy as np
import pandas as pd
import pymultinest as pmn
import glob
import sys

from astropy.io import fits
from astropy.cosmology import LambdaCDM
from scipy.integrate import quad

from numpy import median
from setup_comp import * 
from compress_params import ratio_mat, ratio_covariance, ratio_covariance_analytic, mk_icov_cmb
from bao_prior import zarr_pl, dzarr_pl, e_dzarr_pl, icov_pl
from sn_data import *

jla_lcpar = pd.read_csv('jla_lcparams.txt', skiprows=1, usecols=(1, 2,4,6,8,10), names=['zcmb', 'zhel', 'mb', 'x1', 'c', 'mhost'], delim_whitespace=True, header=None)

jla_lowM = jla_lcpar[jla_lcpar.mhost < 10.]
jla_highM = jla_lcpar[jla_lcpar.mhost >= 10.]

c = 2.99e5

cmb = bool(sys.argv[1])
bao = bool(sys.argv[2])
cmbao = bool(sys.argv[3])
localh0 = bool(sys.argv[4])


def bao_mat(om, ok, h0, zarr = np.array([.106, .35, .57]), icov= np.diag([4444., 215156., 721487.]), dz_arr = np.array([.336, .1126, .07315])):
    dvec = dz_bao(om, ok, h0, zarr = zarr)
    dif = dvec - dz_arr
    chi2 = np.dot(dif.T, np.dot(icov, dif))
    return chi2 

def cmb_mat(om, ok, h0, rval=1.7382, lval = 301.63):
    #val_vec = [1.7382, 301.63]
    inv_cmb = np.linalg.inv(mk_icov_cmb())
    rth = R_shift_lcdm(om, ok, h0)
    lth = la(om, ok, h0)
    val_vec = np.array([rth, lth])
    d_arr = np.array([rval, lval]) - val_vec 
    cmb_chisq = np.dot(d_arr.T, np.dot(inv_cmb, d_arr))
    return cmb_chisq 

def cmbao_chisq_mat(om, ok, h0, zarr=np.array([.106, .35,.57]), dz_arr=dzarr_pl,e_dzarr = e_dzarr_pl):
    rat_arr = ratio_mat(dz_arr=dz_arr)
#    rat_cov = ratio_covariance()
    rat_cov = ratio_covariance_analytic(dz_arr=dz_arr, e_dzarr = e_dzarr)
    icov = np.linalg.inv(rat_cov)
    rth = [cmbao_ratio(om, ok, h0, z=i)[0] for i in zarr]
    dif = rth -rat_arr
    chisq = np.dot(dif.T, np.dot(np.linalg.inv(rat_cov), dif))
    return chisq


def ez(z,om,ol):
    a = om*(1+z)**3 + ol
    return 1/np.sqrt(a)

def ez_int(zhelarr, zcmbarr, om, ol, h0):
    ezarr = [quad(ez, 0, i, args=(om, ol))[0] for i in zcmbarr]
    return 5*np.log10(c*(1+zhelarr)*np.array(ezarr)/h0)+25.


def llhood_cmb_bao(model_param, ndim, nparam):
    h0, om, ok = [model_param[i] for i in xrange(3)]
    cmb_chisq = cmb_mat(om, ok, h0)
    bao_chisq = bao_mat(om, ok, h0)
    chisq = cmb_chisq + bao_chisq
    return -0.5 * chisq - 0.5*((h0-73.24)**2/1.74**2)

def llhood_compress(model_param, ndim, nparam):
    h0, om = [model_param[i] for i in xrange(2)]
    #om = 10**logom
    ok = 0.	#flatness 
    #om = 1. 	#einstein de-sitter universe
    model = curv_lumdist_lcdm(bdist.zb.values,bdist.zb.values,om, ok, ok, h0)#5*np.log10(lc.luminosity_distance(bdist.zb.values).value)+25
    dif = model - bdist.mub.values
    chisq = np.dot(dif.T, np.dot(inv_covmat, dif))

    if cmb:
	cc = cmb_mat(om, ok, h0) 
	chisq+=cc

    if bao:
	cbao = bao_mat(om, ok, h0, zarr=zarr_pl, dz_arr=dzarr_pl, icov = icov_pl)
    	chisq+=cbao

    if cmbao:
	cchi = cmbao_chisq_mat(om, ok, h0, zarr=zarr_pl)
        chisq += cchi
   
    if localh0:
	chi2 = (h0 - 73.24)**2/1.74**2
        chisq += chi2
    return -0.5*chisq 

def llhood_complete(model_param, ndim, nparam):
    om, ok,al, bet, Mb, dMass, h0 = [model_param[i] for i in xrange(7)]
    zhel = jla_lcpar.zhel.values

    #for i, zval in jla_lcpar.zcmb.values: 
    #thval_low = ez_int(jla_lowM.zhel.values,jla_lowM.zcmb.values, om, ol)
    #delfun_low = jla_lowM.mb.values + al*jla_lowM.x1.values[i] - bet*jla_lowM.c.values - Mb - thval_low

    #thval_high = ez_int(jla_highM.zhel.values, jla_highM.zcmb.values, om, ol)
    #delfun_high = jla_highM.mb.values + al*jla_highM.x1.values[i] - bet*jla_highM.c.values - Mb - thval_high
    #thval = ez_int(jla_lcpar.zhel.values, jla_lcpar.zcmb.values, om, ol, h0)
   
    delfun_nohost = jla_lcpar.mb.values + al*jla_lcpar.x1.values - bet*jla_lcpar.c.values - Mb - curv_lumdist_lcdm(jla_lcpar.zhel.values, jla_lcpar.zcmb.values, om, ok, ok, h0)

    delfun_nohost[jla_lcpar.mhost.values >=10.]+=dMass    
    #print np.std(delfun_nohost), delfun_nohost
    C = mu_cov(al, bet)
    inv_cov = np.linalg.inv(C)
    chisq = np.dot(delfun_nohost.T, np.dot(inv_cov, delfun_nohost))
    if cmb:
	cc = cmb_mat(om, ok, h0) 
	chisq+=cc
    if bao:
	cbao = bao_mat(om, ok, h0, zarr=zarr_pl, dz_arr=dzarr_pl, icov = icov_pl)
    	chisq+=cbao
    if cmbao:
	cchi = cmbao_chisq_mat(om, ok, h0, zarr=zarr_pl)
        chisq += cchi
   
    if localh0:
	chi2 = (h0 - 73.24)**2/1.74**2
        chisq += chi2
    return -0.5*chisq

def llhood_completeflat(model_param, ndim, nparam):
    om, al, bet, Mb, dMass, h0 = [model_param[i] for i in xrange(6)]
    zhel = jla_lcpar.zhel.values
    ok = 0.
    #om = 1.
    delfun_nohost = jla_lcpar.mb.values + al*jla_lcpar.x1.values - bet*jla_lcpar.c.values - Mb - curv_lumdist_lcdm(jla_lcpar.zhel.values, jla_lcpar.zcmb.values, om, ok, ok, h0)

    delfun_nohost[jla_lcpar.mhost.values >=10.]+=dMass    
    #print np.std(delfun_nohost), delfun_nohost
    C = mu_cov(al, bet)
    inv_cov = np.linalg.inv(C)
    chisq = np.dot(delfun_nohost.T, np.dot(inv_cov, delfun_nohost))
    if cmb:
	cc = cmb_mat(om, ok, h0) 
	chisq+=cc
    if bao:
	cbao = bao_mat(om, ok, h0, zarr=zarr_pl, dz_arr=dzarr_pl, icov = icov_pl)
    	chisq+=cbao
    if cmbao:
	cchi = cmbao_chisq_mat(om, ok, h0, zarr=zarr_pl)
        chisq += cchi
   
    return -0.5*chisq


def prior_complete(cube, ndim, nparam):
    cube[0] = cube[0]*.5
    cube[1] = cube[1]*100. - 50.   #switch from the Olam to Ok parametrisation
    cube[2] = cube[2]*1.
    cube[3] = cube[3]*4.
    cube[4] = cube[4]*20. - 35.
    cube[5] = cube[5]*0.2
    cube[6] = cube[6]*50. + 50.


def prior_completeflat(cube, ndim, nparam):
    cube[0] = cube[0]*.5   #switch from the Olam to Ok parametrisation
    cube[1] = cube[1]*1.
    cube[2] = cube[2]*4.
    cube[3] = cube[3]*20. - 35.
    cube[4] = cube[4]*0.2
    cube[5] = cube[5]*50. + 50.

def prior_compress(cube, ndim, nparam):
    cube[0] = cube[0]*50. + 50.
    cube[1] = cube[1]*0.5

def prior_curv(cube, ndim, nparam):
    cube[0] = cube[0]*50. + 50.
    cube[1] = cube[1]*.5
    cube[2] = cube[2]*2. - 1.


#
#print llhood_completeflat([.3,  .8, 3.1, -19.02, .07, 74], 6, 6)
#
#pmn.run(llhood_compress, prior_compress, 2, verbose=True, n_live_points=400)
pmn.run(llhood_cmb_bao, prior_curv, 3, verbose=True)
#pmn.run(llhood_complete, prior_complete, 7, verbose=True, n_live_points=150)
#pmn.run(llhood_completeflat, prior_completeflat, 6, verbose=True, n_live_points=400, sampling_efficiency=0.3)

