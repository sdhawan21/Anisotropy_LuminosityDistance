import numpy as np 
import matplotlib.pyplot as plt
import pymultinest as pmn
import glob
import sys
import os
import astropy.units as u
import sfdmap  #import the dustmaps to cross-check the MW EBV (e.g from the fitres file Dillon sent)


from astropy.cosmology import FlatLambdaCDM
from astropy.coordinates import SkyCoord

c = 299792.458 #speed of light in km/s

def vincenty_sep(lon1, lat1, lon2, lat2):
    """
    Get the on sky separation using Vincenty's formula for the SN - CMB distance
    Transcribed from the astropy formula. Also available at:

    https://en.wikipedia.org/wiki/Great-circle_distance
    """
    sdlon = np.sin(lon2 - lon1)
    cdlon = np.cos(lon2 - lon1)
    slat1 = np.sin(lat1)
    slat2 = np.sin(lat2)
    clat1 = np.cos(lat1)
    clat2 = np.cos(lat2)

    num1 = clat2 * sdlon
    num2 = clat1 * slat2 - slat1 * clat2 * cdlon

    denominator = slat1 * slat2 + clat1 * clat2 * cdlon
    return np.arctan2(np.hypot(num1, num2), denominator)

def get_zcmb(zhel, ra, dec, lcmb = 264.021, bcmb = 48.523, vcmb= 369.82):
    v_hel = zhel*299792.458
    #convert to galactic coordinates 
    coords = SkyCoord(ra, dec, frame='icrs', unit='deg')
    gal_coords = coords.galactic
    sep = vincenty_sep(gal_coords.l.deg * np.pi/180., gal_coords.b.deg * np.pi/180., lcmb * np.pi/180., bcmb * np.pi/180.)
    v_corr = vcmb * np.cos(sep)
    vcmb = v_hel + v_corr
    return vcmb/299792.458

def dl_q0aniso(z, theta, ra, dec, lcmb = 264.021, bcmb = 48.523,   model='const'):
    """
    Written as one massive be-all function with the (l,b) from the CMB fixed
    """    
    if model == 'const':
        h0, q0, qd, j0 = theta

    elif model == 'exp':
        h0, q0, qd, j0, S = theta
    elif model == 'iso':
        h0, q0, j0 = theta

    coords = SkyCoord(ra, dec, frame='icrs', unit="deg")
    gal_coords = coords.galactic
    
    sep = vincenty_sep(gal_coords.l.deg * np.pi/180., gal_coords.b.deg * np.pi/180., lcmb * np.pi/180., bcmb * np.pi/180.)
    if model == 'const':
        q = q0 + qd * np.cos(sep) 
    elif model == 'exp':
        q = q0 + qd * np.cos(sep) * np.exp(- z / S)
    elif model == 'iso':
        q = q0

    term = 1 + (1 - q) * z / 2. - (1 - q - 3*q**2. + j0)  * z ** 2. / 6.
    
    dl = (c * z ) * term/h0
    return dl

def dl_q0dip_h0quad(z, theta, ra, dec, lcmb = 264.021, bcmb = 48.523,  coord1=(118,85), coord2=(341,+4), coord3=(71,-4), model='const'):
    """
    Explicit multiplication of the quadrupole in H0 as a "pre-factor" to the DL expansion 
    In two separate models, i.e. with and without the dipole in q0.
    These models are named "quad_aniso" , "quad_iso" respectively
    
    for "quad_iso" it follows the same as the iso model in the dipole only case
    this includes "quad_exp_iso" where the quadrupole has an exponentially decaying scale but the dipole is isotropic
    "quad_exp_aniso" is with both the quadrupolar and dipolar anisotropy
    """ 
    if model == 'const':
        h0, q0, qd, j0 = theta
    elif model == 'exp':
        h0, q0, qd, j0, S = theta
    elif model == 'iso':
        h0, q0, j0 = theta
    elif model == 'quad_aniso':
        h0, q0, qd, j0, S, lam1, lam2 = theta
    elif model == 'quad_iso':
        h0, q0, j0, lam1, lam2 = theta
    elif model == 'quad_exp_iso': 
        h0, q0, j0, lam1, lam2, Sq = theta
    elif model == 'quad_exp_aniso':
         h0, q0, qd, j0, S, lam1, lam2, Sq = theta

    coords = SkyCoord(ra, dec, frame='icrs', unit="deg")
    gal_coords = coords.galactic

    sep = vincenty_sep(gal_coords.l.deg * np.pi/180., gal_coords.b.deg * np.pi/180., lcmb * np.pi/180., bcmb * np.pi/180.)

    #along with the separation from the CMB, also compute the three separations to the eigendirections for the quadrupole
    #for the directions see Parnovsky + Parnovskii 2012: https://arxiv.org/pdf/1210.0895.pdf below equation 17
    sep1 = vincenty_sep(gal_coords.l.deg * np.pi/180., gal_coords.b.deg * np.pi/180., coord1[0] * np.pi/180., coord1[1] * np.pi/180.)
    sep2 = vincenty_sep(gal_coords.l.deg * np.pi/180., gal_coords.b.deg * np.pi/180., coord2[0] * np.pi/180., coord2[1] * np.pi/180.)
    sep3 = vincenty_sep(gal_coords.l.deg * np.pi/180., gal_coords.b.deg * np.pi/180., coord3[0] * np.pi/180., coord3[1] * np.pi/180.)
    
    #the quadrupole term is just a pre-factor so I wrote it out here as a separate variable
    if model == 'quad_exp_iso' or model == 'quad_exp_aniso' :
        F = np.exp(-z / Sq) 
    else:
        F = 1.
    pre_fac_quad = pow(1 + (lam1 * (np.cos(sep1) ** 2.) + lam2 * (np.cos(sep2) ** 2.) - (lam1 + lam2) * (np.cos(sep3) ** 2.)) * F , -1)

    if model == 'const':
        q = q0 + qd * np.cos(sep) 
    elif model == 'exp' or model == 'quad_aniso' or model == 'quad_exp_aniso':
        q = q0 + qd * np.cos(sep) * np.exp(- z / S)
    elif model == 'iso' or model == 'quad_iso' or model == 'quad_exp_iso':
        q = q0
  

    term = 1 + (1 - q) * z / 2. - (1 - q - 3*q**2. + j0)  * z ** 2. / 6.
    
    dl = (c * z * pre_fac_quad) * term/h0
    return dl

