from bao_prior import rs_zdrag
from astropy.cosmology import LambdaCDM
from cmb_prior import z_lss, rs_approx
from scipy.integrate import quad

import numpy as np

c=2.99e5

def ez_lcdm(z, om, okexp):
    #simple 1/E(z) term for nonflat LambdaCDM 
    t = np.sqrt(om*(1+z)**3 + (1 - om - okexp) + okexp*(1+z)**2)
    return 1./t


def curv_lumdist_lcdm(zhel, zcmb, om, okgeo, okexp, h0, log=True):
    out = np.empty_like(zcmb)
    for i, zval in enumerate(zcmb):
	out[i] = quad(ez_lcdm, 0, zval, args=(om, okexp))[0]
    
    curv_geo = np.sqrt(abs(okgeo))	

    if okgeo > 0:
	temp = np.sinh(curv_geo*out)
        dist_mpc =c*(1+zhel)*temp/(h0 * curv_geo)

    elif okgeo < 0:
	temp = np.sin(curv_geo*out)

	dist_mpc =c*(1+zhel)*temp/(h0 * curv_geo)

    elif okgeo == 0:
	temp = out
        dist_mpc =c*(1+zhel)*temp/(h0)
    
    if log:
	return 5*np.log10(dist_mpc)+25
    else:
        return dist_mpc


def D_v(z, om,ok, h0=70):
    h = h0/100.
    #cosmo = LambdaCDM(h0, om, ol)
    out = np.empty_like(z)
    z = np.array([z])
    d_a = curv_lumdist_lcdm(z, z, om, ok, ok, h0, log=False)/(1+z)#, 0, z, args=(ophi, w0, K))[0]/h0
    out = d_a**2 * c*z
    hz = h0/ez_lcdm(z, om, ok)
    out/=hz
    return pow(out, 1./3.)

def dz_bao(om, ol, h0, zarr= np.array([.106, .35, .57])):
    rsd = rs_zdrag(om, -0.9, 2.3, h0)
    dvarr = np.array([D_v(zval,om,ol, h0=h0)[0] for zval in zarr])
    dzarr = rsd/dvarr
    return dzarr

def R_shift_lcdm(om, ok, h0, omega_bar=0.0448):
    zstar = z_lss(om, h0)
    out = quad(ez_lcdm, 0, zstar, args=(om, ok))[0]
    com_dist = out*c/h0 
    r_shift = np.sqrt(om*h0**2)*com_dist/c
    return r_shift

def la(om, ok, h0):
    rs_star = rs_approx(om, h0)
    d_a  = comov_decoup(om, ok, h0)
    lval = np.pi*d_a/rs_star
    return lval 
####------------------ Ratio bits ------------------#######


def comov_decoup(om, ok, h0):
    zstar = z_lss(om, h0)
    cdist = curv_lumdist_lcdm(np.array([zstar]), np.array([zstar]), om, ok, ok, h0, log=False)/(1+zstar) #LambdaCDM(h0, om, ol).comoving_distance(zstar).value#c*quad(ez_grow, 0, zstar, args=(om, oe, omega_nu))[0]/h0
    return cdist

def D_vnew(z, om, ok, h0=70):
    h = h0/100.
    out = np.empty_like(z)
    zarr = np.array([z])
    d_a = curv_lumdist_lcdm(zarr, zarr, om, ok, ok, h0, log=False)/(1+z)
    out = d_a**2 * c*z
    hz = h0/ez_lcdm(z,om,ok) 
    out/=hz
    return pow(out, 1./3.)

def cmbao_ratio(om, ok, h0, z=0.2):
    d_comov = comov_decoup(om, ok, h0)
    d_v = D_vnew(z, om, ok, h0=h0)
    #although the imaginary part is 0, still take only the real
    ##(comes from solving the quartic to get the E(z))
    return np.real(d_comov/d_v)

##---------- Defined the CMB compressed likelihood and the BAO d_z ----------- ###########

