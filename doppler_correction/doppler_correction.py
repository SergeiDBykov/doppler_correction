# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 12:29:57 2019

@author: s.bykov
"""

import astropy.io.fits as fits
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from tqdm import tqdm as progressbar
import os
day2sec = 86400

orb_params_v0332 = {'P': 33.850*day2sec, 'e': 0.3713, 'asini': 77.81,
                    'w': np.deg2rad(277.43), 'T_p': 57157.88*day2sec}


orb_params_gx304 = {'P': 132.189*day2sec, 'e': 0.462, 'asini': 601,
                    'w': np.deg2rad(130), 'T_p': 55425.60000000*day2sec}


orb_params_gx301 = {'P': 41.498*day2sec, 'e': 0.462, 'asini': 368.3,
                    'w': np.deg2rad(310.4), 'T_p': 48802.79*day2sec}  # https://www.aanda.org/articles/aa/pdf/2005/29/aa1509-04.pdf


orb_params_cenx3 = {'P': 2.086*day2sec, 'e': 0, 'asini': 39.653,
                    'w': np.deg2rad(0), 'T_p': 57355.8361*day2sec}


orb_params_xtej1946 = {'P': 172.218*day2sec, 'e': 0.286, 'asini': 466,
                       'w': 4.782, 'T90': 58182.27*day2sec}

orb_params_dull = {'P': 0, 'e': 0, 'asini': 0,
                   'w': 0, 'T_p': 0}


# https://gammaray.nsstc.nasa.gov/gbm/science/pulsars/lightcurves/swiftj0243.html
orb_params_s0243 = {'P': 27.698899*day2sec, 'e': 0.1029, 'asini': 115.531,
                    'w': np.deg2rad(-74.05), 'T90': np.float(58115.597*day2sec)}


def Tp_from_T90(T_90, e, P):
    E = 2*np.arctan(np.sqrt((1-e)/(1+e)))
    M = E-e*np.sin(E)

    Tp = T_90-P*M/(2*np.pi)
    return Tp


def _kepler_solution(time: float, orb_params: dict):
    '''
    time- time in MJD seconds 
    P- binary orbit, in seconds
    e-eccentricity
    asini- semi major axis projection, lt-sec
    w - longitude of periastron, rad
    T_p - periastron passage time, in seconds


    Peiod_observed=factor* Period_real; Period_real=Period_observad/factor
    => freq_obs=freq_real/factor; f_real=factor*freq_obs.  V_r might be negative.

    returns: rsini,v(angle),v_R(rad velocity),dopplef factor, z coordinate

    '''
    P = orb_params['P']
    e = orb_params['e']
    asini = orb_params['asini']
    w = orb_params['w']
    try:
        T_p = orb_params['T_p']
    except:
        T90 = orb_params['T90']
        T_p = Tp_from_T90(T90, e, P)

    mu = 2*np.pi/P
    K = mu*asini/np.sqrt(1-e**2)

    M = 2*np.pi*(time-T_p)/P  # mean anomaly

    def func(E): return E-e*np.sin(E)-M  # E- eccentric anomaly
    init_guess = M+e*np.sin(M)+e**2/M*np.sin(2*M)  # argyle04

    try:
        E = fsolve(func, init_guess)[0]
    except:
        raise Exception('Failed to find eccentric anomaly!')
        return None, None, None, None, None

    # true anomaly, it is an angle
    v = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
    # line of sight (LOS) velocity, in the units of c
    V_R = K*(e*np.cos(w)+np.cos(v+w))
    # projection of the vector (center -> star) in the LOS
    rsini = asini*(1-e**2)/(1+e*np.cos(v))
    doppler_factor = np.sqrt((1+V_R)/(1-V_R))

    # rsini /=doppler_factor     #see Hilditch
    z = rsini*np.sin(v+w)  # projection of the vector to the vertical axis

    return rsini, v, V_R, doppler_factor, z


def kepler_solution(times: np.ndarray, orb_params: dict):
    '''
    One cannot use time as array in _kepler_solution because of
    numerical solution of an equation
    This procedure return and array of interesting kepler values for an input array

    times: time stemps in MJD seconds
    P- binary orbit, in seconds
    e-eccentricity
    asini- semi major axis projection, lt-sec
    w - longitude of periastron, rad
    T_p - periastron passage time, in seconds
    returns: rsini,v(angle),v_R(rad velocity),dopplef factor, z coordinate

    Peiod_observed=factor* Period_real; Period_real=Period_observad/factor
    => freq_obs=freq_real/factor; f_real=factor*freq_obs.  !!!! V_r might be negative !!!!!
    '''
    times = np.array(times)
    N = len(times)

    arr_rsini = np.zeros(N)
    arr_v = np.zeros(N)
    arr_V_R = np.zeros(N)
    arr_doppler_factor = np.zeros(N)
    arr_z = np.zeros(N)

    for i in progressbar(range(N), desc='Solving kepler equations'):
        rsini, v, V_R, doppler_factor, z = _kepler_solution(
            times[i], orb_params)
        arr_rsini[i] = rsini
        arr_v[i] = v
        arr_V_R[i] = V_R
        arr_doppler_factor[i] = doppler_factor
        arr_z[i] = z

    return arr_rsini, arr_v, arr_V_R, arr_doppler_factor, arr_z


def correct_times(fitsfile: str, orb_params: dict, time_orig_col: str = 'time') -> None:
    filepath = os.path.dirname('./'+fitsfile)
    os.chdir(filepath)
    if not os.path.exists(fitsfile):
        raise Exception('File does not exist')

    filename = fitsfile.split('/')[-1]
    new_filename = filename+'_orb_corr'
    if os.path.exists(new_filename):
        raise Exception('File already exists!')
    os.system(f'cp {filename} {new_filename}')

    if orb_params == orb_params_dull:
        print('no orbital parameters given, copying the original file without changes')

    else:
        ff = fits.open(fitsfile)
        if ff[0].header['timesys'] != 'TDB':
            raise Exception(
                'TIMESYS keyword is not TBD (not a  barycentred time)')

        if 'orb correct' in ff[0].header.comments['timesys']:
            raise Exception(
                'Time has already been corrected for binary motion')
        mjdrefi = ff[1].header['mjdrefi']
        mjdreff = ff[1].header['mjdreff']
        timezero = ff[1].header['timezero']
        # the following formula from https://heasarc.gsfc.nasa.gov/docs/xte/abc/time_tutorial.html #TODO check applicability for nustar
        time_orig_mjd = (ff[1].data[time_orig_col]+timezero) / \
            86400+mjdreff+mjdrefi  # barycentred time in MJD units

        _, _, _, _, dt = kepler_solution(time_orig_mjd*day2sec, orb_params)
        new_time = time_orig_mjd*day2sec-dt
        with fits.open(filepath+'/'+new_filename, mode='update') as hdul:
            hdul[0].header.comments['timesys'] = 'tbd with orb correction'
            hdul[1].data[time_orig_col] = new_time
            hdul.flush()  # changes are written back to original.fits


def plot_orbit(orb_params, points_time=[], points_label=[], ax=None, N_pl=100):
    if ax == None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='polar')
    else:
        ax = ax
    times = np.linspace(0, orb_params['P'], N_pl)
    r, v, _, _, _ = kepler_solution(times, orb_params)
    ax.plot(v+orb_params['w'], r)

    if len(points_time) != 0:
        for time, label in zip(points_time, points_label):
            r, v, _, _, _ = kepler_solution(time, orb_params)
            ax.plot(v+orb_params['w'], r, 'o', label=label)

    ax.set_title('Anti-clockwise movement \n observer is at the bottom')
    plt.show()
    plt.legend()
