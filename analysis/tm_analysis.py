#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TM analysis module
"""
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import t
from .calc import boltzmann_exp, calc_tm_derivative


def analyze_tm_derivative(T, F, window_length=21, return_all_peaks=False, enable_interpolation=False, use_deconvolution=False):
    """
    Analyze TM using first derivative method
    
    Parameters:
        T (np.ndarray): Temperature array
        F (np.ndarray): Fluorescence array
        window_length (int): Savitzky-Golay filter window length
        return_all_peaks (bool): Whether to return all potential peaks (for multi-peak detection)
        enable_interpolation (bool): Whether to use interpolation for smoother curves
        use_deconvolution (bool): Whether to use Gaussian deconvolution for peak detection
        
    Returns:
        tuple: (TM value, smoothed fluorescence, derivative curve, peak index, additional peaks list)
               If interpolation is enabled, also returns (T_interp, T_orig, F_orig)
    """
    result = calc_tm_derivative(T, F, window_length, return_all_peaks, enable_interpolation, use_deconvolution)
    
    # Handle the case where interpolation is enabled
    if enable_interpolation:
        tm_value, smooth_F, smooth_derivative, peak_idx_global, additional_peaks, T_interp, T_orig, F_orig = result
        return tm_value, smooth_F, smooth_derivative, peak_idx_global, additional_peaks, T_interp, T_orig, F_orig
    else:
        return result


def analyze_tm_boltzmann(T, F):
    """
    Analyze TM using Boltzmann equation
    
    Parameters:
        T (np.ndarray): Temperature array
        F (np.ndarray): Fluorescence array
        
    Returns:
        tuple: (TM value, confidence interval, standard error, state SNR, R², log ΔAIC, optimized parameters, covariance matrix)
    """
    def one_state(T, A, alpha, D):
        return A * np.exp(-alpha * T) + D
    
    # Calculate data characteristics
    F_range = F.max() - F.min()
    F_center = (F.max() + F.min()) / 2
    T_range = T.max() - T.min()
    T_center = (T.max() + T.min()) / 2
    
    # Initial parameters
    p0 = [
        F.max(),     # A_N
        0.005,       # alpha
        F.min(),     # D_N
        F.max()*0.8, # A_D
        0.005,       # beta
        F.min()*1.2, # D_D
        T_center,    # Tm
        0.3          # k
    ]
    
    # Fit single-state model
    try:
        popt1, pcov1 = curve_fit(one_state, T, F, p0=[F.max(), 0.005, F.min()], maxfev=200000)
        y1 = one_state(T, *popt1)
        rss1 = np.sum((F - y1)**2)
    except:
        rss1 = np.sum((F - F.mean())**2)
    
    # Multiple initial parameters fitting
    best_rss = float('inf')
    best_popt = None
    best_pcov = None
    
    initial_params = [
        p0,
        [F.max(), 0.01, F.min(), F.max()*0.9, 0.01, F.min()*1.1, T_center, 0.4],
        [F.max(), 0.003, F.min(), F.max()*0.7, 0.003, F.min()*1.3, T_center, 0.2],
    ]
    
    for p0_try in initial_params:
        try:
            popt2, pcov2 = curve_fit(boltzmann_exp, T, F, p0=p0_try, maxfev=200000)
            y2 = boltzmann_exp(T, *popt2)
            rss2 = np.sum((F - y2)**2)
            
            if rss2 < best_rss:
                best_rss = rss2
                best_popt = popt2
                best_pcov = pcov2
        except:
            continue
    
    if best_popt is None:
        return np.nan, (np.nan, np.nan), np.nan, np.nan, np.nan, np.nan, None, None
    
    # Calculate statistics
    Tm = best_popt[6]
    se = np.sqrt(np.diag(best_pcov))[6]
    dfree = len(T) - len(best_popt)
    tval = t.ppf(0.975, dfree)
    ci = (Tm - tval*se, Tm + tval*se)
    
    # Calculate residuals
    y2 = boltzmann_exp(T, *best_popt)
    resid2 = F - y2
    rss2 = np.sum(resid2**2)
    sigma_resid = np.sqrt(rss2/dfree)
    
    # Calculate state SNR
    A_N, alpha, D_N, A_D, beta, D_D = best_popt[:6]
    FN = A_N*np.exp(-alpha*Tm)+D_N
    FD = A_D*np.exp(-beta*Tm)+D_D
    deltaF = abs(FD-FN)
    snr_state = deltaF/sigma_resid if sigma_resid else np.nan
    
    # Calculate R²
    ss_tot = np.sum((F-F.mean())**2)
    r2 = 1 - rss2/ss_tot if ss_tot else np.nan
    
    # Calculate AIC
    n = len(T)
    aic1 = n*np.log(rss1/n) + 2*3
    aic2 = n*np.log(rss2/n) + 2*8
    delta_aic = aic1 - aic2
    log_delta_aic = np.log10(delta_aic) if delta_aic > 0 else 0.0
    
    return Tm, ci, se, snr_state, r2, log_delta_aic, best_popt, best_pcov 