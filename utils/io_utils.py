#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
I/O utility functions for nanoDSF data
"""
import os
import zipfile
import io
import pandas as pd
import numpy as np
from .parser import parse_concentration


def read_zip_data(file_obj, channel, method, window_length=21, progress_callback=None, enable_multi_peak=False, enable_interpolation=False, use_deconvolution=False):
    """
    Process nanoDSF data from a ZIP archive
    
    Parameters:
        file_obj: File-like object containing the ZIP archive
        channel (str): Data channel to use ('350/330 nm ratio', '350 nm', '330 nm')
        method (str): Analysis method ('Two-state Boltzmann', 'First derivative')
        window_length (int): Window length for Savitzky-Golay filtering
        progress_callback (callable): Optional callback function to report progress
        enable_multi_peak (bool): Whether to detect and report multiple transitions
        enable_interpolation (bool): Whether to use interpolation for smoother curves
        use_deconvolution (bool): Whether to use Gaussian deconvolution for peak detection
        
    Returns:
        tuple: (DataFrame with results, dictionary with capillary data, list of csv file names used)
    """
    from analysis import analyze_tm_derivative, analyze_tm_boltzmann
    
    results = []
    cap_data = {}
    csv_files_processed = [] # To store names of CSVs actually processed
    
    with zipfile.ZipFile(file_obj, "r") as z:
        all_files_in_zip = z.namelist()
        # Filter for raw.csv files initially - this list is used for detection later
        relevant_csv_names_for_detection = sorted([f for f in all_files_in_zip if f.lower().endswith("raw.csv")])
        
        if not relevant_csv_names_for_detection:
            # If no raw.csv files at all, return empty results and the empty list of names
            return pd.DataFrame(), {}, [] 
        
        total_files_to_process = 0 # Will count files that match channel criteria
        files_matching_channel = []

        for csv_path in relevant_csv_names_for_detection:
            # Pre-filter by channel to count total_files_to_process accurately
            original_csv_path = csv_path # Keep original name for parsing conc, etc.
            # Ensure csv_path for channel matching is just the filename if it includes full path from zip
            csv_filename_for_channel_match = os.path.basename(csv_path).lower()

            passes_channel_filter = False
            if channel.startswith("350/330") and 'ratio' in csv_filename_for_channel_match:
                passes_channel_filter = True
            elif channel == '350 nm' and '350nm' in csv_filename_for_channel_match.replace(" ", "") and 'ratio' not in csv_filename_for_channel_match:
                passes_channel_filter = True
            elif channel == '330 nm' and '330nm' in csv_filename_for_channel_match.replace(" ", "") and 'ratio' not in csv_filename_for_channel_match:
                passes_channel_filter = True
            
            if passes_channel_filter:
                files_matching_channel.append(original_csv_path)
        
        total_files_to_process = len(files_matching_channel)
        if total_files_to_process == 0:
            # No files matched the selected channel criteria
            return pd.DataFrame(), {}, relevant_csv_names_for_detection # Return all raw.csv names for detection

        for i, csv_path in enumerate(files_matching_channel):
            csv_files_processed.append(csv_path) # Add to list of processed files
            rep = os.path.splitext(os.path.basename(csv_path))[0]
            conc = parse_concentration(csv_path) # Use original path for concentration parsing
            
            if progress_callback:
                progress_callback(i+1, total_files_to_process, rep)
            
            # Read CSV data (already filtered by channel and raw.csv extension)
            raw = z.read(csv_path)
            df = pd.read_csv(io.BytesIO(raw), sep='\t')
            df.columns = [c.strip() for c in df.columns]
            
            T = df['T[°C]'].values
            F = df[df.columns[1]].values
            
            if len(T) != len(F) or len(T) < 10:
                continue
            
            if method.startswith("First derivative"):
                if enable_interpolation:
                    result = analyze_tm_derivative(
                        T, F, window_length, return_all_peaks=enable_multi_peak,
                        enable_interpolation=enable_interpolation, use_deconvolution=use_deconvolution
                    )
                    Tm, smooth, deriv, tm_idx, additional_peaks, T_interp, T_orig, F_orig = result
                    cap_data[rep] = {
                        'T': T_orig, 'F': F_orig, 'T_interp': T_interp,
                        'smooth': smooth, 'deriv': deriv, 'is_interpolated': True, 'tm_idx': tm_idx
                    }
                else:
                    Tm, smooth, deriv, tm_idx, additional_peaks = analyze_tm_derivative(
                        T, F, window_length, return_all_peaks=enable_multi_peak,
                        enable_interpolation=False, use_deconvolution=use_deconvolution
                    )
                    cap_data[rep] = {
                        'T': T, 'F': F, 'smooth': smooth, 'deriv': deriv,
                        'is_interpolated': False, 'tm_idx': tm_idx
                    }
                
                snr = np.nan # Placeholder, actual SNR calc follows
                if not np.isnan(tm_idx):
                    idx_for_snr = int(tm_idx)
                    len_deriv = len(deriv)
                    base_values = []
                    if len_deriv > 60:
                        if idx_for_snr > 35: base_values.extend(deriv[10:30])
                        if idx_for_snr < len_deriv - 35: base_values.extend(deriv[len_deriv-30:len_deriv-10])
                    if not base_values and len_deriv > 20:
                        segment_len = max(5, len_deriv // 4)
                        if idx_for_snr > segment_len + 2: base_values.extend(deriv[:segment_len])
                        if idx_for_snr < len_deriv - (segment_len + 2): base_values.extend(deriv[-segment_len:])
                    base_array = np.array(base_values)
                    if base_array.size > 1 and base_array.std() != 0:
                        snr = (deriv[idx_for_snr] - base_array.mean()) / base_array.std()
                
                ci_low = ci_high = se = r2 = log_dAIC = np.nan
                if enable_multi_peak and additional_peaks:
                    cap_data[rep]['additional_peaks'] = additional_peaks
            else: # Boltzmann
                Tm, (ci_low, ci_high), se, snr, r2, log_dAIC, popt, pcov = analyze_tm_boltzmann(T, F)
                cap_data[rep] = {'T': T, 'F': F, 'popt': popt}
            
            flag = "⚠️" if (not np.isnan(snr) and snr < 2) or (not np.isnan(log_dAIC) and log_dAIC < 1) else ""
            results.append({
                'Capillary': rep, 'TM (°C)': Tm, 'CI Lower': ci_low, 'CI Upper': ci_high,
                'SE (°C)': se, 'State SNR': snr, 'R²': r2, 'log ΔAIC': log_dAIC,
                'Flag': flag, 'Sample Info': '', 'Concentration': conc
            })
    
    # Return all raw.csv names found in the zip for experiment type detection,
    # regardless of channel filtering for actual processing.
    return pd.DataFrame(results), cap_data, relevant_csv_names_for_detection


def detect_experiment_type(csv_names, concentration_threshold=3):
    """
    Detects the experiment type based on concentration diversity in filenames.

    Parameters:
        csv_names (list): List of CSV file names (e.g., from zip archive).
        concentration_threshold (int): Minimum number of unique concentrations 
                                     to classify as 'dose-response'.

    Returns:
        str: 'dose-response' or 'screening'.
    """
    from .parser import parse_concentration # Ensure parse_concentration is available

    if not csv_names:
        return 'screening' # Default if no files

    concentrations = set()
    has_dose_folder = False
    has_sp_folder = False

    for name in csv_names:
        # Check for common folder structures as a hint
        # Use uppercase for case-insensitive comparison and check both path separators
        name_upper = name.upper()
        if 'DOSE/' in name_upper or 'DOSE\\' in name_upper: # Double backslash for literal
            has_dose_folder = True
        if 'SP/' in name_upper or 'SP\\' in name_upper: # Double backslash for literal
            has_sp_folder = True
        
        conc = parse_concentration(name)
        if conc is not None and not np.isnan(conc):
            concentrations.add(float(conc))
    
    # Strong indicator from folder structure
    if has_dose_folder and not has_sp_folder:
        return 'dose-response'
    if has_sp_folder and not has_dose_folder:
        return 'screening'
    
    # Fallback to concentration diversity
    if len(concentrations) > concentration_threshold:
        return 'dose-response'
    else:
        return 'screening' 