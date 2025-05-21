#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Plot functions for nanoDSF data visualization
"""
import numpy as np
import matplotlib.pyplot as plt
from analysis import boltzmann_exp, hill4


def plot_tm_curve(T_raw, F_raw, T_processed=None, popt=None, tm_idx=None, smooth=None, deriv=None, method="boltzmann",
               figsize=(8, 4), additional_peaks=None):
    """
    Plot TM curve with optional fit and derivative
    
    Parameters:
        T_raw (np.ndarray): Temperature array for raw data
        F_raw (np.ndarray): Fluorescence array for raw data
        T_processed (np.ndarray, optional): Temperature array for processed data (smooth, deriv). 
                                         If None, T_raw is used.
        popt (np.ndarray, optional): Optimized parameters for Boltzmann fit
        tm_idx (int, optional): TM peak index for derivative method (relative to T_processed or T_raw)
        smooth (np.ndarray, optional): Smoothed fluorescence data (should correspond to T_processed or T_raw)
        deriv (np.ndarray, optional): Derivative data (should correspond to T_processed or T_raw)
        method (str): Analysis method ('boltzmann' or 'derivative')
        figsize (tuple): Figure size
        additional_peaks (list, optional): List of additional peaks to mark on the derivative plot.
                                         Indices should be relative to T_processed or T_raw.
        
    Returns:
        list: List of created figures
    """
    figures = []
    
    # Determine the T-axis for processed data (smooth, deriv, and peak indices)
    T_for_processing = T_processed if T_processed is not None else T_raw
    
    if method == "boltzmann":
        # Boltzmann fit is typically on raw data
        fig1, ax1 = plt.subplots(figsize=figsize)
        ax1.plot(T_raw, F_raw, '.', label='Raw data')
        
        if popt is not None:
            # Generate fit curve on T_raw
            ax1.plot(T_raw, boltzmann_exp(T_raw, *popt), '-', label='Boltzmann fit')
            # Mark TM point
            tm_value = popt[6]  # Tm is the 7th parameter
            ax1.axvline(tm_value, color='red', linestyle='--', label=f'TM = {tm_value:.2f}°C')
        
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Fluorescence')
        ax1.legend()
        figures.append(fig1)
        
    elif method == "derivative":
        # Plot raw data
        fig1, ax1 = plt.subplots(figsize=figsize)
        ax1.plot(T_raw, F_raw, '.', markersize=2, alpha=0.7, label='Raw data')
        
        if smooth is not None:
            if len(T_for_processing) == len(smooth):
                ax1.plot(T_for_processing, smooth, '-', linewidth=2, label='Smoothed data')
            else:
                # This indicates a mismatch that should ideally be resolved upstream
                # For now, attempt to plot with a warning if lengths differ
                print(f"Warning: Mismatch in lengths for smoothed data plotting. T: {len(T_for_processing)}, Smooth: {len(smooth)}")
                min_len = min(len(T_for_processing), len(smooth))
                ax1.plot(T_for_processing[:min_len], smooth[:min_len], '-', linewidth=2, label='Smoothed data (trimmed)')
        
        ax1.set_xlabel('Temperature (°C)')
        ax1.set_ylabel('Fluorescence')
        ax1.legend()
        figures.append(fig1)
        
        # Plot derivative
        if deriv is not None:
            fig2, ax2 = plt.subplots(figsize=figsize)
            
            # Ensure T_for_processing has enough points for edge_trim
            if len(T_for_processing) > 20 : # Arbitrary threshold to allow trimming
                edge_trim = min(10, len(T_for_processing) // 20)
            else:
                edge_trim = 0


            if len(T_for_processing) == len(deriv):
                if edge_trim == 0 or (len(T_for_processing) - 2 * edge_trim > 0) :
                     ax2.plot(T_for_processing[edge_trim:-edge_trim], deriv[edge_trim:-edge_trim], '-', linewidth=2, label='dF/dT')
                else: # Not enough points to trim
                     ax2.plot(T_for_processing, deriv, '-', linewidth=2, label='dF/dT (no trim)')
            else:
                print(f"Warning: Mismatch in lengths for derivative data plotting. T: {len(T_for_processing)}, Deriv: {len(deriv)}")
                min_len = min(len(T_for_processing), len(deriv))
                # Recalculate safe_edge_trim based on min_len
                safe_edge_trim = 0
                if min_len > 20:
                    safe_edge_trim = min(10, min_len // 20)
                
                if safe_edge_trim == 0 or (min_len - 2 * safe_edge_trim > 0):
                    ax2.plot(T_for_processing[:min_len][safe_edge_trim:-safe_edge_trim if safe_edge_trim > 0 else min_len], 
                             deriv[:min_len][safe_edge_trim:-safe_edge_trim if safe_edge_trim > 0 else min_len], 
                             '-', linewidth=2, label='dF/dT (trimmed)')
                else:
                     ax2.plot(T_for_processing[:min_len], deriv[:min_len], '-', linewidth=2, label='dF/dT (no trim)')

            # Check if we have deconvolved peaks with fitted curves
            has_deconvolved_peaks = False
            if additional_peaks:
                for peak in additional_peaks:
                    if isinstance(peak, dict) and peak.get('deconvolved', False) and 'fitted_curve' in peak:
                        has_deconvolved_peaks = True
                        # The 'fitted_curve' is on the T_for_processing axis
                        break
            
            if has_deconvolved_peaks:
                for peak in additional_peaks:
                    if 'fitted_curve' in peak:
                        fitted_curve = peak['fitted_curve']
                        # Ensure fitted_curve corresponds to T_for_processing
                        if len(T_for_processing) == len(fitted_curve):
                             ax2.plot(T_for_processing, fitted_curve, '--', color='purple', linewidth=1.5, label='Gaussian fit')
                        else:
                            print(f"Warning: Mismatch in lengths for Gaussian fit plotting. T: {len(T_for_processing)}, Fit: {len(fitted_curve)}")
                        break 
                
                colors = ['red', 'green', 'blue', 'orange', 'cyan']
                for i, peak in enumerate(additional_peaks):
                    if peak.get('deconvolved', False):
                        amp = peak.get('amplitude', 0)
                        cen = peak.get('temp', 0) # temp is absolute
                        wid = peak.get('width', 1)
                        # Individual Gaussian curves should be generated on a fine version of T_for_processing or its range
                        gaussian_x = np.linspace(T_for_processing.min(), T_for_processing.max(), 200)
                        gaussian_y = amp * np.exp(-(gaussian_x - cen)**2 / (2 * wid**2))
                        color = colors[i % len(colors)]
                        ax2.plot(gaussian_x, gaussian_y, '-', color=color, linewidth=1, alpha=0.7,
                                label=f'Peak {i+1} (Tm={cen:.2f}°C)')
            
            marked_temps = set()
            all_transitions_display = [] # Renamed to avoid conflict
            
            # Primary transition using tm_idx (relative to T_for_processing)
            if tm_idx is not None and not np.isnan(tm_idx):
                tm_idx = int(tm_idx)
                if 0 <= tm_idx < len(T_for_processing):
                    tm_value = T_for_processing[tm_idx]
                    all_transitions_display.append({
                        'temp': tm_value, 'label': 'Low Tm', 'color': 'red', 
                        'idx': tm_idx, 'priority': 1
                    })
                    marked_temps.add(round(tm_value, 2))
            
            # Additional transitions from additional_peaks (indices relative to T_for_processing)
            if additional_peaks:
                for i, peak in enumerate(additional_peaks):
                    if isinstance(peak, dict):
                        peak_temp_val = peak.get('temp') # Absolute temperature
                        peak_idx_val = peak.get('idx')   # Index on T_for_processing

                        current_peak_temp = np.nan
                        if peak_idx_val is not None and 0 <= peak_idx_val < len(T_for_processing):
                            current_peak_temp = T_for_processing[peak_idx_val]
                        elif peak_temp_val is not None: # Fallback to temp if idx is invalid/missing
                            current_peak_temp = peak_temp_val
                            if peak_idx_val is None: # If idx was missing, try to find it
                                peak_idx_val = np.argmin(np.abs(T_for_processing - current_peak_temp))

                        if np.isnan(current_peak_temp): continue

                        rounded_temp = round(current_peak_temp, 2)
                        if rounded_temp in marked_temps: continue
                            
                        label = peak.get('label', f'Transition {i+1}')
                        color = peak.get('color', 'purple')
                        
                        # Adjust label/color for deconvolved peaks if not explicitly set
                        if peak.get('deconvolved', False) and ('label' not in peak or 'color' not in peak):
                            is_primary_deconv = not any(t['deconvolved'] for t in all_transitions_display if t['temp'] < current_peak_temp)
                            label = 'Low Tm' if is_primary_deconv else 'High Tm'
                            color = 'red' if is_primary_deconv else 'green'
                                
                        all_transitions_display.append({
                            'temp': current_peak_temp, 'label': label, 'color': color, 
                            'idx': peak_idx_val, 'priority': 2, 
                            'deconvolved': peak.get('deconvolved', False)
                        })
                        marked_temps.add(rounded_temp)
            
            all_transitions_display.sort(key=lambda x: x['temp'])
            
            for i, transition in enumerate(all_transitions_display):
                peak_temp_plot = transition['temp']
                label_plot = transition['label']
                color_plot = transition['color']
                
                # Adjust labels for multiple similar transitions
                # This logic might need refinement if relying on exact previous label
                # Check for identical labels from previous items in sorted list
                if i > 0 and label_plot == all_transitions_display[i-1]['label'] and all_transitions_display[i-1]['label'] not in ['Low Tm', 'High Tm']:
                     label_plot = f"{label_plot} #{i+1 - sum(1 for k in range(i) if all_transitions_display[k]['label'] == label_plot.split(' #')[0])}"
                elif i > 0 and label_plot == "High Tm" and all_transitions_display[i-1]['label'].startswith("High Tm"):
                     count = sum(1 for k in range(i + 1) if all_transitions_display[k]['label'].startswith("High Tm"))
                     if count > 1: label_plot = f"High Tm #{count}"


                ax2.axvline(peak_temp_plot, color=color_plot, linestyle='--', 
                           label=f'{label_plot} = {peak_temp_plot:.2f}°C')
                
                if 'idx' in transition and transition['idx'] is not None:
                    idx_plot = transition['idx']
                    # Ensure deriv has data at idx_plot if lengths mismatch earlier
                    if 0 <= idx_plot < len(T_for_processing) and 0 <= idx_plot < len(deriv):
                        marker_style = 'o' if not transition.get('deconvolved', False) else '*'
                        marker_size = 8 if not transition.get('deconvolved', False) else 10
                        ax2.plot(T_for_processing[idx_plot], deriv[idx_plot], marker_style, color=color_plot, markersize=marker_size)
            
            ax2.set_xlabel('Temperature (°C)')
            ax2.set_ylabel('dF/dT')
            ax2.grid(True, linestyle='--', alpha=0.6)
            
            # Set y-axis limits, ensure deriv has data for trimming if used
            min_deriv_val = np.min(deriv)
            max_deriv_val = np.max(deriv)

            if edge_trim > 0 and len(deriv) > 2 * edge_trim :
                 min_deriv_val = np.min(deriv[edge_trim:-edge_trim])
                 max_deriv_val = np.max(deriv[edge_trim:-edge_trim])
            
            y_plot_range = max_deriv_val - min_deriv_val
            if y_plot_range > 0: # Avoid issues if derivative is flat
                ax2.set_ylim([min_deriv_val - 0.1*y_plot_range, 
                             max_deriv_val + 0.1*y_plot_range])
            
            ax2.legend(loc='best', framealpha=0.9)
            figures.append(fig2)
    
    return figures


def plot_ec50_curve(conc, tm_values, errors=None, popt=None, figsize=(8, 6)):
    """
    Plot EC50 dose-response curve
    
    Parameters:
        conc (np.ndarray): Concentration array
        tm_values (np.ndarray): TM values array
        errors (np.ndarray, optional): Standard errors for TM values
        popt (np.ndarray, optional): Optimized parameters for Hill equation
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot experimental data with error bars
    if errors is not None:
        ax.errorbar(conc, tm_values, yerr=errors, fmt='o', label='Data ± SE')
    else:
        ax.scatter(conc, tm_values, label='Data')
    
    # Plot fit curve if parameters are provided
    if popt is not None:
        # Generate smooth curve for plotting
        x_smooth = np.logspace(np.log10(conc.min()/2), np.log10(conc.max()*2), 200)
        y_smooth = hill4(x_smooth, *popt)
        ec50 = popt[2]  # EC50 is the 3rd parameter
        ax.semilogx(x_smooth, y_smooth, '-', label=f'Fit EC₅₀={ec50:.2e} M')
    
    ax.set_xlabel('Concentration (M)')
    ax.set_ylabel('TM (°C)')
    ax.legend()
    
    return fig


def plot_delta_tm(sample_names, delta_tm_values, errors=None, figsize=(10, 6)):
    """
    Plot delta TM bar chart for screening
    
    Parameters:
        sample_names (list): List of sample names
        delta_tm_values (np.ndarray): Delta TM values array
        errors (np.ndarray, optional): Standard errors for delta TM values
        figsize (tuple): Figure size
        
    Returns:
        matplotlib.figure.Figure: Created figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create bar chart
    if errors is not None:
        ax.bar(sample_names, delta_tm_values, yerr=errors)
    else:
        ax.bar(sample_names, delta_tm_values)
    
    # Rotate x labels for better readability
    ax.set_xticklabels(sample_names, rotation=45, ha='right')
    ax.set_ylabel('ΔTM (°C)')
    
    # Add zero line
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    return fig 