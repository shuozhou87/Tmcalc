#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
nanoDSF Tm Calculation & Screening Streamlit App
"""
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from analysis import (
    boltzmann_exp,
    hill4,
    analyze_ec50,
    analyze_global_fit,
    calculate_delta_tm
)
from utils import read_zip_data, detect_experiment_type
from visualization import plot_tm_curve, plot_ec50_curve, plot_delta_tm, format_results_table
import zipfile


# Avoid path conflicts if script name matches module
curdir = os.path.dirname(__file__)
if curdir in sys.path:
    sys.path.remove(curdir)
sys.path.insert(0, curdir)


# Page configuration
st.set_page_config(
    page_title="nanoDSF Tm Calculator & Screening",
    layout="wide"
)
st.title("nanoDSF Tm Calculation & Screening App")


# Sidebar settings
st.sidebar.header("Analysis Settings")
method = st.sidebar.selectbox(
    "Select Tm calculation method:",
    ["Two-state Boltzmann", "First derivative"]
)
channel = st.sidebar.selectbox(
    "Select data channel:",
    ["350/330 nm ratio", "350 nm", "330 nm"]
)

# Add multi-peak detection option
enable_multi_peak = st.sidebar.checkbox(
    "Enable multi-peak detection",
    value=False,
    help="Detect multiple transitions in the same sample (for complex unfolding or ligand-induced transitions)"
)

# Add interpolation option
enable_interpolation = st.sidebar.checkbox(
    "Enable curve interpolation",
    value=False,
    help="Use cubic interpolation to create smoother curves (helps detect subtle transitions)"
)

# Add Gaussian deconvolution option
use_deconvolution = st.sidebar.checkbox(
    "Use Gaussian deconvolution",
    value=False,
    help="Fit multiple Gaussian peaks to the derivative curve for better detection of overlapping transitions"
)

# Display note that multi-peak is most effective with First derivative
if enable_multi_peak and not method.startswith("First derivative"):
    st.sidebar.warning("Note: Multi-peak detection works best with the First derivative method.")

# Display note about interpolation
if enable_interpolation:
    st.sidebar.info("💡 Interpolation creates smoother curves that can help detect subtle shoulder peaks.")

# Display note about Gaussian deconvolution
if use_deconvolution:
    if not enable_multi_peak:
        st.sidebar.warning("Note: Please enable multi-peak detection to use Gaussian deconvolution.")
    else:
        st.sidebar.info("💡 Gaussian deconvolution automatically identifies overlapping transitions without manual parameter tuning.")

window_length = 25  # Default Savitzky-Golay window length
if method.startswith("First derivative"):
    # Add help text based on whether multi-peak detection is enabled
    window_help = "Must be odd, higher values give smoother curves but may shift peak positions"
    if enable_multi_peak:
        if use_deconvolution:
            window_help = "Must be odd. For Gaussian deconvolution, a larger window (25-31) often works best to reduce noise."
        else:
            window_help = "Must be odd. For multi-peak detection: 23-25 recommended. Values >35 may shift peak positions and affect accuracy."
    
    # Calculate a reasonable maximum based on data characteristics
    max_window = 51  # Default reasonable maximum
    
    window_length = st.sidebar.number_input(
        "Savitzky–Golay window length:",
        min_value=5,
        max_value=max_window,
        step=2,
        value=25,  # Use 25 as the default window size for all cases
        help=window_help
    )
    
    # Add a warning if window size is too large
    if window_length > 35 and not use_deconvolution:
        st.sidebar.warning("⚠️ Large window sizes (>35) may shift peak positions. Peak refinement will be applied, but results should be verified.")
    
    # Add a note about window size impact when multi-peak detection is enabled
    if enable_multi_peak and not use_deconvolution:
        st.sidebar.info("💡 Tip: If transitions are missing, try adjusting the window length. Larger values (21-25) usually work well for detecting both main transitions in protein unfolding curves.")

# Table display settings
st.sidebar.header("Table Display Settings")
st.sidebar.caption("Select columns to display in the results table")

# Column display names (more user-friendly names)
column_display_names = {
    "TM (°C)": "Tm (°C)",
    "SE (°C)": "Std Error (°C)",
    "CI Lower": "CI Lower (°C)",
    "CI Upper": "CI Upper (°C)",
    "State SNR": "SNR",
    "R²": "R²",
    "log ΔAIC": "log ΔAIC",
    "Flag": "Quality Flag",
    "Include in EC50": "Include in EC50",
    "Secondary Tm (°C)": "Secondary Tm (°C)",
    "Secondary SNR": "Secondary SNR",
    "Weighted Tm (°C)": "Weighted Tm (°C)"
}

# Default columns to show (essential ones)
default_columns = ["Capillary", "TM (°C)", "State SNR", "Concentration", "Include in EC50"]

# Quick column presets
column_presets = {
    "Minimal": ["Capillary", "TM (°C)", "Concentration", "Include in EC50"],
    "Standard": ["Capillary", "TM (°C)", "State SNR", "SE (°C)", "Flag", "Concentration", "Include in EC50"],
    "Complete": ["Capillary", "TM (°C)", "State SNR", "SE (°C)", "CI Lower", "CI Upper", "R²", "log ΔAIC", "Flag", "Concentration", "Include in EC50"],
}

# If multi-peak detection is enabled, add those columns to the presets
if enable_multi_peak:
    column_presets["Standard"].extend(["Secondary Tm (°C)", "Secondary SNR", "Weighted Tm (°C)"])
    column_presets["Complete"].extend(["Primary Tm (°C)", "Primary SNR", "Secondary Tm (°C)", "Secondary SNR", "Weighted Tm (°C)"])
    # Add a multi-peak specific preset
    column_presets["Multi-peak Focus"] = ["Capillary", "Primary Tm (°C)", "Primary SNR", "Secondary Tm (°C)", "Secondary SNR", "Weighted Tm (°C)", "Concentration", "Include in EC50"]

# Add a preset selector with Standard as default
preset_options = ["Custom"] + list(column_presets.keys())
default_preset_index = preset_options.index("Standard") if "Standard" in preset_options else 0

selected_preset = st.sidebar.selectbox(
    "Column presets:",
    preset_options,
    index=default_preset_index
)

if selected_preset != "Custom":
    selected_columns = column_presets[selected_preset]
    st.sidebar.info(f"Selected preset: {selected_preset}. Customize further below if needed.")
else:
    # Additional columns available for custom selection
    optional_columns = {
        "SE (°C)": True,
        "CI Lower": False,
        "CI Upper": False,
        "R²": False,  # Default to hide R² in custom view
        "Flag": True,
        "log ΔAIC": False,
        "Sample Info": False  # Add Sample Info as an optional column that's hidden by default
    }
    
    # Multi-peak columns (only shown if multi-peak detection is enabled)
    if enable_multi_peak:
        optional_columns.update({
            "Secondary Tm (°C)": True,
            "Secondary SNR": True, 
            "Weighted Tm (°C)": True
        })
    
    # Create checkboxes for optional columns
    selected_columns = default_columns.copy()
    for col, default_state in optional_columns.items():
        display_name = column_display_names.get(col, col)
        if st.sidebar.checkbox(f"Show {display_name}", value=default_state):
            selected_columns.append(col)


# Upload data ZIP
uploaded_file = st.file_uploader("Upload nanoDSF ZIP archive", type="zip")
if not uploaded_file:
    st.info("Please upload a ZIP file containing raw CSV files.")
    st.stop()


# Initialize session state
if "current_dataset" not in st.session_state:
    st.session_state.current_dataset = None
    st.session_state.checkbox_states = {}
    st.session_state.edited_concentrations = {}

# Initialize session state for experiment type
if "experiment_type" not in st.session_state:
    st.session_state.experiment_type = None

# Check if new dataset is loaded or settings that trigger re-detection change
if uploaded_file.name != st.session_state.get("current_dataset_for_type_detection") or \
   st.session_state.get("last_file") != uploaded_file.name: # Add other relevant settings if they affect file names
    st.session_state.current_dataset_for_type_detection = uploaded_file.name
    
    # Get CSV names from the zip for type detection
    # This requires a way to peek into the zip or get names from read_zip_data more directly
    # For now, let's assume read_zip_data can also return csv_names for this purpose
    # We might need to modify read_zip_data or add a helper to get names first.
    # TEMPORARY: Read zip to get names (this is inefficient as it's read again later)
    temp_csv_names = []
    with zipfile.ZipFile(uploaded_file, "r") as z_temp:
        temp_csv_names = sorted([f for f in z_temp.namelist() if f.lower().endswith("raw.csv")])
    
    if temp_csv_names:
        st.session_state.experiment_type = detect_experiment_type(temp_csv_names)
    else:
        st.session_state.experiment_type = 'screening' # Default if no CSVs

# Display detected experiment type
if st.session_state.experiment_type:
    st.sidebar.info(f"Detected Experiment Type: {st.session_state.experiment_type.replace('-', ' ').title()}")

# Process the data
# Create progress indicators
progress_bar = st.progress(0, text="Processing data...")
progress_text = st.empty()

def update_progress(current, total, capillary):
    """Update the progress bar and display current capillary"""
    progress = current / total
    progress_bar.progress(progress, text=f"Processing {current}/{total} files...")
    progress_text.text(f"Analyzing capillary: {capillary}")

# Initialize session state for caching results and experiment type
if "processed_results" not in st.session_state:
    st.session_state.processed_results = None
    st.session_state.capillary_data = None
    st.session_state.all_csv_names_in_zip = [] 
    st.session_state.experiment_type = None 
    st.session_state.last_file_for_type_detection = None # Track file for type detection

needs_reprocessing = False
if st.session_state.get("last_file") != uploaded_file.name or \
   st.session_state.get("last_channel") != channel or \
   st.session_state.get("last_method") != method or \
   st.session_state.get("last_window") != window_length or \
   st.session_state.get("last_multi_peak") != enable_multi_peak or \
   st.session_state.get("last_interpolation") != enable_interpolation or \
   st.session_state.get("last_deconvolution") != use_deconvolution:
    needs_reprocessing = True

if needs_reprocessing:
    df_results, capillary_data, all_csv_names = read_zip_data(
        uploaded_file, channel, method, window_length, update_progress, 
        enable_multi_peak, enable_interpolation, use_deconvolution
    )
    st.session_state.processed_results = df_results
    st.session_state.capillary_data = capillary_data
    st.session_state.all_csv_names_in_zip = all_csv_names
    # Update last processed parameters
    st.session_state.last_file = uploaded_file.name
    st.session_state.last_channel = channel
    st.session_state.last_method = method
    st.session_state.last_window = window_length
    st.session_state.last_multi_peak = enable_multi_peak
    st.session_state.last_interpolation = enable_interpolation
    st.session_state.last_deconvolution = use_deconvolution
    
    # Experiment type detection should happen if the file changed, or if it hasn't been detected yet for these csvs
    if st.session_state.all_csv_names_in_zip:
        st.session_state.experiment_type = detect_experiment_type(st.session_state.all_csv_names_in_zip)
        st.session_state.last_file_for_type_detection = uploaded_file.name # Mark that type detection was run for this file content
    elif df_results is None: # Check if read_zip_data indicated no processable files
        st.error("No raw CSV files found in the ZIP archive that match channel criteria or are processable.")
        st.session_state.experiment_type = 'screening' # Default, though data is likely unusable
        st.stop() # Stop if no data could be processed
    else:
        st.session_state.experiment_type = 'screening' # Default if no CSVs somehow, but df_results exist

elif st.session_state.get("last_file_for_type_detection") != uploaded_file.name and st.session_state.all_csv_names_in_zip:
    # This covers the case where processing parameters didn't change, but the file did (e.g. re-upload of same name but new content)
    # or if loaded from cache and type detection never ran for *this specific set* of all_csv_names_in_zip
    st.session_state.experiment_type = detect_experiment_type(st.session_state.all_csv_names_in_zip)
    st.session_state.last_file_for_type_detection = uploaded_file.name

else:
    # Use cached results if no reprocessing needed
    df_results = st.session_state.processed_results
    capillary_data = st.session_state.capillary_data
    # experiment_type should persist from session_state

# Display detected experiment type if available
if st.session_state.experiment_type:
    st.sidebar.info(f"Detected Experiment Type: {st.session_state.experiment_type.replace('-', ' ').title()}")

# Clear progress indicators 
progress_bar.empty()
progress_text.empty()

if df_results is None or df_results.empty:
    # This check is crucial after all data loading/caching logic
    st.error("No data processed. Please check the ZIP file and selected channel.")
    st.stop()

# Process multi-peak data if enabled
if enable_multi_peak:
    # Initialize columns for secondary transitions
    if "Secondary Tm (°C)" not in df_results.columns:
        df_results["Primary Tm (°C)"] = np.nan
        df_results["Secondary Tm (°C)"] = np.nan
        df_results["Primary SNR"] = np.nan
        df_results["Secondary SNR"] = np.nan
        df_results["Weighted Tm (°C)"] = np.nan
    
    # First pass: collect all transitions
    all_transitions = []
    for cap_id, data in capillary_data.items():
        # Get the primary transition
        cap_mask = df_results["Capillary"] == cap_id
        if not any(cap_mask):
            continue
            
        primary_tm = df_results.loc[cap_mask, "TM (°C)"].values[0]
        primary_snr = df_results.loc[cap_mask, "State SNR"].values[0]
        
        # Add primary transition to the list
        all_transitions.append({
            "capillary": cap_id,
            "tm": primary_tm,
            "snr": primary_snr,
            "type": "primary"
        })
        
        # Add secondary transitions if they exist
        if "additional_peaks" in data and data["additional_peaks"]:
            for peak in data["additional_peaks"]:
                all_transitions.append({
                    "capillary": cap_id,
                    "tm": peak["temp"],
                    "snr": peak["snr"],
                    "type": "secondary"
                })
    
    # Group transitions by temperature clusters if we have enough data
    if len(all_transitions) >= 3:
        # Convert to DataFrame for easier analysis
        transitions_df = pd.DataFrame(all_transitions)
        
        # Calculate median temperature to separate clusters
        all_temps = transitions_df["tm"].values
        
        # If we have a clear bimodal distribution, find the optimal separation point
        # Otherwise, use the median as a simple separator
        if len(all_temps) >= 6:  # Need enough points to detect clusters
            # Sort temperatures
            sorted_temps = np.sort(all_temps)
            
            # Look for a gap in the sorted temperatures
            temp_diffs = np.diff(sorted_temps)
            if np.max(temp_diffs) > 2.0:  # If there's a gap of at least 2°C
                # Use the midpoint of the largest gap as the separator
                gap_idx = np.argmax(temp_diffs)
                separator = (sorted_temps[gap_idx] + sorted_temps[gap_idx + 1]) / 2
            else:
                # Use the median if no clear gap
                separator = np.median(all_temps)
        else:
            # Use the median for smaller datasets
            separator = np.median(all_temps)
        
        # Classify transitions as low or high based on the separator
        transitions_df["cluster"] = transitions_df["tm"].apply(lambda x: "low" if x < separator else "high")
        
        # Process each capillary to assign and reclassify primary/secondary
        for cap_id in df_results["Capillary"].unique():
            cap_transitions = transitions_df[transitions_df["capillary"] == cap_id]
            
            if len(cap_transitions) <= 1:
                continue  # Skip if only one transition
            
            # Check if we have both low and high transitions
            has_low = any(cap_transitions["cluster"] == "low")
            has_high = any(cap_transitions["cluster"] == "high")
            
            if has_low and has_high:
                # Get the best low and high transitions by SNR
                best_low = cap_transitions[cap_transitions["cluster"] == "low"].sort_values("snr", ascending=False).iloc[0]
                best_high = cap_transitions[cap_transitions["cluster"] == "high"].sort_values("snr", ascending=False).iloc[0]
                
                # Assign as primary (low) and secondary (high) transitions
                low_tm = best_low["tm"]
                low_snr = best_low["snr"]
                high_tm = best_high["tm"]
                high_snr = best_high["snr"]
                
                # Calculate weighted average
                if not np.isnan(low_snr) and not np.isnan(high_snr) and low_snr > 0 and high_snr > 0:
                    total_snr = low_snr + high_snr
                    weighted_tm = (low_tm * low_snr + high_tm * high_snr) / total_snr
                else:
                    weighted_tm = np.nan
                
                # Update the results dataframe
                cap_mask = df_results["Capillary"] == cap_id
                if any(cap_mask):
                    df_results.loc[cap_mask, "Primary Tm (°C)"] = low_tm
                    df_results.loc[cap_mask, "Secondary Tm (°C)"] = high_tm
                    df_results.loc[cap_mask, "Primary SNR"] = low_snr
                    df_results.loc[cap_mask, "Secondary SNR"] = high_snr
                    df_results.loc[cap_mask, "Weighted Tm (°C)"] = weighted_tm
                    
                    # Also update the TM column to be the low temperature transition for consistency
                    df_results.loc[cap_mask, "TM (°C)"] = low_tm
                    df_results.loc[cap_mask, "State SNR"] = low_snr

# Apply any manually edited concentrations
if st.session_state.edited_concentrations:
    for index, row in df_results.iterrows():
        cap_id = row["Capillary"]
        if cap_id in st.session_state.edited_concentrations:
            df_results.loc[index, "Concentration"] = st.session_state.edited_concentrations[cap_id]


# Initialize checkbox states for all capillaries
for _, row in df_results.iterrows():
    if row["Capillary"] not in st.session_state.checkbox_states:
        st.session_state.checkbox_states[row["Capillary"]] = True


# Add checkbox column using preserved states
df_results["Include in EC50"] = df_results["Capillary"].map(st.session_state.checkbox_states)


# Sort the dataframe by capillary ID
try:
    df_results = df_results.sort_values("Capillary")
except:
    pass


# Reset index for display
df_display = df_results.reset_index(drop=True)


# Convert concentration to string for TextColumn compatibility
if "Concentration" in df_display.columns:
    df_display["Concentration"] = df_display["Concentration"].apply(
        lambda x: f"{x:.2e}" if pd.notnull(x) else ""
    )


# Summary table
st.header("Summary of Tm Results")
st.caption("Customize displayed columns using the 'Table Display Settings' in the sidebar")

# Filter columns based on user selection
valid_columns = [col for col in selected_columns if col in df_display.columns]
df_display_filtered = df_display[valid_columns].copy()

# Set up the table configuration
column_config = format_results_table(df_display, enable_multi_peak)
column_config_filtered = {col: column_config[col] for col in valid_columns if col in column_config}

# Display the editable table
editor_key = f"editor_{uploaded_file.name}_{channel}_{method}"
edited_df = st.data_editor(
    df_display_filtered,
    key=editor_key,
    hide_index=True,
    use_container_width=True,
    column_config=column_config_filtered
)


# Update session state with edited values
for index, row in edited_df.iterrows():
    cap_id = row["Capillary"]
    
    # Update Include in EC50 if present
    if "Include in EC50" in row:
        st.session_state.checkbox_states[cap_id] = row["Include in EC50"]
    
    # Update Concentration if present
    if "Concentration" in row:
        conc_str = str(row["Concentration"])
        if conc_str.strip() == "":
            st.session_state.edited_concentrations[cap_id] = None
        else:
            try:
                parsed_conc = float(conc_str)
                st.session_state.edited_concentrations[cap_id] = parsed_conc
            except ValueError:
                st.warning(
                    f"Invalid concentration for {cap_id}: '{conc_str}'. "
                    "Use numbers or scientific notation (e.g., 1e-7)."
                )
                if cap_id not in st.session_state.edited_concentrations:
                    st.session_state.edited_concentrations[cap_id] = None
    
    # Update Sample Info if present
    if "Sample Info" in row and "Sample Info" in df_results.columns:
        sample_info = row["Sample Info"]
        # Find the corresponding row in the full results dataframe
        mask = df_results["Capillary"] == cap_id
        if any(mask):
            df_results.loc[mask, "Sample Info"] = sample_info


# Prepare data for analysis
st.info("Fill 'Sample Info' and 'Concentration', then run EC50 or ΔTm analysis.")


# EC50 analysis section
if st.button("Calculate EC50"):
    # Create dataframe for fitting with updated concentrations
    df_fit = df_results.copy()
    
    # Update with edited concentrations
    for idx, row in df_fit.iterrows():
        cap_id = row["Capillary"]
        if cap_id in st.session_state.edited_concentrations:
            df_fit.loc[idx, "Concentration"] = st.session_state.edited_concentrations[cap_id]
    
    # Filter for selected capillaries with valid concentrations
    df_fit = df_fit[df_fit["Include in EC50"]].copy()
    df_fit["Concentration"] = pd.to_numeric(df_fit["Concentration"], errors="coerce")
    df_fit.dropna(subset=["Concentration"], inplace=True)
    
    if len(df_fit) < 3:
        st.error("Need at least 3 selected capillaries with concentration values for EC50 fitting.")
        st.stop()
    
    # Check if we need to calculate EC50 for multiple transitions
    calculate_secondary = enable_multi_peak and "Secondary Tm (°C)" in df_fit.columns and df_fit["Secondary Tm (°C)"].notna().sum() >= 3
    
    # Prepare columns for EC50 fitting
    df_fit["Primary TmToFit"] = df_fit["TM (°C)"]
    
    if calculate_secondary:
        # For secondary transition, create a valid subset
        df_fit_secondary = df_fit.dropna(subset=["Secondary Tm (°C)"]).copy()
    
    # Extract data for primary transition fitting
    x_primary = df_fit["Concentration"].astype(float).values
    y_primary = df_fit["Primary TmToFit"].values
    errors_primary = df_fit["SE (°C)"].astype(float).values
    
    # Display header based on whether we have multiple transitions
    if calculate_secondary:
        st.subheader("Primary Transition: Dose–Response Fit Results (4PL)")
    else:
        st.subheader("Dose–Response Fit Results (4PL)")
    
    # Perform EC50 analysis for primary transition
    ec50_primary, ci_primary, se_primary, r2_primary, popt_primary, pcov_primary = analyze_ec50(x_primary, y_primary)
    
    # Display results for primary transition
    st.write(f"EC50 = {ec50_primary:.2e} M (95% CI: {ci_primary[0]:.2e}–{ci_primary[1]:.2e})")
    st.write(f"R² = {r2_primary:.3f}")
    
    # Plot EC50 curve for primary transition
    fig_primary = plot_ec50_curve(x_primary, y_primary, errors_primary, popt_primary)
    st.pyplot(fig_primary)
    plt.close(fig_primary)
    
    # If we have enough data for secondary transition EC50, calculate it
    if calculate_secondary:
        # Extract data for secondary transition fitting
        x_secondary = df_fit_secondary["Concentration"].astype(float).values
        y_secondary = df_fit_secondary["Secondary Tm (°C)"].values
        errors_secondary = df_fit_secondary["SE (°C)"].astype(float).values
        
        # Secondary transition header
        st.subheader("Secondary Transition: Dose–Response Fit Results (4PL)")
        
        # Perform EC50 analysis for secondary transition
        try:
            ec50_secondary, ci_secondary, se_secondary, r2_secondary, popt_secondary, pcov_secondary = analyze_ec50(x_secondary, y_secondary)
            
            # Display results for secondary transition
            st.write(f"EC50 = {ec50_secondary:.2e} M (95% CI: {ci_secondary[0]:.2e}–{ci_secondary[1]:.2e})")
            st.write(f"R² = {r2_secondary:.3f}")
            
            # Plot EC50 curve for secondary transition
            fig_secondary = plot_ec50_curve(x_secondary, y_secondary, errors_secondary, popt_secondary)
            st.pyplot(fig_secondary)
            plt.close(fig_secondary)
        except Exception as e:
            st.error(f"Failed to calculate EC50 for secondary transition: {e}")


# Single-dose ΔTm screening section
# This section is shown if experiment_type is 'screening'
st.header("Single-Dose ΔTm Screening")
st.markdown("Select a control capillary. ΔTm will be calculated for all other capillaries **marked as included** in the summary table above.")

# Get all capillary options for the control dropdown from the original, unfiltered df_results
all_cap_options_for_control = df_results["Capillary"].unique().tolist()

if not all_cap_options_for_control:
    st.warning("No capillary data available to select a control for ΔTm.")
else:
    control_cap = st.selectbox(
        "Select control capillary", 
        all_cap_options_for_control, 
        key="delta_tm_control_selectbox_screening"
    )

    if st.button("Calculate ΔTm for All Other Included Samples", key="delta_tm_button_screening"):
        if not control_cap:
            st.warning("Please select a control capillary.")
        else:
            # Filter df_results to get only those samples marked for inclusion for this analysis
            # df_results already has the "Include in EC50" column updated from session_state
            df_included_samples = df_results[df_results["Include in EC50"] == True].copy()

            # Ensure the control capillary exists in the original df_results to get its Tm
            control_mask_original = df_results["Capillary"] == control_cap
            if not any(control_mask_original):
                st.error(f"Control capillary '{control_cap}' not found in the base data. This should not happen.")
                st.stop()
            
            t0_row_original = df_results.loc[control_mask_original]
            t0 = t0_row_original["TM (°C)"].values[0]
            s0 = t0_row_original.get("SE (°C)", pd.Series([0.0])).fillna(0.0).values[0]

            # Test capillaries are those from df_included_samples that are not the control_cap
            test_capillaries_from_included = [
                c for c in df_included_samples["Capillary"].unique().tolist() if c != control_cap
            ]

            if not test_capillaries_from_included:
                st.warning("No other *included* capillaries available to compare against the control. Please check the 'Include in EC50' column in the summary table.")
            else:
                delta_tm_rows = []
                for cap_id_test in test_capillaries_from_included:
                    # Get data for the test capillary from the df_included_samples
                    test_mask_included = df_included_samples["Capillary"] == cap_id_test
                    # Given how test_capillaries_from_included is constructed, mask should always find something
                    test_row_included = df_included_samples.loc[test_mask_included]
                    
                    tm_test = test_row_included["TM (°C)"].values[0]
                    se_test = test_row_included.get("SE (°C)", pd.Series([0.0])).fillna(0.0).values[0]
                    
                    delta_tm, se_delta = calculate_delta_tm(t0, tm_test, s0, se_test)
                    
                    sample_info_val = ""
                    # Sample Info should come from the potentially edited data in edited_df
                    # We need to ensure edited_df is used here for Sample Info consistency
                    # If this section runs before edited_df is defined in this script run, fall back to df_results
                    # (Actually, df_results has Sample Info updated from edited_df in the previous run)
                    source_df_for_sample_info = edited_df if 'edited_df' in locals() and not edited_df.empty else df_results
                    
                    info_mask_for_sample = source_df_for_sample_info["Capillary"] == cap_id_test
                    if any(info_mask_for_sample) and "Sample Info" in source_df_for_sample_info.columns:
                        sample_info_val = source_df_for_sample_info.loc[info_mask_for_sample, "Sample Info"].values[0]
                    
                    delta_tm_rows.append({
                        "Capillary": cap_id_test,
                        "Sample Info": sample_info_val if pd.notnull(sample_info_val) and sample_info_val.strip() != "" else cap_id_test,
                        "ΔTm (°C)": delta_tm,
                        "SE ΔTm (°C)": se_delta
                    })
                    
                if delta_tm_rows:
                    df_delta_screening = pd.DataFrame(delta_tm_rows).sort_values("ΔTm (°C)", ascending=False)
                    st.subheader("ΔTm Results (for Included Samples)")
                    st.dataframe(df_delta_screening, column_config={
                        "Capillary": "Capillary ID",
                        "Sample Info": "Sample Name/Info",
                        "ΔTm (°C)": st.column_config.NumberColumn("ΔTm (°C)", format="%.2f"),
                        "SE ΔTm (°C)": st.column_config.NumberColumn("SE ΔTm (°C)", format="%.2f"),
                    }, use_container_width=True, hide_index=True)
                    
                    chart_labels_screening = df_delta_screening["Sample Info"].tolist()
                    # Ensure uniqueness for labels, fallback to Capillary ID if Sample Info is not unique or empty
                    if len(set(chart_labels_screening)) != len(chart_labels_screening) or any(s == cap_id for s, cap_id in zip(chart_labels_screening, df_delta_screening["Capillary"].tolist())):
                        chart_labels_screening = df_delta_screening["Capillary"].tolist()
                        
                    fig_delta_tm_screening = plot_delta_tm(chart_labels_screening, df_delta_screening["ΔTm (°C)"].values, df_delta_screening["SE ΔTm (°C)"].values)
                    st.pyplot(fig_delta_tm_screening)
                    plt.close(fig_delta_tm_screening)
                else:
                    st.info("No ΔTm values were calculated for the included test samples.")


# Detailed per-capillary plots
st.header("Detailed Curves")
st.info("Click on a capillary to view detailed curves.")

for cap_id, data in capillary_data.items():
    with st.expander(f"Capillary {cap_id}"):
        T_raw_plot = data['T']
        F_raw_plot = data['F']
        
        if method.startswith("First derivative"):
            # Determine the correct T-axis for plotting processed data and for peak indices
            T_processed_for_plot = data.get('T_interp') if data.get('is_interpolated') else None
            T_plot_axis = T_processed_for_plot if T_processed_for_plot is not None else T_raw_plot
            
            current_smooth = data.get('smooth')
            current_deriv = data.get('deriv')
            
            transitions = [] # Initialize the list for peaks to plot

            # Populate 'transitions' list
            # Full UI logic for peak selection (st.data_editor) would go here if enable_multi_peak is True.
            # This is a simplified version for the fix:
            if enable_multi_peak:
                all_calculated_peaks = data.get("additional_peaks", []) # From calc_tm_derivative
                if not all_calculated_peaks and data.get('tm_idx') is not None and not np.isnan(data.get('tm_idx')):
                    # If multi-peak enabled but no 'additional_peaks', use primary tm_idx as a fallback peak
                    primary_idx = int(data['tm_idx'])
                    if 0 <= primary_idx < len(T_plot_axis):
                        all_calculated_peaks = [{
                            'idx_global': primary_idx,
                            'temp': T_plot_axis[primary_idx],
                            'snr': data.get('snr', np.nan), # Attempt to get primary SNR if available
                            'type': 'peak', # Assume it's a peak
                            'deconvolved': False
                        }]
                
                for i, peak_info in enumerate(all_calculated_peaks):
                    idx_for_plot = peak_info.get('idx_global') # This index is on T_plot_axis
                    temp_val = peak_info.get('temp')
                    
                    if idx_for_plot is not None and temp_val is not None:
                        # Simplified labeling for this fix - can be enhanced later
                        label = f"Peak {i+1}"
                        color = "purple"
                        if peak_info.get('type') == 'dip':
                            label = f"Dip {i+1}"
                            color = "orange"
                        elif i == 0 and peak_info.get('type') == 'peak': # First peak is often primary
                            label = "Low Tm"
                            color = "red"
                        elif peak_info.get('type') == 'peak':
                            label = f"High Tm {i}"
                            color = "green"

                        if peak_info.get('deconvolved'):
                             label = f"Deconv {label}" # Prepend if deconvolved

                        transitions.append({
                            'idx': idx_for_plot,
                            'temp': temp_val,
                            'label': label,
                            'color': color,
                            'snr': peak_info.get('snr'),
                            'deconvolved': peak_info.get('deconvolved', False),
                            'amplitude': peak_info.get('amplitude'), 
                            'width': peak_info.get('width'),
                            'fitted_curve': peak_info.get('fitted_curve')
                        })
            else: # Not enable_multi_peak - use primary Tm from df_results as the only transition
                cap_mask = df_results["Capillary"] == cap_id
                if any(cap_mask):
                    primary_tm_val_from_results = df_results.loc[cap_mask, "TM (°C)"].iloc[0]
                    if pd.notnull(primary_tm_val_from_results):
                        # Find index of this Tm on the T_plot_axis
                        tm_idx_for_plot = np.argmin(np.abs(T_plot_axis - primary_tm_val_from_results))
                        transitions.append({
                            'idx': tm_idx_for_plot, 
                            'temp': primary_tm_val_from_results, 
                            'label': "Tm", 
                            'color': "red"
                        })
            
            # Sort transitions by temperature for consistent labeling and plotting order
            transitions.sort(key=lambda x: x['temp'])
            
            # (Optional: Add more sophisticated relabeling logic here if needed, e.g., Low Tm, High Tm #2 etc.)

            # Call plot_tm_curve for the derivative method
            figs = plot_tm_curve(
                T_raw_plot, 
                F_raw_plot,
                T_processed=T_plot_axis, 
                tm_idx=None,  # Primary Tm marking is now handled by 'additional_peaks' if it's the first item
                smooth=current_smooth,
                deriv=current_deriv,
                method="derivative",
                additional_peaks=transitions 
            )
            for fig_item in figs:
                st.pyplot(fig_item)
                plt.close(fig_item)
        
        else: # Boltzmann method
            popt = data.get("popt")
            # For Boltzmann, T_processed is None; plotting is on T_raw_plot
            figs = plot_tm_curve(
                T_raw_plot, 
                F_raw_plot,
                T_processed=None, 
                popt=popt,
                method="boltzmann"
            )
            for fig_item in figs:
                st.pyplot(fig_item)
                plt.close(fig_item)

st.success("Analysis complete.") 