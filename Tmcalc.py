# MIT License
# 
# Copyright (c) 2024 Shuo Zhou
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import sys

def calculate_tm_and_r2_adjusted(data):
    # Filter out temperatures outside the 30°C to 90°C range but retain original indices
    filtered_data = data[(data['Temperature'] > 30) & (data['Temperature'] < 90)]

    if filtered_data.empty:
        return None, None, None  # Return None if no data within specified range

    min_derivative_index = filtered_data['Derivative'].idxmin()
    window_size = 15  # 7 points on each side

    start_index = max(min_derivative_index - window_size // 2, filtered_data.index.min())
    end_index = min(min_derivative_index + window_size // 2 + 1, filtered_data.index.max() + 1)

    if end_index - start_index < 8:
        return None, None, None  # Insufficient data points for a reliable fit

    temp_fit = filtered_data.loc[start_index:end_index, 'Temperature']
    deriv_fit = filtered_data.loc[start_index:end_index, 'Derivative']

    coefficients = np.polyfit(temp_fit, deriv_fit, 2)
    poly_func = np.poly1d(coefficients)

    tm_calculated = -coefficients[1] / (2 * coefficients[0])
    r_squared = r2_score(deriv_fit, poly_func(temp_fit))

    return tm_calculated, r_squared, coefficients

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py path_to_your_file.xlsx output_filename.csv")
        sys.exit(1)

    file_path = sys.argv[1]
    output_file = sys.argv[2]
    data = pd.read_excel(file_path, sheet_name='Melt Curve Raw Data', skiprows=41)
    data = data[['Well Position', 'Temperature', 'Derivative']].reset_index(drop=True)

    well_data = {well: grp.reset_index(drop=True) for well, grp in data.groupby('Well Position')}
    tm_results = {well: calculate_tm_and_r2_adjusted(well_data[well]) for well in well_data.keys()}
    
    results_df = pd.DataFrame(columns=['Well Position', 'Tm Values', 'R Squares'])

    for well, results in tm_results.items():
        if results[0] is not None:
            new_row = pd.DataFrame({'Well Position': [well], 'Tm Values': [results[0]], 'R Squares': [results[1]]})
            results_df = pd.concat([results_df, new_row], ignore_index=True)
        else:
            new_row = pd.DataFrame({'Well Position': [well], 'Tm Values': ['Not enough data'], 'R Squares': ['Not enough data']})
            results_df = pd.concat([results_df, new_row], ignore_index=True)
    
    results_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    main()
