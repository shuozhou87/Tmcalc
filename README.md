# nanoDSF Analysis Tool

A Streamlit-based application for analyzing nanoDSF (nano Differential Scanning Fluorimetry) data.

## Project Structure

```
.
├── main.py                 # Main application entry point
├── analysis/              # Analysis module
│   ├── __init__.py        # Module exports
│   ├── calc/             # Core calculation functions
│   │   ├── __init__.py
│   │   ├── tm_calc.py    # Tₘ calculation functions
│   │   └── curve_fit.py  # Curve fitting functions
│   ├── tm_analysis.py    # Tₘ analysis functions
│   ├── ec50_analysis.py  # EC₅₀ analysis functions
│   └── screening.py      # Screening analysis functions
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── io_utils.py       # File I/O utilities
│   └── parser.py         # Data parsing utilities
├── visualization/       # Data visualization
│   ├── __init__.py
│   └── plots.py         # Plotting functions
├── DOSE/                # Folder for dose-response datasets
├── SP/                  # Folder for single-point experiment datasets
└── requirements.txt      # Project dependencies
```

## Data Organization

The application is designed to work with nanoDSF data organized in specific folders:

- **DOSE/** - Contains dose-response datasets where the same protein is analyzed with a series of different ligand concentrations. Used for EC50 calculation.
  - Example: `DOSE/XBB_S22_DOSE.zip` contains capillary data for EC50 analysis

- **SP/** - Contains single-point experiment datasets where proteins are analyzed with single concentrations of different ligands. Used for ΔTm screening.

Each folder should contain zip archives of nanoDSF raw data exported from the Prometheus instrument.

## Module Description

### Main Application (`main.py`)
- Entry point for the Streamlit application
- Handles user interface and data flow
- Integrates all analysis modules

### Analysis Module (`analysis/`)
- Core analysis functionality
- Organized into submodules for different analysis types

#### Calculation Submodule (`analysis/calc/`)
- Core mathematical functions
- Basic curve fitting algorithms
- Utility functions for data processing

#### Tₘ Analysis (`analysis/tm_analysis.py`)
- Tₘ calculation methods
- Derivative analysis
- Boltzmann fitting

#### EC₅₀ Analysis (`analysis/ec50_analysis.py`)
- EC₅₀ calculation
- Dose-response fitting
- Global fitting analysis

#### Screening Analysis (`analysis/screening.py`)
- ΔTₘ analysis
- Statistical significance testing
- Data filtering and processing

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the application:
```bash
streamlit run main.py
```

## Features

- Tₘ calculation using multiple methods:
  - First derivative method (with optional multi-peak detection)
  - Two-state Boltzmann fitting
- EC₅₀ analysis with dose-response curves
- Screening analysis with ΔTₘ calculations
- Interactive data visualization
- Multi-peak detection for complex protein unfolding behaviors
- Export functionality for results 