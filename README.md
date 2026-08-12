# Bayesian Switching Dynamical Systems

This repository contains a full object-oriented Python implementation of state-of-the-art Bayesian switching algorithms for regime discovery in financial time series.

## Features
- **Hidden Markov Models (HMM)**
- **Sticky Hierarchical Dirichlet Process AR-HMM (HDP-AR-HMM)**
- **Sticky HDP Switching Linear Dynamical System (HDP-SLDS)**
- **Recurrent Switching Linear Dynamical System (rSLDS)** with Polya-Gamma augmentation.

## Architecture
The repository uses a highly modular architecture built around Gibbs sampling, Forward-Filtering Backward-Sampling (FFBS), and robust Bayesian updates. 

* `src/recurrent_slds.py` - Core rSLDS model.
* `src/hdp_slds.py` - Sticky HDP-SLDS model.
* `src/hdp_arhmm.py` - Sticky HDP-AR-HMM model.
* `src/backtest.py` - Regime strategy evaluation and plotting.
* `get_data.py` - Utility to fetch 14 years of market data using `yfinance` and FRED.
* `run_master_backtest.py` - Master Command-Line Interface (CLI) for running backtests.

## Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
First, fetch the market features dataset:
```bash
python get_data.py --macro us
```

Then, run the master backtest for a specific model (e.g. `rslds`, `hdp_slds`, `hdp_arhmm`):
```bash
python run_master_backtest.py --model rslds --full
```

Results (including cumulative equity curves, regime weights, and extracted latent waves) will be saved in the `results/` folder.
