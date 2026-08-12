import pandas as pd
from src.recurrent_slds import RecurrentSLDS
from src.initialization import initialize_rslds
from src.backtest import simulate_regime_strategy

print("Loading data...")
df = pd.read_csv("data/market_features.csv", parse_dates=["Date"])
dates = df["Date"].values
returns = df["nifty_log_return"].values
y_features = df[["nifty_log_return", "vix_change"]].values
y_scaled = (y_features - y_features.mean(axis=0)) / y_features.std(axis=0)

print("Initializing the rSLDS Pipeline (PCA -> AR-HMM -> Logistic Regression)...")
x_init, z_init, W_init, r_init = initialize_rslds(
    y=y_scaled,
    K=3,
    state_dim=2,
    ar_order=1,
    random_state=42
)

print("Training the Recurrent SLDS (Continuous waves predicting regimes!)...")
model = RecurrentSLDS(K=3, state_dim=2, obs_dim=2, pg_trunc=50, random_state=42)

# Set the pre-trained weights from initialization
model.W = W_init
model.r = r_init

# Run Gibbs sampling! (This is heavy math, it might take ~10 seconds)
history = model.gibbs(y_scaled, n_iters=100, burn_in=50, x_init=x_init, z_init=z_init)

print("\nRunning Backtest on rSLDS Regimes...")
inferred_regimes = history["last_z"]

results = simulate_regime_strategy(
    dates=dates,
    asset_returns=returns,
    inferred_regimes=inferred_regimes,
    save_path="results/rslds_backtest.png",
    show_plot=False
)

print(f"\nFinal Strategy Return: {results['strategy_metrics']['cumulative_return'] * 100:.2f}%")
print(f"Buy & Hold Return:   {results['buy_hold_metrics']['cumulative_return'] * 100:.2f}%")
print("\nSuccess! Open 'results/rslds_backtest.png' to see the chart!")
