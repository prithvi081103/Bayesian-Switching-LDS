import pandas as pd
from src.hdp_arhmm import StickyHDPARHMM
from src.backtest import simulate_regime_strategy

print("Loading data...")
df = pd.read_csv("data/market_features.csv", parse_dates=["Date"])
dates = df["Date"].values
returns = df["nifty_log_return"].values
y_features = df[["nifty_log_return", "vix_change"]].values
y_scaled = (y_features - y_features.mean(axis=0)) / y_features.std(axis=0)

# Set L=10 maximum regimes!
print("Training HDP-AR-HMM with a maximum of L=10 regimes...")
model = StickyHDPARHMM(L=10, D=2, ar_order=1, random_state=42)

# Run Gibbs sampling (This will take a few seconds because it's doing heavy math!)
history = model.gibbs(y_scaled, n_iters=100, burn_in=50)

print("\n--- Regime Discovery ---")
print("Notice how the active number of regimes changes as the HDP deletes unneeded ones:")
print(history["num_active"])

print("\nRunning Backtest on HDP Regimes...")
# Use the final simulated regime timeline
inferred_regimes = history["last_z"]

results = simulate_regime_strategy(
    dates=dates,
    asset_returns=returns,
    inferred_regimes=inferred_regimes,
    save_path="results/hdp_backtest.png",
    show_plot=False
)

print(f"\nFinal Strategy Return: {results['strategy_metrics']['cumulative_return'] * 100:.2f}%")
print(f"Buy & Hold Return:   {results['buy_hold_metrics']['cumulative_return'] * 100:.2f}%")
print("\nSuccess! Open 'results/hdp_backtest.png' to see the chart!")
