import pandas as pd
from src.hmm_baselines import GaussianHMM
from src.backtest import simulate_regime_strategy

# 1. Load Data
df = pd.read_csv("data/market_features.csv", parse_dates=["Date"])
dates = df["Date"].values

# Use Nifty returns for the backtest PnL, and scaled features for the HMM
returns = df["nifty_log_return"].values
y_features = df[["nifty_log_return", "vix_change"]].values
y_scaled = (y_features - y_features.mean(axis=0)) / y_features.std(axis=0)

# 2. Train HMM
print("Training HMM...")
model = GaussianHMM(K=3, D=2)
model.fit(y_scaled, n_iters=50)

# 3. Get the "most likely" regimes for each day
# We look at the 'gamma' probabilities (from your Forward-Backward algorithm) 
# and pick the regime with the highest probability (argmax)
gamma, _ = model._forward_backward(y_scaled)
inferred_regimes = gamma.argmax(axis=1)

# 4. Run Backtest
print("Running Strategy Backtest...")
results = simulate_regime_strategy(
    dates=dates,
    asset_returns=returns,
    inferred_regimes=inferred_regimes,
    save_path="results/baseline_backtest.png",
    show_plot=False
)

print(f"\nFinal Strategy Return: {results['strategy_metrics']['cumulative_return'] * 100:.2f}%")
print(f"Buy & Hold Return:   {results['buy_hold_metrics']['cumulative_return'] * 100:.2f}%")
print("\nSuccess! Open the 'results/baseline_backtest.png' file to see your chart!")
