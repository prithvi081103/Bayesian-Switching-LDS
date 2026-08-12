import pandas as pd
from src.hmm_baselines import GaussianHMM

# 1. Load the data we built in Phase 1
df = pd.read_csv("data/market_features.csv")
# Use Nifty returns and VIX changes as our two features (D=2)
y = df[["nifty_log_return", "vix_change"]].values

# 2. Standardize the data (mean=0, std=1) so the Gaussian math is stable
y = (y - y.mean(axis=0)) / y.std(axis=0)

# 3. Create and train the model (let's assume 3 regimes: Bull, Bear, Neutral)
print("Training HMM with 3 regimes...")
model = GaussianHMM(K=3, D=2)
model.fit(y, n_iters=50)

# 4. Print the learned parameters!
print("\n--- Training Complete! ---")
for k in range(3):
    print(f"\nRegime {k}:")
    print(f"  Nifty Return Mean (Scaled): {model.mu[k][0]:.4f}")
    print(f"  VIX Change Mean (Scaled):   {model.mu[k][1]:.4f}")
