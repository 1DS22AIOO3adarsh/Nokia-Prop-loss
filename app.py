# pathloss_all_models_pipeline_ml_only_clean.py
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Classical ML Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

# Load Data
df = pd.read_csv("synthetic_pathloss_noisy.csv")

X = df[['Tx_X','Tx_Y','Tx_Z','Rx_X','Rx_Y','Rx_Z',
        'Distance_3D','LOS','Building_Height_Avg','Azimuth_Deg','Elevation_Deg']]
y = df['Path_Loss_dB']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

results = []

# Utility
def evaluate(name, y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    results.append({"Model": name, "RMSE": rmse, "R2": r2})
    print(f"{name:<25} | RMSE: {rmse:.3f} | R²: {r2:.3f}")

print("\nTraining Classical ML Models\n")

models_ml = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "ElasticNet": ElasticNet(),
    "Decision Tree": DecisionTreeRegressor(max_depth=10),
    "Random Forest": RandomForestRegressor(n_estimators=300),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=300),
    "AdaBoost": AdaBoostRegressor(n_estimators=300),
    "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6),
    "CatBoost": CatBoostRegressor(iterations=300, learning_rate=0.05, depth=8, verbose=0),
}

for name, model in models_ml.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    evaluate(name, y_test, y_pred)

# Visualize Results
results_df = pd.DataFrame(results).sort_values(by="RMSE")
print("\nSummary of All Models:")
print(results_df)

plt.figure(figsize=(10,8))
sns.barplot(x="RMSE", y="Model", data=results_df, palette="viridis")
plt.title("RMSE Comparison (Lower is Better)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10,8))
sns.barplot(x="R2", y="Model", data=results_df, palette="coolwarm")
plt.title("R² Comparison (Higher is Better)")
plt.tight_layout()
plt.show()

results_df.to_csv("all_models_comparison.csv", index=False)
print("\nResults saved as 'all_models_comparison.csv'")
