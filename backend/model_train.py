import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Load dataset
df = pd.read_csv('../dataset/dataset.csv')

# Features and target
X = df[['past_score', 'subjects', 'last_week_hours', 'stress_level', 'sleep_hours', 'target_score']]
y = df['recommended_study_hours']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(random_state=42)
}

best_mse = float('inf')
best_model = None

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{name} -> MSE: {mse:.2f}, R2: {r2:.2f}")
    
    if mse < best_mse:
        best_mse = mse
        best_model = model
        best_model_name = name

print(f"Best Model: {best_model_name} with MSE: {best_mse:.2f}")

# Optional hyperparameter tuning for Random Forest or Gradient Boosting
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    if best_model_name == 'Random Forest':
        param_grid = {'n_estimators':[100,200],'max_depth':[None,10,20],'min_samples_split':[2,5]}
        grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3)
    else:
        param_grid = {'n_estimators':[100,200],'learning_rate':[0.05,0.1],'max_depth':[3,5]}
        grid = GridSearchCV(GradientBoostingRegressor(random_state=42), param_grid, cv=3)
    grid.fit(X_train_scaled, y_train)
    best_model = grid.best_estimator_
    print("Best hyperparameters:", grid.best_params_)

# Save model and scaler
joblib.dump(best_model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Model and scaler saved!")
