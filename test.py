import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import xgboost as xgb

def predict_player_stats(player_data):
    # Combine player, opponent, and historical data
    X = player_data
    
    # Define the target variable (e.g., 'yards', 'touchdowns', etc.)
    y = X['passing_yards']
    X = X.drop('passing_yards', axis=1)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
    dtest = xgb.DMatrix(X_test_scaled, label=y_test)
    
    # Set XGBoost parameters
    params = {
        'max_depth': 6,
        'eta': 0.1,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
    
    # Train XGBoost model
    num_rounds = 100
    model = xgb.train(params, dtrain, num_rounds)
    
    # Make predictions
    predictions = model.predict(dtest)
    
    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    print(f"Root Mean Squared Error: {rmse}")
    
    # Feature importance
    importance = model.get_score(importance_type='gain')
    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    print("\nTop 10 important features:")
    for feature, score in sorted_importance[:10]:
        print(f"{feature}: {score}")
    
    return model, scaler

# Example usage
player_data = pd.read_csv('data/encoded_nfl_data_2.csv')
model, scaler = predict_player_stats(player_data)

# To predict for a new game:
# new_game_data = pd.DataFrame(...)
# new_game_data_scaled = scaler.transform(new_game_data)
# dpredict = xgb.DMatrix(new_game_data_scaled)
# prediction = model.predict(dpredict)