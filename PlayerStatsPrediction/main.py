import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
import warnings
warnings.filterwarnings('ignore')

class NFLStatPredictor:
    def __init__(self):
        self.features = None
        self.target = None
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def prepare_data(self, df):
        """
        Prepare data with focus on player historical performance and defensive matchups
        """
        # Separate features that need different preprocessing
        categorical_features = ['defensive_team', 'position', 'position_group', 'offensive_team']
        
        # Create player-specific rolling averages with different windows
        player_stats = [
            'rushing_yards', 'carries', 'rushing_tds',
            'receiving_yards', 'targets', 'receptions', 'receiving_tds',
            'passing_yards', 'attempts', 'completions', 'passing_tds',
            'fantasy_points'
        ]
        
        # Calculate rolling averages for different time windows
        windows = [3, 5, 10]
        for stat in player_stats:
            if stat in df.columns:
                for window in windows:
                    df[f'{stat}_last_{window}'] = (
                        df.groupby('player_id')[stat]
                        .rolling(window=window, min_periods=1)
                        .mean()
                        .reset_index(0, drop=True)
                    )
        
        # Create defensive team statistics (against each position)
        defensive_stats = [
            'rushing_yards_allowed', 'rushing_tds_allowed',
            'receiving_yards_allowed', 'receiving_tds_allowed',
            'passing_yards_allowed', 'passing_tds_allowed',
            'sacks', 'interceptions'
        ]
        
        for stat in defensive_stats:
            if stat in df.columns:
                # Calculate defensive rolling averages
                df[f'def_{stat}_last_3'] = (
                    df.groupby('defensive_team')[stat]
                    .rolling(window=3, min_periods=1)
                    .mean()
                    .reset_index(0, drop=True)
                )
        
        # Create matchup-based features
        if all(col in df.columns for col in ['rushing_yards_last_3', 'def_rushing_yards_allowed_last_3']):
            df['rushing_matchup_rating'] = (
                df['rushing_yards_last_3'] / (df['def_rushing_yards_allowed_last_3'] + 1)
            )
        else:
            df['rushing_matchup_rating'] = 0  # or another default value

        if all(col in df.columns for col in ['receiving_yards_last_3', 'def_receiving_yards_allowed_last_3']):
            df['receiving_matchup_rating'] = (
                df['receiving_yards_last_3'] / (df['def_receiving_yards_allowed_last_3'] + 1)
            )
        else:
            df['receiving_matchup_rating'] = 0  # or another default value
        
        # Add game context features
        # df['home_game'] = (df['offensive_team'] == df['home_team']).astype(int)
        # df['divisional_game'] = (df['offensive_division'] == df['defensive_division']).astype(int)
        df['season_progress'] = df['week'] / 18  # Updated for 18-week season
        
        # Weather features (if available)
        weather_features = ['temperature', 'wind_speed', 'precipitation']
        
        # Label encode categorical variables
        for cat_feature in categorical_features:
            if cat_feature in df.columns:
                self.label_encoders[cat_feature] = LabelEncoder()
                df[cat_feature] = self.label_encoders[cat_feature].fit_transform(df[cat_feature].astype(str))
        
        # Handle missing values
        df = df.fillna(0)
        
        return df

    def train_model(self, df, target_variable, exclude_features=None):
        """
        Train an improved XGBoost model with better hyperparameter optimization
        """
        if exclude_features is None:
            exclude_features = []
            
        # Add player_id and date-related columns to exclude_features
        exclude_features.extend(['player_id', 'player_name', 'game_id', 'date', 'week', 'season'])
        
        # Prepare features and target
        features = [col for col in df.columns if col not in exclude_features + [target_variable]]
        X = df[features]
        y = df[target_variable]
        
        # Clean the data
        initial_rows = len(y)
        mask = ~(np.isnan(y) | np.isinf(y) | (np.abs(y) > 1e6))
        X = X[mask]
        y = y[mask]
        filtered_rows = len(y)
        print(f"Filtered {initial_rows - filtered_rows} rows with invalid values")
        
        # Split the data with consideration for time series nature
        train_idx = int(len(X) * 0.8)
        X_train, X_test = X[:train_idx], X[train_idx:]
        y_train, y_test = y[:train_idx], y[train_idx:]
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define improved hyperparameter space
        space = {
            'max_depth': hp.choice('max_depth', range(3, 12)),
            'learning_rate': hp.loguniform('learning_rate', np.log(0.005), np.log(0.3)),
            'n_estimators': hp.choice('n_estimators', range(100, 2000, 100)),
            'min_child_weight': hp.choice('min_child_weight', range(1, 7)),
            'gamma': hp.uniform('gamma', 0, 0.5),
            'subsample': hp.uniform('subsample', 0.6, 1.0),
            'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0),
            'reg_alpha': hp.loguniform('reg_alpha', np.log(1e-6), np.log(1.0)),
            'reg_lambda': hp.loguniform('reg_lambda', np.log(1e-6), np.log(1.0))
        }
        
        # Objective function for hyperparameter optimization
        def objective(params):
            model = xgb.XGBRegressor(
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                n_estimators=params['n_estimators'],
                min_child_weight=params['min_child_weight'],
                gamma=params['gamma'],
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                reg_alpha=params['reg_alpha'],
                reg_lambda=params['reg_lambda'],
                random_state=42
            )
            
            # Use time series cross-validation
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = []
            
            for train_idx, val_idx in tscv.split(X_train_scaled):
                X_train_cv, X_val_cv = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_train_cv, y_val_cv = y_train.iloc[train_idx], y_train.iloc[val_idx]
                
                model.fit(X_train_cv, y_train_cv)
                pred = model.predict(X_val_cv)
                score = mean_squared_error(y_val_cv, pred, squared=False)  # RMSE
                cv_scores.append(score)
            
            return {'loss': np.mean(cv_scores), 'status': STATUS_OK}
        
        # Run hyperparameter optimization
        trials = Trials()
        best = fmin(fn=objective,
                   space=space,
                   algo=tpe.suggest,
                   max_evals=25,  # Increased number of evaluations
                   trials=trials)
        
        # Train final model with best parameters
        self.model = xgb.XGBRegressor(
            max_depth=best['max_depth'],
            learning_rate=best['learning_rate'],
            n_estimators=best['n_estimators'],
            min_child_weight=best['min_child_weight'],
            gamma=best['gamma'],
            subsample=best['subsample'],
            colsample_bytree=best['colsample_bytree'],
            reg_alpha=best['reg_alpha'],
            reg_lambda=best['reg_lambda'],
            random_state=42
        )
        
        # Fit the model
        self.model.fit(
            X_train_scaled, 
            y_train,
            eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
            # early_stopping_rounds=50,
            verbose=100
        )
        
        # Make predictions and evaluate
        train_predictions = self.model.predict(X_train_scaled)
        test_predictions = self.model.predict(X_test_scaled)
        
        # Create comparison DataFrame for test predictions
        comparison_df = pd.DataFrame({
            'Actual': y_test,
            'Predicted': test_predictions,
            'Difference': y_test - test_predictions,
            'Percent_Error': abs((y_test - test_predictions) / y_test) * 100
        })
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, train_predictions)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, test_predictions)),
            'train_mae': mean_absolute_error(y_train, train_predictions),
            'test_mae': mean_absolute_error(y_test, test_predictions),
            'train_r2': r2_score(y_train, train_predictions),
            'test_r2': r2_score(y_test, test_predictions)
        }
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return metrics, feature_importance, comparison_df
    
    def predict(self, new_data):
        """
        Make predictions on new data
        """
        # Prepare new data using the same preprocessing steps
        prepared_data = self.prepare_data(new_data.copy())
        scaled_data = self.scaler.transform(prepared_data[self.features])
        return self.model.predict(scaled_data)

# Example usage
def main():
    # Load data
    df = pd.read_csv('data/encoded_nfl_data_filled_30MB.csv')
    
    # Initialize predictor
    predictor = NFLStatPredictor()
    
    # Prepare data
    df_prepared = predictor.prepare_data(df)
    
    # Define target variables to predict (can be changed based on needs)
    # ['fantasy_points', 'rushing_yards', 'receiving_yards', 'passing_yards']
    target_variables = ['fantasy_points']
    
    # Train separate models for each target variable
    results = {}
    for target in target_variables:
        print(f"\nTraining model for {target}")
        exclude_features = ['player_id', 'player_name','week', 'season'] + target_variables
        metrics, feature_importance, comparison_df = predictor.train_model(df_prepared, target, exclude_features)
        
        results[target] = {
            'metrics': metrics,
            'top_features': feature_importance.head(10),
            'predictions': comparison_df
        }
        
        print(f"Test RMSE for {target}: {metrics['test_rmse']:.2f}")
        print(f"Test R² for {target}: {metrics['test_r2']:.3f}")
        print("\nTop 5 important features:")
        print(feature_importance.head().to_string())
        
        print("\nPrediction Analysis:")
        print("\nSummary Statistics:")
        print(comparison_df.describe())
        print("\nSample of Predictions (first 10 rows):")
        print(comparison_df.head(10).round(2))
        
        # Optional: Add visualization
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            plt.scatter(comparison_df['Actual'], comparison_df['Predicted'], alpha=0.5)
            plt.plot([comparison_df['Actual'].min(), comparison_df['Actual'].max()], 
                    [comparison_df['Actual'].min(), comparison_df['Actual'].max()], 
                    'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'Actual vs Predicted {target}')
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib not installed. Skipping visualization.")

if __name__ == "__main__":
    main()