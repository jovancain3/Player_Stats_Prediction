import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os

# ==========================
# 1. Data Loading
# ==========================

# Replace 'data.csv' with your actual file path
file_path = 'data/data.csv'

# Check if the file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The file {file_path} does not exist. Please check the path.")

# Load the dataset
print("Loading data...")
data = pd.read_csv(file_path)

print(f"Data loaded successfully with shape: {data.shape}")

# ==========================
# 2. Data Preprocessing
# ==========================

# 2.1. Handle Missing Values
print("\nHandling missing values...")

# Separate numeric and categorical columns
numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = data.select_dtypes(include=['object']).columns

# Fill missing numeric values with median
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Fill missing categorical values with a new category 'Unknown'
data[categorical_cols] = data[categorical_cols].fillna('Unknown')

print("Missing values handled.")

# 2.2. Encode Categorical Variables
print("\nEncoding categorical variables...")

# List of categorical columns to encode
categorical_columns = ['player_name', 'player_id', 'position', 'position_group', 'offensive_team', 'defensive_team', 'season_type']

# Initialize LabelEncoders for each categorical column
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col}.")

print("Categorical variables encoded.")

# 2.3. Feature Selection

# Define input and output columns as per your specification
input_columns = [
    'week', 'games_played', 'season', 'season_type', 'player_id', 'player_name', 'position',
    'position_group', 'offensive_team', 'defensive_team', 'def_tackles_avg', 'def_tackles_for_loss_avg',
    'def_tackles_for_loss_yards_avg', 'def_fumbles_forced_avg', 'def_sacks_avg', 'def_sack_yards_avg', 
    'def_qb_hits_avg', 'def_interceptions_avg', 'def_interception_yards_avg', 'def_pass_defended_avg', 
    'def_tds_avg', 'def_fumbles_avg', 'def_safety_avg', 'def_penalty_avg', 'def_penalty_yards_avg', 
    'completions_avg', 'attempts_avg', 'passing_yards_avg', 'passing_tds_avg', 'interceptions_avg', 
    'sacks_avg', 'sack_yards_avg', 'passing_air_yards_avg', 'passing_yards_after_catch_avg', 
    'passing_first_downs_avg', 'passing_epa_avg', 'passing_2pt_conversions_avg', 'pacr_avg', 'dakota_avg', 
    'carries_avg', 'rushing_yards_avg', 'rushing_tds_avg', 'rushing_fumbles_avg', 'rushing_fumbles_lost_avg', 
    'rushing_first_downs_avg', 'rushing_epa_avg', 'rushing_2pt_conversions_avg', 'receptions_avg', 
    'targets_avg', 'receiving_yards_avg', 'receiving_tds_avg', 'receiving_fumbles_avg', 
    'receiving_fumbles_lost_avg', 'receiving_air_yards_avg', 'receiving_yards_after_catch_avg', 
    'receiving_first_downs_avg', 'receiving_epa_avg', 'receiving_2pt_conversions_avg', 'racr_avg', 
    'target_share_avg', 'air_yards_share_avg', 'wopr_avg', 'special_teams_tds_avg', 'fantasy_points_avg', 
    'fantasy_points_ppr_avg'
]

output_columns = [
    'def_tackles', 'def_tackles_for_loss', 'def_tackles_for_loss_yards', 'def_fumbles_forced', 'def_sacks',
    'def_sack_yards', 'def_qb_hits', 'def_interceptions', 'def_interception_yards', 'def_pass_defended', 
    'def_tds', 'def_fumbles', 'def_safety', 'def_penalty', 'def_penalty_yards', 'completions', 'attempts', 
    'passing_yards', 'passing_tds', 'interceptions', 'sacks', 'sack_yards', 'passing_air_yards', 
    'passing_yards_after_catch', 'passing_first_downs', 'passing_epa', 'passing_2pt_conversions', 'pacr', 
    'dakota', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost', 
    'rushing_first_downs', 'rushing_epa', 'rushing_2pt_conversions', 'receptions', 'targets', 
    'receiving_yards', 'receiving_tds', 'receiving_fumbles', 'receiving_fumbles_lost', 'receiving_air_yards', 
    'receiving_yards_after_catch', 'receiving_first_downs', 'receiving_epa', 'receiving_2pt_conversions', 
    'racr', 'target_share', 'air_yards_share', 'wopr', 'special_teams_tds', 'fantasy_points', 
    'fantasy_points_ppr', 'games_played'
]

# Ensure all input and output columns exist in the dataset
missing_input_cols = set(input_columns) - set(data.columns)
missing_output_cols = set(output_columns) - set(data.columns)

if missing_input_cols:
    raise ValueError(f"The following input columns are missing from the dataset: {missing_input_cols}")

if missing_output_cols:
    raise ValueError(f"The following output columns are missing from the dataset: {missing_output_cols}")

# Split the data into inputs (X) and outputs (y)
X = data[input_columns]
y = data[output_columns]

print("\nFeature selection completed.")
print(f"Input features shape: {X.shape}")
print(f"Output features shape: {y.shape}")

# 2.4. Feature Scaling (Optional but recommended for some models)
print("\nScaling numerical features...")

# Initialize a scaler
scaler = StandardScaler()

# Fit and transform the input features
X_scaled = scaler.fit_transform(X)

print("Feature scaling completed.")

# ==========================
# 3. Train-Test Split
# ==========================

print("\nSplitting data into training and testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# ==========================
# 4. Model Training
# ==========================

print("\nTraining the model...")

# Initialize the base regressor
base_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Initialize the multi-output regressor
multi_output_regressor = MultiOutputRegressor(base_regressor, n_jobs=-1)

# Train the model
multi_output_regressor.fit(X_train, y_train)

print("Model training completed.")

# ==========================
# 5. Evaluation
# ==========================

print("\nEvaluating the model...")

# Predict on the test set
y_pred = multi_output_regressor.predict(X_test)

# Calculate evaluation metrics for each output
mse = mean_squared_error(y_test, y_pred, multioutput='raw_values')
mae = mean_absolute_error(y_test, y_pred, multioutput='raw_values')
r2 = r2_score(y_test, y_pred, multioutput='raw_values')

# Create a DataFrame to display the metrics
metrics_df = pd.DataFrame({
    'Output_Variable': y.columns,
    'MSE': mse,
    'MAE': mae,
    'R2_Score': r2
})

print("\nEvaluation Metrics:")
print(metrics_df)

# Optionally, calculate average metrics
average_mse = np.mean(mse)
average_mae = np.mean(mae)
average_r2 = np.mean(r2)

print("\nAverage Metrics:")
print(f"Mean Squared Error (MSE): {average_mse:.4f}")
print(f"Mean Absolute Error (MAE): {average_mae:.4f}")
print(f"R-squared (R2): {average_r2:.4f}")

# ==========================
# 6. Model Saving
# ==========================

print("\nSaving the trained model and scaler...")

# Create a directory to save models if it doesn't exist
model_dir = 'saved_models'
os.makedirs(model_dir, exist_ok=True)

# Save the multi-output regressor
model_path = os.path.join(model_dir, 'multi_output_regressor.joblib')
joblib.dump(multi_output_regressor, model_path)

# Save the scaler
scaler_path = os.path.join(model_dir, 'scaler.joblib')
joblib.dump(scaler, scaler_path)

# Save the label encoders
encoders_path = os.path.join(model_dir, 'label_encoders.joblib')
joblib.dump(label_encoders, encoders_path)

print(f"Model saved to {model_path}")
print(f"Scaler saved to {scaler_path}")
print(f"Label encoders saved to {encoders_path}")

# ==========================
# 7. (Optional) Loading and Using the Model
# ==========================

# Example of how to load and use the saved model
def load_model_and_predict(input_data):
    """
    Load the saved model, scaler, and label encoders to make predictions on new data.
    
    Parameters:
    - input_data (pd.DataFrame): New data with the same structure as the training inputs.
    
    Returns:
    - predictions (np.ndarray): Predicted outputs.
    """
    # Load the saved components
    loaded_model = joblib.load(model_path)
    loaded_scaler = joblib.load(scaler_path)
    loaded_encoders = joblib.load(encoders_path)
    
    # Handle missing values
    input_data = input_data.copy()
    input_data[numeric_cols] = input_data[numeric_cols].fillna(input_data[numeric_cols].median())
    input_data[categorical_cols] = input_data[categorical_cols].fillna('Unknown')
    
    # Encode categorical variables
    for col in categorical_columns:
        le = loaded_encoders[col]
        input_data[col] = le.transform(input_data[col].astype(str))
    
    # Scale the input features
    input_scaled = loaded_scaler.transform(input_data[input_columns])
    
    # Make predictions
    predictions = loaded_model.predict(input_scaled)
    
    return predictions

# Example usage:
new_data = pd.read_csv('data/encoded_nfl_data_2.csv')  # Replace with your new data file
predictions = load_model_and_predict(new_data)
print(predictions)
