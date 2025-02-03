import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt


#########################
### DATA PREPARATION ###
#########################

# Load the dataset
file_path = 'data/processed_nfl_data.csv'
data = pd.read_csv(file_path)

data = data[data['position'] == 'QB']

# Fill missing values with median for numeric columns
data.fillna(data.median(numeric_only=True), inplace=True)

# Categorical columns
categorical_columns = ['player_name', 'player_id', 'position', 'position_group', 'offensive_team', 'defensive_team', 'season_type']

# Encoding categorical columns
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Define input and output columns as per your specification
# input_columns = [
#     'week', 'games_played', 'season', 'season_type', 'player_id', 'player_name', 'position',
#     'position_group', 'offensive_team', 'defensive_team', 'def_tackles_avg', 'def_tackles_for_loss_avg',
#     'def_tackles_for_loss_yards_avg', 'def_fumbles_forced_avg', 'def_sacks_avg', 'def_sack_yards_avg', 
#     'def_qb_hits_avg', 'def_interceptions_avg', 'def_interception_yards_avg', 'def_pass_defended_avg', 
#     'def_tds_avg', 'def_fumbles_avg', 'def_safety_avg', 'def_penalty_avg', 'def_penalty_yards_avg', 
#     'completions_avg', 'attempts_avg', 'passing_yards_avg', 'passing_tds_avg', 'interceptions_avg', 
#     'sacks_avg', 'sack_yards_avg', 'passing_air_yards_avg', 'passing_yards_after_catch_avg', 
#     'passing_first_downs_avg', 'passing_epa_avg', 'passing_2pt_conversions_avg', 'pacr_avg', 'dakota_avg', 
#     'carries_avg', 'rushing_yards_avg', 'rushing_tds_avg', 'rushing_fumbles_avg', 'rushing_fumbles_lost_avg', 
#     'rushing_first_downs_avg', 'rushing_epa_avg', 'rushing_2pt_conversions_avg', 'receptions_avg', 
#     'targets_avg', 'receiving_yards_avg', 'receiving_tds_avg', 'receiving_fumbles_avg', 
#     'receiving_fumbles_lost_avg', 'receiving_air_yards_avg', 'receiving_yards_after_catch_avg', 
#     'receiving_first_downs_avg', 'receiving_epa_avg', 'receiving_2pt_conversions_avg', 'racr_avg', 
#     'target_share_avg', 'air_yards_share_avg', 'wopr_avg', 'special_teams_tds_avg', 'fantasy_points_avg', 
#     'fantasy_points_ppr_avg'
# ]
input_columns = [
    'week', 'games_played', 'player_id', 'player_name', 'position',
    'offensive_team', 'defensive_team',
    'completions_avg', 'attempts_avg', 'passing_yards_avg', 'passing_tds_avg', 'interceptions_avg', 
    'fantasy_points_avg'
]

# output_columns = [
#     'def_tackles', 'def_tackles_for_loss', 'def_tackles_for_loss_yards', 'def_fumbles_forced', 'def_sacks',
#     'def_sack_yards', 'def_qb_hits', 'def_interceptions', 'def_interception_yards', 'def_pass_defended', 
#     'def_tds', 'def_fumbles', 'def_safety', 'def_penalty', 'def_penalty_yards', 'completions', 'attempts', 
#     'passing_yards', 'passing_tds', 'interceptions', 'sacks', 'sack_yards', 'passing_air_yards', 
#     'passing_yards_after_catch', 'passing_first_downs', 'passing_epa', 'passing_2pt_conversions', 'pacr', 
#     'dakota', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_fumbles', 'rushing_fumbles_lost', 
#     'rushing_first_downs', 'rushing_epa', 'rushing_2pt_conversions', 'receptions', 'targets', 
#     'receiving_yards', 'receiving_tds', 'receiving_fumbles', 'receiving_fumbles_lost', 'receiving_air_yards', 
#     'receiving_yards_after_catch', 'receiving_first_downs', 'receiving_epa', 'receiving_2pt_conversions', 
#     'racr', 'target_share', 'air_yards_share', 'wopr', 'special_teams_tds', 'fantasy_points', 
#     'fantasy_points_ppr', 'games_played'
# ]
output_columns = [ 
    'week', 'player_name',
    'passing_yards_current', 'passing_tds_current', 'interceptions_current'
]

X = data[input_columns]
y = data[output_columns]

# Normalize the data using MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Reshape X to be 3D: (samples, timesteps, features)
X_reshaped = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

#########################
   ### BUILD LSTM ###
#########################

# Define the LSTM model
model = Sequential()

# Add LSTM layer - input_shape is (time_steps, features)
model.add(LSTM(units=64, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))

# Add a Dense output layer (you can adjust the number of units based on your output)
model.add(Dense(y_train.shape[1]))  # Number of outputs should match the number of target columns

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Summary of the model
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')

# Make predictions
y_pred = model.predict(X_test)

# Compare predictions with actual values
print(y_pred[:5])  # Show first 5 predictions
print(y_test[:5])  # Compare with first 5 true values

# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error (MAE): {mae}')

# Calculate Root Mean Squared Error (RMSE)
rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'Root Mean Squared Error (RMSE): {rmse}')

# Calculate Mean Absolute Percentage Error (MAPE)
def mean_absolute_percentage_error(y_true, y_pred): 
    epsilon = 1e-10
    y_true = np.where(y_true == 0, epsilon, y_true)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

mape = mean_absolute_percentage_error(y_test, y_pred)
print(f'Mean Absolute Percentage Error (MAPE): {mape}%')

def smape(y_true, y_pred):
    return 100/len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# Calculate SMAPE
smape_value = smape(y_test, y_pred)
print(f'SMAPE: {smape_value}%')

# Plot the loss
# Plot true vs predicted values
# plt.scatter(y_test, y_pred)
# plt.xlabel("True Values")
# plt.ylabel("Predicted Values")
# plt.title("True vs Predicted Values")
# plt.show()

# # Residuals (y_true - y_pred)
# residuals = y_test - y_pred

# # Plot residuals
# plt.hist(residuals, bins=50)
# plt.xlabel("Residual")
# plt.ylabel("Frequency")
# plt.title("Distribution of Residuals")
# plt.show()

# # Plot distribution of actual values (y_test)
# plt.hist(y_test, bins=50)
# plt.xlabel("True Values")
# plt.ylabel("Frequency")
# plt.title("Distribution of True Values")
# plt.show()

# plt.plot(history.history['loss'], label='train')
# plt.plot(history.history['val_loss'], label='test')
# plt.title('Model Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()