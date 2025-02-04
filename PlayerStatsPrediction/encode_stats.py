import pandas as pd
import json

def encode_nfl_stats(df, columns_to_encode):
    # Dictionary to store mappings
    mappings = {}
    
    df[columns_to_encode] = df[columns_to_encode].fillna('0')

    df.fillna(0, inplace=True)
    
    # Encode each specified column
    for column in columns_to_encode:
        if column in df.columns:
            # Get unique values and create a mapping
            unique_values = df[column].unique()
            value_map = {value: i for i, value in enumerate(unique_values)}
            
            # Apply mapping to the column
            df[column] = df[column].map(value_map)
            
            # Store mapping for this column
            mappings[column] = value_map
        else:
            print(f"Warning: Column '{column}' not found in the dataframe.")
    
    df.fillna(0, inplace=True)

    # Save mappings to a JSON file
    with open('nfl_stats_mappings.json', 'w') as f:
        json.dump(mappings, f, indent=4)
    
    return df

# Columns to encode
columns_to_encode = [
    'defensive_team', 'offensive_team', 'player_name', 
    'position', 'position_group', 'season_type','player_id'
]

# Assuming your dataframe is called 'nfl_data'
nfl_data = pd.read_csv('data/processed_nfl_data.csv')
encoded_data = encode_nfl_stats(nfl_data, columns_to_encode)
encoded_data.to_csv('data/encoded_nfl_data_3.csv', index=False)

# To load and use the mappings later:
# with open('nfl_stats_mappings.json', 'r') as f:
#     mappings = json.load(f)
# 
# # Example: Decode a value
# column = 'player_name'
# encoded_value = 42
# original_value = list(mappings[column].keys())[list(mappings[column].values()).index(encoded_value)]