import pandas as pd
import numpy as np

def aggregate_team_stats(input_file, output_file):
    # Read the CSV file
    df = pd.read_csv(input_file)

    # Identify columns to group by
    group_columns = ['team', 'week', 'season']

    # Identify numeric columns to aggregate, excluding 'week' and 'season'
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_columns = [col for col in numeric_columns if col not in group_columns]

    # Group by team, week, and season, then sum the numeric columns
    team_stats = df.groupby(group_columns)[numeric_columns].sum().reset_index()

    # Sort the results for better readability
    team_stats = team_stats.sort_values(['season', 'week', 'team'])

    # Save the result to a new CSV file
    team_stats.to_csv(output_file, index=False)
    print(f"Team stats have been saved to {output_file}")

    # Display the first few rows and column names of the result
    print("\nFirst few rows of aggregated data:")
    print(team_stats.head())
    
    print("\nColumns in the aggregated data:")
    print(team_stats.columns.tolist())

    # Print some summary statistics
    print("\nSummary of the data:")
    print(team_stats.describe())

    # Print the number of unique seasons, teams, and weeks
    print(f"\nNumber of unique seasons: {team_stats['season'].nunique()}")
    print(f"Number of unique teams: {team_stats['team'].nunique()}")
    print(f"Number of unique weeks: {team_stats['week'].nunique()}")
import pandas as pd

import pandas as pd

def join_csv_files(team_stats_file, opponent_data_file, output_file):
    # Read the CSV files
    team_stats = pd.read_csv(team_stats_file)
    opponent_data = pd.read_csv(opponent_data_file)

    # Rename the 'opponent_team' column in the opponent_data to 'team' for joining
    opponent_data = opponent_data.rename(columns={'opponent_team': 'team'})

    # Perform the join operation
    merged_data = pd.merge(team_stats, opponent_data, 
                           on=['team', 'week', 'season'], 
                           how='inner',
                           suffixes=('', '_opponent'))

    # Save the result to a new CSV file
    merged_data.to_csv(output_file, index=False)
    print(f"Merged data has been saved to {output_file}")

    # Display information about the merged dataset
    print("\nShape of the merged dataset:")
    print(merged_data.shape)

    print("\nColumns in the merged dataset:")
    print(merged_data.columns.tolist())

    print("\nFirst few rows of the merged dataset:")
    print(merged_data.head())

    # Check for any rows that didn't match
    total_rows = len(team_stats)
    merged_rows = len(merged_data)
    if total_rows != merged_rows:
        print(f"\nWarning: {total_rows - merged_rows} rows from team_stats were not matched.")
        print("This could be due to mismatched team, week, or season values.")

    # Print some summary statistics
    print("\nSummary of the merged data:")
    print(merged_data.describe())

    # Print unique values in key columns to help spot any mismatches
    print("\nUnique values in key columns:")
    for col in ['team', 'week', 'season']:
        print(f"\n{col}:")
        print(merged_data[col].unique())

import pandas as pd
import numpy as np

def calculate_weekly_averages(csv_file):
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Ensure the dataframe is sorted by player, season, and week
    df = df.sort_values(['season','week','player_display_name'])
    
    # Get the list of stat columns (assuming all numeric columns except 'player', 'season', and 'week' are stats)
    stat_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    stat_columns = [col for col in stat_columns if col not in ['week', 'season']]
    
    # Calculate cumulative sum and count for each player within each season
    df[stat_columns] = df.groupby(['player_display_name', 'season'])[stat_columns].cumsum()
    df['games_played'] = df.groupby(['player_display_name', 'season']).cumcount() + 1
    
    # Calculate averages
    for col in stat_columns:
        df[f'{col}_avg'] = df[col] / df['games_played']
    
    # Set the average for week 1 of each season to 0
    df.loc[df['week'] == 1, [f'{col}_avg' for col in stat_columns]] = 0

    # Save the result to a new CSV file
    df.to_csv('data/data.csv', index=False)
    
    return df

def process_nfl_stats(df):
    # Sort the dataframe
    df = df.sort_values(['season', 'week','player_name'])
    
    # Identify numeric columns that don't end with 'avg'
    numeric_cols = df.select_dtypes(include=[int, float]).columns
    cols_to_process = [col for col in numeric_cols if not col.endswith('avg')]
    
    # Group by season and player_name
    grouped = df.groupby(['season', 'player_name'])
    
    for col in cols_to_process:
        # Calculate the difference
        df[f'{col}_current'] = grouped[col].diff().fillna(df[col])
    
    return df

# Example usage
team_stats_file = 'data/team_stats_by_season.csv'
opponent_data_file = 'data/player_stats.csv'
output_file = 'data/merged_nfl_data.csv'


# Example usage
input_file = 'data/player_stats_def.csv'
opponent_output_file = 'data/team_stats_by_season.csv'

# Usage
file_path = 'data/merged_nfl_data.csv'

#aggregate_team_stats(input_file, opponent_output_file)
#join_csv_files(team_stats_file, opponent_data_file, output_file)
#calculate_weekly_averages(file_path)

# Assuming your dataframe is called 'nfl_data'
nfl_data = pd.read_csv('data/data.csv')
processed_data = process_nfl_stats(nfl_data)
processed_data.to_csv('data/processed_nfl_data.csv', index=False)