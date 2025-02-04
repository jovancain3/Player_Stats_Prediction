import statsapi
import json
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta

def fetch_schedule(start_date, end_date):
    schedule = statsapi.schedule(date=None, start_date=None, end_date=None, team="", opponent="", sportId=1, game_id=None)

    return schedule 

def format_schedule(schedule):
    
    new_dict = defaultdict(dict)
    
    for game in schedule:
        date = game['game_date']
        game_id = game['game_id']
        new_dict[date][game_id] = game
    
    # Print the result
    #print(json.dumps(new_dict, indent=2))
    
    return dict(new_dict)

def get_boxscore_data(schedule):
    print(schedule.keys())
    for k,v in schedule.items():
        print(v)
        for x,y in v.items():
            print(x)
            data = statsapi.boxscore_data(x, timecode=None)
            print(schedule[k][v])
            schedule[k][v]['game_data'] = data
            # statsapi.get('game_timestamps',{'gamePk':565997})
    print(len(schedule))
    return schedule


def fetch_player_stats(player_id, start_date, end_date):
    stats = statsapi.player_stat_data(
        personId=player_id,
        group="hitting",
        type="gameLog",
        sportId=1
    )
    return stats

def collect_player_data(player_ids, start_date, end_date):
    all_data = []
    for player_id in player_ids:
        stats = fetch_player_stats(player_id, start_date, end_date)
        for k,v in stats.items():
            print(f"Key:{k}")
        if 'stats' in stats and len(stats['stats']) > 0:
            all_stats = stats['stats']
            print(type(all_stats))
            print(all_stats[0])
            # for k in all_stats:
            #     print()
            #     print(f"Key:{k}")
            for game in stats['stats']['splits']:
                game_data = {
                    'player_id': player_id,
                    'game_date': game['date'],
                    'opponent': game['opponent']['id'],
                    'atBats': game['stat'].get('atBats', 0),
                    'hits': game['stat'].get('hits', 0),
                    'doubles': game['stat'].get('doubles', 0),
                    'triples': game['stat'].get('triples', 0),
                    'homeRuns': game['stat'].get('homeRuns', 0),
                    'rbi': game['stat'].get('rbi', 0),
                    'baseOnBalls': game['stat'].get('baseOnBalls', 0),
                    'strikeOuts': game['stat'].get('strikeOuts', 0),
                    'avg': game['stat'].get('avg', '.000'),
                    'obp': game['stat'].get('obp', '.000'),
                    'slg': game['stat'].get('slg', '.000'),
                    'ops': game['stat'].get('ops', '.000'),
                }
                all_data.append(game_data)
    return pd.DataFrame(all_data)

# Example usage
player_ids = [
    547180,  # Mookie Betts
    624413,  # Aaron Judge
    660670,  # Juan Soto
    665742,  # Vladimir Guerrero Jr.
    # Add more player IDs as needed
]
start_date = datetime(2023, 4, 1)  # Start of 2023 MLB season
end_date = datetime(2023, 10, 1)   # End of 2023 MLB regular season

schedule = fetch_schedule(start_date,end_date)
formatted_schedule = format_schedule(schedule)
get_boxscore_data(formatted_schedule)
#df = collect_player_data(player_ids, start_date, end_date)
#df.to_csv("mlb_player_hitting_stats_2023.csv", index=False)
#print(df.head())
#print(df.columns)