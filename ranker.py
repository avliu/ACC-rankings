from sklearn.linear_model import LinearRegression
import pandas as pd


"""

 METHODOLOGY:
 
    The routine written below tries to answer the question:
    
        If team 'x' played against a hypothetical team representing the average non-'x' ACC team, how would it perform?
    
    1. Train a linear model to predict score differential, given the game's stats
    2. Collect each team's average home and away stats throughout the season
    3. For each team 'x', feed the model a list of stats representing the hypothetical game:
        a. The home team's stats would be the team x's home averages for the season
        b. The away team's stats would be the averages of all teams' away averages EXCEPT for team x's
        c. Repeat with the home and away teams flipped
        d. Combine the hypothetical home game and the hypothetical away game to produce a rating

    Edits made to 'ACCGames1819.csv':
        For each game, I estimated the number of total possessions by adding FGA + TOV for both teams.
        For each game, I added a 'Target' column representing the score differential for the Away Team (2 if Away Team 
        wins by 2, -2 if Away Team loses by 2). It represents the target value the model wants to predict.
        
    Additional preprocessing:
        In the script, I deleted the columns for 'GameDate', 'NeutralSite', 'AwayScore' and 'HomeScore'.
        I divided each stat by the estimated number of possessions for that team.
        I then normalized each stat by subtracting from the mean and dividing by the standard deviation.
        
"""


data = pd.read_csv(open("ACCGames1819.csv", "r"), delimiter=",")

"""
 PREPROCESSING - TRAINING DATA 
"""

data_train = data.copy()
data_train = data_train.drop('GameDate', axis=1)
data_train = data_train.drop('NeutralSite', axis=1)
data_train = data_train.drop('AwayTeam', axis=1)
data_train = data_train.drop('HomeTeam', axis=1)
data_train = data_train.drop('AwayScore', axis=1)
data_train = data_train.drop('HomeScore', axis=1)

feature_length = 28

# divide each stat by the amount of estimated possessions for that game
for i in range(0, feature_length):
    data_train.iloc[:, i] = data_train.iloc[:, i] / (data_train.iloc[:, feature_length] / 2)
data_train.iloc[:, feature_length+1] = data_train.iloc[:, feature_length+1] / (data_train.iloc[:, feature_length] / 2)

data_train = data_train.drop('TotalPoss', axis=1)

# normalize each column
data_train = (data_train-data_train.mean())/data_train.std()

# features as np array
train_features = data_train.iloc[:, :feature_length]
print('Games to train on: ')
print(train_features.head())
print('\n')
train_features = train_features.values

# labels as np array
train_labels = data_train.iloc[:, feature_length]
print('Score differentials to train on: ')
print(train_labels.head())
print('\n')
train_labels = train_labels.values


"""
 PREPROCESSING - TEST DATA 
"""

data_test = data.copy()
data_test = data_test.drop('GameDate', axis=1)
data_test = data_test.drop('NeutralSite', axis=1)
data_test = data_test.drop('AwayScore', axis=1)
data_test = data_test.drop('HomeScore', axis=1)
away_teams = data_test['AwayTeam']
home_teams = data_test['HomeTeam']
data_test = data_test.drop('AwayTeam', axis=1)
data_test = data_test.drop('HomeTeam', axis=1)
data_test = data_test.drop('Target', axis=1)

# divide each stat by the amount of estimated possessions for that game
for i in range(0, feature_length):
    data_test.iloc[:, i] = data_test.iloc[:, i] / (data_test.iloc[:, feature_length] / 2)

data_test = data_test.drop('TotalPoss', axis=1)

# normalize each column
data_test = (data_test-data_test.mean())/data_test.std()

data_test['AwayTeam'] = away_teams
data_test['HomeTeam'] = home_teams

team_averages = pd.DataFrame(index = data_train.columns)
teams = data_test['AwayTeam'].unique()

# for each team, collect its average for each stat
for i in range(0, len(teams)):
    away_rows = data_test[data_test['AwayTeam'].isin([teams[i]])]
    away_average = away_rows.mean()
    home_rows = data_test[data_test['HomeTeam'].isin([teams[i]])]
    home_average = home_rows.mean()
    team_averages[teams[i]] = away_average[:int(feature_length/2)].append(home_average[int(feature_length/2):])

team_averages = team_averages.transpose()
team_averages = team_averages.drop('Target', axis=1)

print('Average stats for each team: ')
print(team_averages.head())
print('\n')


"""
 TRAINING MODEL
"""

model = LinearRegression()
model.fit(train_features, train_labels)

"""
 PREDICTING WITH MODEL
"""

# for each team, predict the final score differential of a game between that team and the averages of all other teams
# for each team, repeat above for both a home game and away game
# use the sum of the differentials as the prediction rating
ratings = pd.Series(index=teams)
for i in range(0, len(teams)):
    away_game = (team_averages.iloc[i, :int(feature_length/2)].append(team_averages.iloc[:, int(feature_length/2):].mean())).values.reshape(1, feature_length)
    home_game = (team_averages.iloc[:, :int(feature_length/2)].mean().append(team_averages.iloc[i, int(feature_length/2):])).values.reshape(1, feature_length)
    ratings[teams[i]] = model.predict(away_game)-model.predict(home_game)

rankings = pd.DataFrame(index=teams)
rankings['rating'] = ratings
rankings = rankings.sort_values(by='rating', ascending=False)
rankings['rank'] = [i for i in range(1,16)]

print('Predicted ratings and rankings: ')
print(rankings)
print('\n')

rankings.to_csv(path_or_buf='ACCRankings1819.csv', index_label='team')