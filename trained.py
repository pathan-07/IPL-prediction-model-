import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load your dataset
matches_df = pd.read_csv('matches.csv')

# Preprocess the dataset to fit the model
# Assuming you have the following features: 'batting_team', 'bowling_team', 'city', 'runs_left', 'balls_left', 'wickets', 'target', 'crr', 'rrr'
# And target variable 'winning_team' (1 for batting team win, 0 for bowling team win)

# Here we simulate the process with dummy data (replace it with your actual preprocessing)
data = {
    'batting_team': ['Mumbai Indians', 'Chennai Super Kings', 'Kolkata Knight Riders', 'Sunrisers Hyderabad'] * 50,
    'bowling_team': ['Chennai Super Kings', 'Mumbai Indians', 'Sunrisers Hyderabad', 'Kolkata Knight Riders'] * 50,
    'city': ['Mumbai', 'Chennai', 'Kolkata', 'Hyderabad'] * 50,
    'runs_left': [50, 60, 70, 80] * 50,
    'balls_left': [30, 24, 18, 12] * 50,
    'wickets': [3, 4, 5, 6] * 50,
    'target': [180, 190, 200, 210] * 50,
    'crr': [7.5, 7.9, 8.3, 8.6] * 50,
    'rrr': [10, 12, 14, 16] * 50,
    'winning_team': [1, 0, 1, 0] * 50  # 1 for batting team win, 0 for bowling team win
}

df = pd.DataFrame(data)

# One-hot encode categorical variables
df = pd.get_dummies(df, columns=['batting_team', 'bowling_team', 'city'])

# Features and target
X = df.drop('winning_team', axis=1)
y = df['winning_team']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model to a .pkl file
with open('pipe.pkl', 'wb') as file:
    pickle.dump(model, file)
