import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the pre-trained model
model = pickle.load(open('pipe.pkl', 'rb'))

# Define the teams and cities from the dataset
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals', 'Gujarat Lions', 'Rising Pune Supergiant']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru', 'Rajkot']

st.title('IPL Win Predictor')

# Add a header image
ipl_image = 'image.jpg'  # Ensure this image is in the same directory as your script
if os.path.exists(ipl_image):
    st.image(ipl_image, use_column_width=True)
else:
    st.write("Please add an image named 'ipl_image.jpg' to the directory.")

col1, col2 = st.columns(2)

with col1:
    batting_team = st.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.selectbox('Select the bowling team', sorted(teams))

selected_city = st.selectbox('Select host city', sorted(cities))

target = st.number_input('Target')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.number_input('Score')
with col4:
    overs = st.number_input('Overs completed', min_value=0.0, max_value=20.0, step=0.1)
with col5:
    wickets = st.number_input('Wickets out', min_value=0, max_value=10, step=1)

if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets_left = 10 - wickets
    
    # Check if overs or balls_left are zero or invalid to avoid division by zero
    if overs == 0:
        st.error("Overs completed cannot be zero.")
    elif balls_left <= 0:
        st.error("Balls left must be positive.")
    else:
        crr = score / overs
        rrr = (runs_left * 6) / balls_left if balls_left > 0 else float('inf')

        input_df = pd.DataFrame({
            'batting_team': [batting_team],
            'bowling_team': [bowling_team],
            'city': [selected_city],
            'runs_left': [runs_left],
            'balls_left': [balls_left],
            'wickets': [wickets_left],
            'target': [target],
            'crr': [crr],
            'rrr': [rrr]
        })

        # One-hot encoding for categorical variables
        input_df = pd.get_dummies(input_df)

        # Align the input_df with the training dataframe
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

        result = model.predict_proba(input_df)
        loss = result[0][0]
        win = result[0][1]
        st.header(f"{batting_team} - {win*100:.2f}%")
        st.header(f"{bowling_team} - {loss*100:.2f}%")

        # Display bar graphs to visualize data
        fig, ax = plt.subplots(2, 2, figsize=(15, 10))

        sns.barplot(x=['Runs Left', 'Balls Left', 'Wickets Left'], y=[runs_left, balls_left, wickets_left], ax=ax[0, 0])
        ax[0, 0].set_title('Match Situation')

        sns.barplot(x=['CRR', 'RRR'], y=[crr, rrr], ax=ax[0, 1])
        ax[0, 1].set_title('Run Rates')

        # Dummy data for run rates over time
        overs_dummy = list(range(1, int(overs)+1))
        crr_dummy = [score/o for o in overs_dummy]
        rrr_dummy = [(runs_left * 6)/(120 - o*6) for o in overs_dummy]

        sns.lineplot(x=overs_dummy, y=crr_dummy, ax=ax[1, 0], label='CRR')
        sns.lineplot(x=overs_dummy, y=rrr_dummy, ax=ax[1, 0], label='RRR')
        ax[1, 0].set_title('Run Rates Over Time')
        ax[1, 0].set_xlabel('Overs')
        ax[1, 0].set_ylabel('Run Rate')

        sns.scatterplot(x=['Runs Left', 'Balls Left', 'Wickets Left'], y=[runs_left, balls_left, wickets_left], ax=ax[1, 1])
        ax[1, 1].set_title('Match Situation Scatter Plot')

        st.pyplot(fig)

        # Display historical performance
        historical_data = {
            'Sunrisers Hyderabad': {'Wins': 5, 'Losses': 3},
            'Mumbai Indians': {'Wins': 4, 'Losses': 4},
            # Add more teams as needed
        }

        if batting_team in historical_data:
            st.subheader(f"{batting_team} Historical Performance")
            st.write(f"Wins: {historical_data[batting_team]['Wins']}")
            st.write(f"Losses: {historical_data[batting_team]['Losses']}")

        if bowling_team in historical_data:
            st.subheader(f"{bowling_team} Historical Performance")
            st.write(f"Wins: {historical_data[bowling_team]['Wins']}")
            st.write(f"Losses: {historical_data[bowling_team]['Losses']}")
