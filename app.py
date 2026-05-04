from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open('ipl_model.pkl', 'rb') as f:
    model = pickle.load(f)


def predict_score(bat_team, bowl_team, runs, wickets, overs, runs_last_5, wickets_last_5):
    temp_array = []

    teams = [
        'Chennai Super Kings',
        'Delhi Daredevils',
        'Kings XI Punjab',
        'Kolkata Knight Riders',
        'Mumbai Indians',
        'Rajasthan Royals',
        'Royal Challengers Bangalore',
        'Sunrisers Hyderabad'
    ]

    # One-hot encoding for batting team
    for team in teams:
        if bat_team == team:
            temp_array.append(1)
        else:
            temp_array.append(0)

    # One-hot encoding for bowling team
    for team in teams:
        if bowl_team == team:
            temp_array.append(1)
        else:
            temp_array.append(0)

    # Add match stats
    temp_array.extend([runs, wickets, overs, runs_last_5, wickets_last_5])

    temp_array = np.array([temp_array])

    prediction = model.predict(temp_array)[0]

    return int(prediction)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    bat_team = request.form['bat_team']
    bowl_team = request.form['bowl_team']
    runs = int(request.form['runs'])
    wickets = int(request.form['wickets'])
    overs = float(request.form['overs'])
    runs_last_5 = int(request.form['runs_last_5'])
    wickets_last_5 = int(request.form['wickets_last_5'])

    prediction = predict_score(
        bat_team, bowl_team, runs, wickets, overs, runs_last_5, wickets_last_5
    )

    return render_template('index.html', prediction_text=f"Predicted Final Score: {prediction}")


if __name__ == "__main__":
    app.run(debug=True, port=4007,host='0.0.0.0')
