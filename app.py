import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load data and model
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open('RidgeModel.pkl', 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        location = request.form.get("location")
        bhk = float(request.form.get('bhk'))
        bath = float(request.form.get('bath'))
        sqft = float(request.form.get('total_sqrt'))  # keep form name same

        # Create dataframe with correct column names for model
        input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                                columns=['location', 'total_sqft', 'bath', 'bhk'])  # ✅ fixed here

        prediction = pipe.predict(input_df)[0] * 1e5  # assuming model outputs in lakhs
        return str(np.round(prediction, 2))

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    app.run(debug=True, port=5001)
