import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
from lime.lime_tabular import LimeTabularExplainer
import io
import os
from sklearn.preprocessing import LabelEncoder

# Set Matplotlib backend to 'Agg' to prevent GUI-related errors
plt.switch_backend('Agg')

app = Flask(__name__)
CORS(app)

# Load the model
try:
    with open('/Users/ayyalashriyatha/Desktop/model.pkl', 'rb') as file:
        loaded_model = pickle.load(file)
except FileNotFoundError:
    print("Model file not found!")
    exit(1)

# Load training data for LIME explainer
with open('/Users/ayyalashriyatha/Desktop/X_train_1.pkl', 'rb') as file:
    X_train = pickle.load(file)
with open('/Users/ayyalashriyatha/Desktop/y_train_1.pkl', 'rb') as file:
    y_train = pickle.load(file)


# Define label encoders
revenue_encoder = LabelEncoder()
votes_encoder = LabelEncoder()

# Initialize LIME explainer
explainer = LimeTabularExplainer(
    X_train.values, 
    training_labels=np.array(y_train), 
    mode="classification", 
    feature_names=X_train.columns,
    class_names=["Not Success", "Success"]
)

# Define bin edges and labels for Revenue and Votes
# Example of binning revenue
revenue_bins = [0, 100, 500, 1000, np.inf]
revenue_labels = ['Low', 'Medium', 'High', 'Very High']
votes_bins = [0, 20000, 100000, 300000, 600000, np.inf]
votes_labels = ['Low', 'Medium', 'High', 'Very High', 'Extremely High']



# Fit label encoders with bin labels
revenue_encoder.fit(revenue_labels)
votes_encoder.fit(votes_labels)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)  # Receive JSON
    print("Received data:", data)  # Debug: Log the incoming data

    try:
        # Generate features from the data (similar to what you did in the frontend)
        features = {
            'Runtime (Minutes)': float(data.get('Runtime (Minutes)', 0)),
            'Rating': float(data.get('Rating', 0)),
            'Votes': int(data.get('Votes', 0)),
            'Revenue (Millions)': float(data.get('Revenue (Millions)', 0)),
            'Metascore': float(data.get('Metascore', 0))
        }

        # Bin the 'Revenue (Millions)' and 'Votes' features into 'revenue_binned' and 'votes_binned'
        features['Revenue_Binned'] = pd.cut(
            [features['Revenue (Millions)']], bins=revenue_bins, labels=revenue_labels
        )[0]
        features['Votes_Binned'] = pd.cut(
            [features['Votes']], bins=votes_bins, labels=votes_labels
        )[0]

        # Encode the binned 'revenue_binned' and 'votes_binned' columns
        features['Revenue_Binned'] = revenue_encoder.transform([features['Revenue_Binned']])[0]
        features['Votes_Binned'] = votes_encoder.transform([features['Votes_Binned']])[0]

        # Remove original 'Revenue (Millions)' and 'Votes' columns as we now use binned versions
        del features['Revenue (Millions)']
        del features['Votes']

        # Create a DataFrame for prediction
        features_df = pd.DataFrame([features])

        # Ensure the column order matches the model's expected input
        expected_columns = X_train.columns.tolist()
        features_df = features_df[expected_columns]  # Ensure features match training data columns

        # Handle missing values by filling them with 0 (if applicable)
        features_df = features_df.fillna(0)

        # Debug: Check the features before prediction
        print("Features before prediction:", features_df)

        # Make the prediction
        prediction = loaded_model.predict(features_df)
        prediction_proba = loaded_model.predict_proba(features_df)

        # Debug: Check the prediction result
        print("Prediction:", prediction)
        print("Prediction probabilities:", prediction_proba)

        # Determine success based on the prediction probabilities
        is_successful = prediction[0] == 1  # Assuming '1' is the 'Success' class
        response = {
            "success_status": "Successful" if is_successful else "Not Successful",
            "prediction_probability": prediction_proba.tolist()
        }
        return jsonify(response)

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500







@app.route('/lime_explanation', methods=['POST'])
def lime_explanation():
    data = request.get_json(force=True)

    try:
        # Ensure the data fields are numeric (or set to 0 if missing or invalid)
        features = {
            'Runtime (Minutes)': float(data.get('Runtime (Minutes)', 0)),  # Ensure it's float
            'Rating': float(data.get('Rating', 0)),  # Ensure it's float
            'Votes': int(data.get('Votes', 0)),  # Ensure it's an integer
            'Revenue (Millions)': float(data.get('Revenue (Millions)', 0)),  # Ensure it's float
            'Metascore': float(data.get('Metascore', 0)),  # Ensure it's float
        }

        # Now perform binning for Revenue and Votes
        features['Revenue_Binned'] = pd.cut(
            [features['Revenue (Millions)']], bins=revenue_bins, labels=revenue_labels
        )[0]
        features['Votes_Binned'] = pd.cut(
            [features['Votes']], bins=votes_bins, labels=votes_labels
        )[0]

        # Encode the binned 'revenue_binned' and 'votes_binned' columns
        features['Revenue_Binned'] = revenue_encoder.transform([features['Revenue_Binned']])[0]
        features['Votes_Binned'] = votes_encoder.transform([features['Votes_Binned']])[0]

        # Remove original 'Revenue (Millions)' and 'Votes' columns as we now use binned versions
        del features['Revenue (Millions)']
        del features['Votes']

        # Create DataFrame for prediction with correct columns
        features_df = pd.DataFrame([features])

        # Ensure the order of the columns in features_df matches X_train
        features_df = features_df[X_train.columns]

        # Handle missing values by filling with 0 or other appropriate values
        features_df = features_df.fillna(0)

    except KeyError as e:
        return jsonify({"error": f"Missing feature: {e.args[0]}"}), 400

    # Generate LIME explanation
    explanation = explainer.explain_instance(
        features_df.iloc[0].values, 
        loaded_model.predict_proba, 
        num_features=5
    )

    fig = explanation.as_pyplot_figure()
    fig.subplots_adjust(left=0.4, right=0.9, top=0.9, bottom=0.1)

    img_bytes = io.BytesIO()
    fig.savefig(img_bytes, format='png')
    img_bytes.seek(0)
    plt.close(fig)

    return send_file(img_bytes, mimetype='image/png')


if __name__ == '__main__':
    app.run(debug=True)
