from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pickle
import pandas as pd

# Load the pipeline (preprocessing + model)
with open("salary_prediction_pipeline.pkl", "rb") as file:
    pipeline = pickle.load(file)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return "API is working!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Validate input fields
        required_fields = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                           'marital-status', 'occupation', 'relationship', 'race', 
                           'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 
                           'native-country']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {missing_fields}"}), 400

        # Convert input data to DataFrame
        input_df = pd.DataFrame([data])

        # Use the pipeline for preprocessing + prediction
        prediction = pipeline.predict(input_df)

        return jsonify({"prediction": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
