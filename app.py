from flask import Flask, request, jsonify,render_template
from flask_cors import CORS  
import pickle
import pandas as pd

with open("models/salary_prediction_pipeline.pkl", "rb") as file:
    pipeline = pickle.load(file)

app = Flask(__name__)
CORS(app)  

@app.route('/')
def home():
    return render_template('index.html')
   

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

       
        required_fields = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 
                           'marital-status', 'occupation', 'relationship', 'race', 
                           'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 
                           'native-country']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({"error": f"Missing fields: {missing_fields}"}), 400

       
        input_df = pd.DataFrame([data])

        
        prediction = pipeline.predict(input_df)

        return jsonify({"prediction": prediction[0]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
