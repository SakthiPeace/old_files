from flask import Flask, json, request
import numpy as np
from joblib import load

app = Flask(__name__)

filename = './diabetes.joblib'
classifier = load(filename)

def validate_input(data):
    """
    Validates the input data.

    Args:
        data (dict): Input data dictionary.

    Returns:
        bool: True if input data is valid, False otherwise.
    """
    try:
        required_fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigree', 'Age']
        for field in required_fields:
            if field not in data:
                return False
            
            # Check if field value is numeric
            value = float(data[field])
            
            # Check if field value is within a reasonable range
            if field == 'Pregnancies' and not 0 <= value <= 20:
                return False
            elif field == 'Glucose' and not 50 <= value <= 200:
                return False
            elif field == 'BloodPressure' and not 40 <= value <= 150:
                return False
            elif field == 'SkinThickness' and not 10 <= value <= 100:
                return False
            elif field == 'Insulin' and not 0 <= value <= 1000:
                return False
            elif field == 'BMI' and not 15 <= value <= 50:
                return False
            elif field == 'DiabetesPedigree' and not 0 <= value <= 2:
                return False
            elif field == 'Age' and not 10 <= value <= 90:
                return False
            
    except (ValueError, TypeError):
        return False

    return True

@app.route('/diabetes', methods=['POST'])
def predict_diabetes():
    if request.method == 'POST':
        try:
            data = request.get_json()  # Get JSON data from request body
            
            if not validate_input(data):
                return json.dumps({"error": "Invalid input data. Please provide all required fields with valid numeric values within a reasonable range."}), 400
            
            values = [float(value) for value in data.values()]
            
            x = np.array([values])
            prediction = classifier.predict(x)
            
            if prediction == 1:
                result = 'You have diabetes problem.'
            else:
                result = 'You don\'t have diabetes issue.'
            
            return json.dumps({"message": result}), 200
        except Exception as e:
            return json.dumps({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
