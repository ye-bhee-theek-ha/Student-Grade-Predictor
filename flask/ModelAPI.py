from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.linear_model import LinearRegression
import joblib

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the pre-trained model
model = joblib.load('./linear_regression_model.pkl')

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.get_json()

    # Extract numerical values from input data
    input_data = [float(data[key]) for key in data]

    # Convert input data to numpy array
    input_data = [input_data]

    # Make prediction using the pre-trained model
    prediction = model.predict(input_data)

    # Return the prediction as JSON response
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
