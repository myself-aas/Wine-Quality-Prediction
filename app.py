import joblib
import numpy as np
from flask import Flask, render_template, request
from sklearn.preprocessing import MinMaxScaler

# Initialize Flask app
app = Flask(__name__)

# Load the model, scaler, and input columns
wine_params = joblib.load('wine.joblib')
model = wine_params['model']
scaler = wine_params['scaler']
input_cols = wine_params['input_cols']

# Route to home page
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle the prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user inputs
        input_values = [float(request.form[col]) for col in input_cols]
        
        # Scale the input features
        input_scaled = scaler.transform([input_values])
        
        # Make prediction
        prediction = model.predict(input_scaled)
        
        return render_template('index.html', prediction_text=f'Predicted Wine Quality: {prediction[0]}')
    
    except Exception as e:
        return render_template('index.html', prediction_text="Error in prediction: " + str(e))

if __name__ == "__main__":
    app.run(debug=True)
