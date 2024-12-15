from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Load the trained model
model_path = 'model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract features from the form
        features = [
            float(request.form.get('Year')),
            float(request.form.get('Present_Price')),
            float(request.form.get('Kms_Driven')),
            float(request.form.get('Fuel_Type')),
            float(request.form.get('Seller_Type')),
            float(request.form.get('Transmission')),
            float(request.form.get('Owner'))
        ]
        
        # Convert features into numpy array for prediction
        final_features = np.array([features])
        
        # Make the prediction
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        
        return render_template('index.html', prediction_text=f'Predicted Price: ₹{output}')
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
