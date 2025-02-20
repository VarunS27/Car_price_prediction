import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# ✅ Load the model safely
try:
    model = pickle.load(open("model.pkl", "rb"))
    print("Model Loaded Successfully")  
except FileNotFoundError:
    print("❌ Error: model.pkl not found!")
    model = None
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# ✅ Load car data with error handling
try:
    cars_data = pd.read_csv('my live cars data.csv')
    print("✅ Car Data Loaded Successfully")
except FileNotFoundError:
    print("❌ Error: my live cars data.csv not found!")
    cars_data = pd.DataFrame()  # Empty DataFrame as fallback

# ✅ Preprocessing (Safe)
if not cars_data.empty:
    cars_data.drop(columns=['torque', 'max_power'], inplace=True, errors='ignore')
    cars_data.dropna(inplace=True)
    cars_data.drop_duplicates(inplace=True)

    # Convert categorical values into numbers
    cars_data['name'] = cars_data['name'].apply(lambda x: x.split(' ')[0])
    brand_mapping = {brand: idx + 1 for idx, brand in enumerate(cars_data['name'].unique())}
    cars_data['name'].replace(brand_mapping, inplace=True)

    cars_data['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    cars_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    cars_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    cars_data['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], 
                               [1, 2, 3, 4, 5], inplace=True)

# ✅ Route to render cars page
@app.route('/cars')
def car():
    return render_template('cars.html', cars=cars_data.head(20).to_dict(orient="records"))

# ✅ API Route for price prediction
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded'})

    try:
        data = request.json

        # Convert input data to DataFrame
        input_data = pd.DataFrame([data])

        # Ensure all features exist in input_data (Handle missing columns)
        required_features = ['name', 'year', 'selling_price', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
        for feature in required_features:
            if feature not in input_data:
                input_data[feature] = 0  # Default value for missing features

        # Convert brand name using the same encoding as training
        input_data['name'] = input_data['name'].apply(lambda x: brand_mapping.get(x, 0))

        # Make prediction
        predicted_price = model.predict(input_data)[0]

        return jsonify({'predicted_price': round(predicted_price, 2)})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
