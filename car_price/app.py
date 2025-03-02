from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
try:
    model = pickle.load(open("model.pkl", "rb"))
    print("Model Loaded Successfully")
except FileNotFoundError:
    print("Error: model.pkl not found!")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load car dataset (For UI display)
try:
    cars_data = pd.read_csv('my live cars data.csv')
    print("Car Data Loaded Successfully")
except FileNotFoundError:
    print("Error: my live cars data.csv not found!")
    cars_data = pd.DataFrame()

# Data Preprocessing
if not cars_data.empty:
    cars_data.drop(columns=['torque', 'max_power'], inplace=True, errors='ignore')
    cars_data.dropna(inplace=True)
    cars_data.drop_duplicates(inplace=True)

    # Extract car brands
    cars_data['brand'] = cars_data['name'].apply(lambda x: x.split(' ')[0])  
    brand_mapping = {brand: idx + 1 for idx, brand in enumerate(cars_data['brand'].unique())}
    brand_reverse_mapping = {v: k for k, v in brand_mapping.items()}

    # Replace brand names with numeric values
    cars_data['name'] = cars_data['brand'].map(brand_mapping)
    cars_data.drop(columns=['brand'], inplace=True)

    # Convert categorical values into numbers
    mapping_dict = {
        'transmission': {'Manual': 1, 'Automatic': 2},
        'seller_type': {'Individual': 1, 'Dealer': 2, 'Trustmark Dealer': 3},
        'fuel': {'Diesel': 1, 'Petrol': 2, 'LPG': 3, 'CNG': 4},
        'owner': {'First Owner': 1, 'Second Owner': 2, 'Third Owner': 3, 'Fourth & Above Owner': 4, 'Test Drive Car': 5}
    }

    for col, mapping in mapping_dict.items():
        cars_data[col] = cars_data[col].replace(mapping)

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login/<user_type>')
def login(user_type):
    return render_template('login.html', user_type=user_type)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/car')
def car():
    if not cars_data.empty:
        display_data = cars_data.head(20).copy()
        display_data['name'] = display_data['name'].map(brand_reverse_mapping)
    else:
        display_data = pd.DataFrame()
    return render_template('cars.html', cars=display_data.to_dict(orient="records"))

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/add")
def add():
    return render_template("add.html")

@app.route("/form")
def form():
    return render_template("form.html")

# API Endpoint to get brand names for dropdown
@app.route('/get_brands')
def get_brands():
    brands = list(brand_mapping.keys()) if not cars_data.empty else []
    return jsonify({'brands': brands})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Check model.pkl'}), 500

    try:
        data = request.json
        print("Received Data:", data)

        # Validate input data
        required_fields = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'seats']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing required field: {field}'}), 400

        # Convert brand name to ID
        data['name'] = brand_mapping.get(data.get('name', ''), 0)
        if data['name'] == 0:
            return jsonify({'error': 'Invalid brand name'}), 400

        # Create a DataFrame with the correct feature order
        input_data = pd.DataFrame([{
            'name': data['name'],
            'year': data['year'],
            'km_driven': data['km_driven'],
            'fuel': data['fuel'],
            'seller_type': data['seller_type'],
            'transmission': data['transmission'],
            'owner': data['owner'],
            'mileage': data['mileage'],
            'engine': data['engine'],
            'seats': data['seats']
        }])

        # Ensure the feature order matches the training data
        feature_order = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner', 'mileage', 'engine', 'seats']
        input_data = input_data[feature_order]

        # Predict price
        predicted_price = model.predict(input_data)[0]
        print("Predicted Price:", predicted_price)

        return jsonify({'predicted_price': round(predicted_price, 2)})

    except Exception as e:
        print("Error in Prediction:", str(e))
        return jsonify({'error': str(e)}), 500

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
