from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__)

#Load the trained model
try:
    model = pickle.load(open("model.pkl", "rb"))
    print("Model Loaded Successfully")
except FileNotFoundError:
    print("Error: model.pkl not found!")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

#Load car dataset (For UI display)
try:
    cars_data = pd.read_csv('my live cars data.csv')
    print("Car Data Loaded Successfully")
except FileNotFoundError:
    print("Error: my live cars data.csv not found!")
    cars_data = pd.DataFrame()

#Data Preprocessing
if not cars_data.empty:
    cars_data.drop(columns=['torque', 'max_power'], inplace=True, errors='ignore')
    cars_data.dropna(inplace=True)
    cars_data.drop_duplicates(inplace=True)

    #Store brand mappings for display
    cars_data['brand'] = cars_data['name'].apply(lambda x: x.split(' ')[0])  # Extract first word as brand
    brand_mapping = {brand: idx + 1 for idx, brand in enumerate(cars_data['brand'].unique())}
    brand_reverse_mapping = {v: k for k, v in brand_mapping.items()}  # Reverse mapping for display

    #Replace brand names with IDs (for training)
    cars_data['name'] = cars_data['brand'].replace(brand_mapping)
    cars_data.drop(columns=['brand'], inplace=True)

    #Convert categorical values into numbers
    cars_data['transmission'].replace(['Manual', 'Automatic'], [1, 2], inplace=True)
    cars_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3], inplace=True)
    cars_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4], inplace=True)
    cars_data['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], 
                               [1, 2, 3, 4, 5], inplace=True)

#Routes
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

#Corrected /car route to show **actual brand names**
@app.route('/car')
def car():
    if not cars_data.empty:
        display_data = cars_data.head(20).copy()
        display_data['name'] = display_data['name'].map(brand_reverse_mapping)  # Convert IDs back to names
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

#Fixed /predict API
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model is not loaded. Please check model.pkl'})

    try:
        data = request.json
        print("Received Data:", data)  # Debugging

        #Convert JSON to DataFrame
        input_data = pd.DataFrame([data])

        #Ensure required features exist
        required_features = ['name', 'year', 'km_driven', 'fuel', 'seller_type', 'transmission', 'owner']
        for feature in required_features:
            if feature not in input_data:
                input_data[feature] = 0  # Default value for missing features

        #Convert brand name correctly before prediction
        input_data['name'] = input_data['name'].apply(lambda x: brand_mapping.get(x, 0))

        #Make prediction
        predicted_price = model.predict(input_data)[0]
        print("Predicted Price:", predicted_price)

        return jsonify({'predicted_price': round(predicted_price, 2)})

    except Exception as e:
        print("Error in Prediction:", str(e))
        return jsonify({'error': str(e)})

# Run Flask App
if __name__ == '__main__':
    app.run(debug=True)
