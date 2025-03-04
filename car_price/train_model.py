import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load Data
try:
    cars_data = pd.read_csv('cars.csv')
    print("Car Data Loaded Successfully")
except FileNotFoundError:
    print("Error: cars.csv not found!")
    exit()

# Data Preprocessing
cars_data.drop(columns=['torque', 'max_power'], inplace=True, errors='ignore')
cars_data.dropna(inplace=True)
cars_data.drop_duplicates(inplace=True)

# Convert text-based numeric values to numbers
cars_data['mileage'] = cars_data['mileage'].str.extract(r'(\d+\.\d+|\d+)').astype(float)  # Extract numeric part
cars_data['engine'] = cars_data['engine'].str.extract(r'(\d+)').astype(float)  # Extract numeric part

# Convert categorical values into numbers
cars_data['name'] = cars_data['name'].apply(lambda x: x.split(' ')[0])
brand_mapping = {brand: idx + 1 for idx, brand in enumerate(cars_data['name'].unique())}
cars_data.loc[:, 'name'] = cars_data['name'].replace(brand_mapping).astype(int)

cars_data.loc[:, 'transmission'] = cars_data['transmission'].replace(['Manual', 'Automatic'], [1, 2]).astype(int)
cars_data.loc[:, 'seller_type'] = cars_data['seller_type'].replace(['Individual', 'Dealer', 'Trustmark Dealer'], [1, 2, 3]).astype(int)
cars_data.loc[:, 'fuel'] = cars_data['fuel'].replace(['Diesel', 'Petrol', 'LPG', 'CNG'], [1, 2, 3, 4]).astype(int)
cars_data.loc[:, 'owner'] = cars_data['owner'].replace(['First Owner', 'Second Owner', 'Third Owner', 'Fourth & Above Owner', 'Test Drive Car'], [1, 2, 3, 4, 5]).astype(int)

# Prepare Input & Output
X = cars_data.drop(columns=['selling_price'])  # Features
y = cars_data['selling_price']  # Target variable

# Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Save Model
pickle.dump(model, open("model.pkl", "wb"))
print("Model Saved Successfully as model.pkl")
