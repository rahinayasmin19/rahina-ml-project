import pandas as pd
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load your dataset
df = pd.read_csv("C:\\Users\\rksri\\OneDrive\\Desktop\\house_data.csv")  # Update filename if different

# Encode categorical values
le_mainroad = LabelEncoder()
df['Mainroad'] = le_mainroad.fit_transform(df['Mainroad'])

le_location = LabelEncoder()
df['Location'] = le_location.fit_transform(df['Location'])

# Define features and target
X = df[['Area', 'Bedrooms', 'Bathrooms', 'Stories', 'Mainroad', 'Parking', 'Location']]
y = df['Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders to a .pkl file
with open("predict_model.pkl", "wb") as f:
    pickle.dump({
        'model': model,
        'le_mainroad': le_mainroad,
        'le_location': le_location
    }, f)

print("✅ Model saved at:", os.path.abspath("predict_model.pkl"))
