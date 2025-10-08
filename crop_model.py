# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle

# ---------------------- LOAD DATA ----------------------
df = pd.read_csv("E:\data science\crop recommadation system\Crop_recommendation.csv")

# Features (independent variables)
X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

# Label (dependent variable)
y = df['label']

# ---------------------- ENCODE LABELS ----------------------
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# ---------------------- SCALING ----------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ---------------------- SPLIT DATA ----------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# ---------------------- TRAIN MODEL ----------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ---------------------- SAVE MODEL & SCALER ----------------------
pickle.dump(model, open('crop_model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
pickle.dump(le, open('label_encoder.pkl', 'wb'))

# ---------------------- ACCURACY ----------------------
acc = model.score(X_test, y_test)
print(f"✅ Model trained successfully! Accuracy: {acc:.2f}")
