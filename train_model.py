import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('diabetes.csv')

# Map all categorical text to numerical values
mapping = {
    # FamilyHistory & HormonalImbalance
    'Yes': 1, 'No': 0,
    'yes': 1, 'no': 0,
    
    # SleepCycle
    'Good': 1, 'Bad': 0,
    'good': 1, 'bad': 0,
    
    # Lifestyle
    'Active': 2, 'Moderate': 1, 'Lazy': 0,
    'active': 2, 'moderate': 1, 'lazy': 0
}

df = df.replace(mapping)

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

joblib.dump(model, 'diabetes_model.pkl')
print("Model saved successfully as diabetes_model.pkl")