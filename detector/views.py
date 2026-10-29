from django.shortcuts import render
from django.conf import settings
import joblib
import numpy as np
import os

MODEL_PATH = os.path.join(settings.BASE_DIR, 'diabetes_model.pkl')

def get_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        return None

def home(request):
    return render(request, 'home.html')

def login_view(request):
    return render(request, 'login.html')

def diagnosis(request):
    result = None
    if request.method == 'POST':
        model = get_model()
        if model is None:
            result = "Error: Model file 'diabetes_model.pkl' not found. Run train_model.py first."
        else:
            try:
                features = [
                    float(request.POST.get('pregnancies', 0)),
                    float(request.POST.get('glucose', 0)),
                    float(request.POST.get('blood_pressure', 0)),
                    float(request.POST.get('skin_thickness', 0)),
                    float(request.POST.get('insulin', 0)),
                    float(request.POST.get('family_history', 0)),
                    float(request.POST.get('bmi', 0)),
                    float(request.POST.get('cortisol_level', 0)),
                    float(request.POST.get('hormonal_imbalance', 0)),
                    float(request.POST.get('sleep_cycle', 0)),
                    float(request.POST.get('lifestyle', 0)),
                    float(request.POST.get('dpf', 0)),
                    float(request.POST.get('age', 0))
                ]
                
                prediction = model.predict([np.array(features)])
                result = "Positive (Diabetes Detected)" if prediction[0] == 1 else "Negative (No Diabetes)"
            except Exception as e:
                result = f"Error processing data: {str(e)}"
                
    return render(request, 'diagnosis.html', {'result': result})

def reversal(request):
    return render(request, 'reversal.html')