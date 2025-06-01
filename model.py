import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# List of symptoms
symptoms = ['back_pain', 'constipation', 'abdominal_pain', 'diarrhoea', 'mild_fever', 'yellow_urine',
'yellowing_of_eyes', 'acute_liver_failure', 'fluid_overload', 'swelling_of_stomach',
'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision', 'phlegm', 'throat_irritation',
'redness_of_eyes', 'sinus_pressure', 'runny_nose', 'congestion', 'chest_pain', 'weakness_in_limbs',
'fast_heart_rate', 'pain_during_bowel_movements', 'pain_in_anal_region', 'bloody_stool',
'irritation_in_anus', 'neck_pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen_legs',
'swollen_blood_vessels', 'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails',
'swollen_extremeties', 'excessive_hunger', 'extra_marital_contacts', 'drying_and_tingling_lips',
'slurred_speech', 'knee_pain', 'hip_joint_pain', 'muscle_weakness', 'stiff_neck', 'swelling_joints',
'movement_stiffness', 'spinning_movements', 'loss_of_balance', 'unsteadiness',
'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort', 'foul_smell_of urine',
'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body', 'belly_pain',
'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite', 'polyuria', 'family_history', 'mucoid_sputum',
'rusty_sputum', 'lack_of_concentration', 'visual_disturbances', 'receiving_blood_transfusion',
'receiving_unsterile_injections', 'coma', 'stomach_bleeding', 'distention_of_abdomen',
'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling',
'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister', 'red_sore_around_nose',
'yellow_crust_ooze']

# List of diseases
diseases = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
'Migraine', 'Cervical spondylosis',
'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis',
'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
'Impetigo']

# Disease to index mapping and vice versa
disease_to_index = {disease: index for index, disease in enumerate(diseases)}
index_to_disease = {index: disease for index, disease in enumerate(diseases)}

# Models dictionary to store trained models
models = {}

def train_models():
    """Train all machine learning models and store them in the models dictionary"""
    
    # Load the training data
    try:
        # Try to load with relative path
        df = pd.read_csv("D:\coding\Disease-prediction-using-Machine-Learning-master\Disease-prediction-using-Machine-Learning-master\Training.csv")
        tr = pd.read_csv("D:\coding\Disease-prediction-using-Machine-Learning-master\Disease-prediction-using-Machine-Learning-master\Testing.csv")
    except:
        # Fallback to original path (should be changed to your actual path)
        df = pd.read_csv(r"D:\coding\Disease-prediction-using-Machine-Learning-master\Disease-prediction-using-Machine-Learning-master\Training.csv")
        tr = pd.read_csv(r"D:\coding\Disease-prediction-using-Machine-Learning-master\Disease-prediction-using-Machine-Learning-master\Testing.csv")
    
    # Replace disease names with disease indices
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    df['prognosis'] = le.fit_transform(df['prognosis'])
    tr['prognosis'] = le.transform(tr['prognosis'])
    
    
    # Prepare training data
    X = df[symptoms]
    y = df[["prognosis"]]
    y = np.ravel(y)
    
    # Prepare testing data
    X_test = tr[symptoms]
    y_test = tr[["prognosis"]]
    y_test = np.ravel(y_test)
    
    # Decision Tree
    dt_model = tree.DecisionTreeClassifier()
    dt_model.fit(X, y)
    dt_preds = dt_model.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_preds)
    models['DecisionTree'] = {
        'model': dt_model,
        'accuracy': round(dt_accuracy * 100, 2)
    }
    
    # Random Forest
    rf_model = RandomForestClassifier()
    rf_model.fit(X, y)
    rf_preds = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_preds)
    models['RandomForest'] = {
        'model': rf_model,
        'accuracy': round(rf_accuracy * 100, 2)
    }
    
    # Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X, y)
    nb_preds = nb_model.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_preds)
    models['NaiveBayes'] = {
        'model': nb_model,
        'accuracy': round(nb_accuracy * 100, 2)
    }
    
    return models

def predict_disease(selected_symptoms, model_name='DecisionTree'):
    """
    Predict disease based on symptoms
    
    Args:
        selected_symptoms (list): List of symptoms
        model_name (str): Name of model to use for prediction
    
    Returns:
        str: Predicted disease
    """
    # Check if models are trained
    if not models:
        train_models()
    
    # Check if model exists
    if model_name not in models:
        return "Invalid model name"
    
    # Convert symptoms to feature vector
    input_vector = [0] * len(symptoms)
    for symptom in selected_symptoms:
        if symptom in symptoms:
            index = symptoms.index(symptom)
            input_vector[index] = 1
    
    # Make prediction
    model = models[model_name]['model']
    prediction = model.predict([input_vector])
    predicted_disease = diseases[prediction[0]]
    
    return predicted_disease

def get_all_symptoms():
    """Return all symptoms"""
    return sorted(symptoms)

def get_all_diseases():
    """Return all diseases"""
    return diseases

def get_model_accuracy():
    """Return accuracy of each model"""
    if not models:
        train_models()
    
    return {name: info['accuracy'] for name, info in models.items()}

# Train models when the module is imported
if __name__ == "__main__":
    train_models()