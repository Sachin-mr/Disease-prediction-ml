from flask import Flask, request, jsonify, render_template
import model as disease_model
import os

app = Flask(__name__)

# Load and train models at startup
disease_model.train_models()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    """API endpoint to get all symptoms"""
    symptoms = disease_model.get_all_symptoms()
    return jsonify(symptoms)

@app.route('/api/diseases', methods=['GET'])
def get_diseases():
    """API endpoint to get all diseases"""
    diseases = disease_model.get_all_diseases()
    return jsonify(diseases)

@app.route('/api/accuracy', methods=['GET'])
def get_accuracy():
    """API endpoint to get model accuracy"""
    accuracy = disease_model.get_model_accuracy()
    return jsonify(accuracy)

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint to predict disease based on symptoms"""
    data = request.get_json()
    
    # Validate input
    if not data or 'symptoms' not in data:
        return jsonify({'error': 'No symptoms provided'}), 400
        
    symptoms = data['symptoms']
    model_name = data.get('model', 'DecisionTree')
    
    try:
        prediction = disease_model.predict_disease(symptoms, model_name)
        return jsonify({
            'prediction': prediction,
            'model': model_name
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Start the Flask app
    app.run(debug=True)

