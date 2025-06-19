from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

log_reg_model = joblib.load('log_reg_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = [float(x) for x in data['features']]
    prediction = log_reg_model.predict(np.array([features]))
    return jsonify({'prediction': int(prediction[0])})

svm_model = joblib.load('svm_model.pkl')

@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    data = request.get_json(force=True)
    features = [float(x) for x in data['features']]
    prediction = svm_model.predict(np.array([features]))
    return jsonify({'prediction': int(prediction[0])})

decision_tree_model = joblib.load('decision_tree_model.pkl')

@app.route('/predict_tree', methods=['POST'])
def predict_tree():
    data = request.get_json(force=True)
    features = [float(x) for x in data['features']]
    prediction = decision_tree_model.predict(np.array([features]))
    return jsonify({'prediction': int(prediction[0])})

knn_model = joblib.load('knn_model.pkl')

@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    data = request.get_json(force=True)
    features = [float(x) for x in data['features']]
    prediction = knn_model.predict(np.array([features]))
    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True, port=5001)