import requests

url_log_reg = 'http://localhost:5001/predict'
url_svm = 'http://localhost:5001/predict_svm'
url_tree = 'http://localhost:5001/predict_tree'
url_knn = 'http://localhost:5001/predict_knn'

data = {'features': [140.56, 102.58, 103.01, 136.75, 88.72, 200.12, 150.5, 75.6]}

response_log_reg = requests.post(url_log_reg, json=data)
response_svm = requests.post(url_svm, json=data)
response_tree = requests.post(url_tree, json=data)
response_knn = requests.post(url_knn, json=data)

print("Predicción de la Regresión Logística:", response_log_reg.json())
print("Predicción del SVM:", response_svm.json())
print("Predicción del Árbol de Decisión:", response_tree.json())
print("Predicción del KNN:", response_knn.json())