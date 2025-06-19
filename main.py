import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('pulsar_stars.csv')

X = data.iloc[:, :-1]
y = data['target_class']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)

from sklearn.svm import SVC
svm = SVC()
svm.fit(X_train, y_train)

from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

print("Exactitud de la Regresión Logística: ", log_reg.score(X_test, y_test))
print("Exactitud del SVM: ", svm.score(X_test, y_test))
print("Exactitud del Árbol de Decisión: ", decision_tree.score(X_test, y_test))
print("Exactitud del KNN: ", knn.score(X_test, y_test))

import joblib

joblib.dump(log_reg, 'log_reg_model.pkl')
joblib.dump(svm, 'svm_model.pkl')
joblib.dump(decision_tree, 'decision_tree_model.pkl')
joblib.dump(knn, 'knn_model.pkl')

print("Modelos entrenados y guardados exitosamente.")