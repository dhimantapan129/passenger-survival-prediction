# Passenger Survival Prediction using Decision Tree

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv("titanic.csv")
data = data[["Pclass", "Sex", "Age", "Survived"]]
data["Age"] = data["Age"].fillna(data["Age"].mean())
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

X = data[["Pclass", "Sex", "Age"]]
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Model Accuracy:", round(accuracy * 100, 2), "%")

print("\nEnter Passenger Details")
pclass = int(input("Passenger Class (1, 2, 3): "))
sex = input("Gender (male/female): ").lower()
age = float(input("Age: "))

sex = 0 if sex == "male" else 1

prediction = model.predict([[pclass, sex, age]])

if prediction[0] == 1:
    print("\nPrediction: Passenger Survived")
else:
    print("\nPrediction: Passenger Did Not Survive")
