from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

model = joblib.load("./model/loan_default.pkl")
print(model)
print("Model loaded scuccessfully!")

x_test= pd.read_csv('./data/X_test.csv')
y_test= pd.read_csv('./data/Y_test.csv')

print(x_test.head())
print(y_test.head())

y_pred= model.predict(x_test)
print(y_pred)
