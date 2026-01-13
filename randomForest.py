import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

final_df = pd.read_csv("training_matrix_sully.csv")

X = final_df.iloc[:, :8]  
y = final_df.iloc[:, 8:] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

score = model.score(X_test, y_test)
print(f"Model Accuracy (R²): {score:.4f}")

sample_input = X_test.iloc[0:1]
prediction = model.predict(sample_input)

print("\nReal Values:\n", y_test.iloc[0].values)
print("Predicted:\n", prediction[0])