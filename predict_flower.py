import joblib

# Load the saved model
model = joblib.load('iris_model.pkl')

# Define the target names manually (since we don't have the iris object here)
target_names = ['setosa', 'versicolor', 'virginica']

# Take user input
sepal_length = float(input("Enter sepal length (cm): "))
sepal_width = float(input("Enter sepal width (cm): "))
petal_length = float(input("Enter petal length (cm): "))
petal_width = float(input("Enter petal width (cm): "))

# Create input array
input_data = [[sepal_length, sepal_width, petal_length, petal_width]]

# Predict
prediction = model.predict(input_data)
predicted_species = target_names[prediction[0]]

print(f"\nPredicted species: {predicted_species}")
