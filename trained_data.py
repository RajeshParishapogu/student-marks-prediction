import pandas as pd
import pickle # model save
from sklearn.linear_model import LinearRegression  #ml model
from sklearn.model_selection import train_test_split  # split the data 
from sklearn.metrics import r2_score   #accuarcy finding

# Load dataset
data = pd.read_csv("student_data.csv")
data.columns = data.columns.str.strip()
X =data[['studenthours']]
y = data['Marks']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict test data
y_pred = model.predict(X_test)

# Calculate accuracy (R2 Score)
accuracy = r2_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save accuracy separately
with open("accuracy.txt", "w") as f:
    f.write(str(round(accuracy, 3)))

print("Model Saved Successfully!")



