from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Correct file path using raw string
url = r"C:\Users\bhimr\OneDrive\Desktop\height-weight.csv"

# Load your dataset
df = pd.read_csv(url)

# Independent and dependent features
X = df[['Weight']]
y = df['Height']

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply Simple Linear Regression
regression = LinearRegression(n_jobs=-1)
regression.fit(X_train_scaled, y_train)

# Inspect the model coefficients
print(f"Model Coefficients: {regression.coef_}")
print(f"Model Intercept: {regression.intercept_}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        weight = float(request.form['weight'])
        
        # Scale the input weight
        weight_scaled = scaler.transform([[weight]])
        
        # Perform prediction
        prediction = regression.predict(weight_scaled)[0]
        
        # Round the prediction to a reasonable number of decimal places
        prediction = round(prediction, 2)
        
        return render_template('result.html', weight=weight, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
