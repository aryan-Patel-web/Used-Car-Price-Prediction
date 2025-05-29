
# 🚗 Used Car Price Prediction

This project predicts the price of used cars using machine learning regression models. The workflow includes data preprocessing, feature engineering, model training, hyperparameter tuning, and deployment readiness for Flask or Streamlit web apps.

---

## Features

- Data cleaning and preprocessing
- Feature engineering and encoding
- Multiple regression models (Random Forest, K-Nearest Neighbors, etc.)
- Model evaluation (MAE, RMSE, R²)
- Hyperparameter tuning with RandomizedSearchCV
- Model and preprocessor saving for deployment
- Ready for deployment with Flask or Streamlit

---

## File Structure

```
used-car-price-prediction/
├── CAR_Price_Predict.ipynb
├── app.py
├── preprocessor.pkl
├── car_price_model.pkl
├── requirements.txt
├── README.md
└── templates/
    └── index.html
```

---

## Best Model

After evaluating several models, **Random Forest Regressor** was found to be the most suitable due to its high accuracy and robustness. This model is used in the deployment phase.

---

## How to Use

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/used-car-price-prediction.git
cd used-car-price-prediction
```

### 2. Install Requirements

pip install -r requirements.txt
```

### 3. Train and Save the Model

Run the Jupyter notebook `CAR_Price_Predict.ipynb` to preprocess data, train models, and save the preprocessor and best model:

```python
import joblib
joblib.dump(preprocessor, "preprocessor.pkl")
joblib.dump(model, "car_price_model.pkl")
```

### 4. Deploy with Flask

Create an `app.py` file like this:

```python
from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
preprocessor = joblib.load("preprocessor.pkl")
model = joblib.load("car_price_model.pkl")

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        year = int(request.form['year'])
        km_driven = int(request.form['km_driven'])
        fuel_type = request.form['fuel_type']
        seller_type = request.form['seller_type']
        transmission_type = request.form['transmission_type']
        model_input = int(request.form['model'])

        input_df = pd.DataFrame([{
            'year': year,
            'km_driven': km_driven,
            'fuel_type': fuel_type,
            'seller_type': seller_type,
            'transmission_type': transmission_type,
            'model': model_input
        }])

        X_processed = preprocessor.transform(input_df)
        prediction = model.predict(X_processed)[0]
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
```

Create a `templates/index.html` file for the web form.

### 5. Deploy to Cloud

- Push your code to GitHub.
- Deploy on AWS, Heroku, or Streamlit Cloud as per your preference.

---

## Sample Input & Output

**Input:**
- Year: 2015
- KM Driven: 50000
- Fuel Type: Petrol
- Seller Type: Individual
- Transmission: Manual
- Model: 120

**Predicted Output:** ₹ 4,50,000

---

## Requirements

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- flask
- joblib

---

## Author

**Aryan Patel**  
📧 aryanpatell77462@gmail.com

---

## License

This project is for educational purposes.
