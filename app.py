from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load preprocessor, model, and label encoder
preprocessor = joblib.load("preprocessor11.pkl")
model = joblib.load("car_price_model_Prediction11.pkl")
model_le = joblib.load("model_label_encoder11.pkl")  # Make sure you saved this during training

# List of available car model names used in dropdown
model_names = list(model_le.classes_)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        # Get input values from form
        vehicle_age = int(request.form['vehicle_age'])
        km_driven = int(request.form['km_driven'])
        fuel_type = request.form['fuel_type']
        seller_type = request.form['seller_type']
        transmission_type = request.form['transmission_type']
        model_name = request.form['model']
        mileage = float(request.form['mileage'])
        engine = int(request.form['engine'])
        max_power = float(request.form['max_power'])
        seats = int(request.form['seats'])

        # Encode model name to integer
        model_input = model_le.transform([model_name])[0]

        # Create DataFrame for prediction
        input_df = pd.DataFrame([{
            'model': model_input,
            'vehicle_age': vehicle_age,
            'km_driven': km_driven,
            'seller_type': seller_type,
            'fuel_type': fuel_type,
            'transmission_type': transmission_type,
            'mileage': mileage,
            'engine': engine,
            'max_power': max_power,
            'seats': seats
        }])

        # Transform input and predict
        X_processed = preprocessor.transform(input_df)
        prediction = model.predict(X_processed)[0]

    return render_template('index.html', prediction=prediction, model_names=model_names)

if __name__ == '__main__':
    app.run(debug=True)