import pickle
import pandas as pd
from flask import Flask, request, render_template
from sklearn.preprocessing import LabelEncoder

# Create flask app
Booking_status_prediction_app = Flask(__name__)
model = pickle.load(open("Model.pkl", "rb"))


@Booking_status_prediction_app.route("/")
def home():
    return render_template("index.html")


@Booking_status_prediction_app.route("/predict", methods=["POST"])
def predict():
    # Extract form values
    form_values = list(request.form.values())
    col_name = ['number of adults', 'number of children', 'number of weekend nights',
                'number of week nights', 'type of meal', 'car parking space',
                'room type', 'lead time', 'market segment type', 'repeated', 'P-C',
                'P-not-C', 'average price', 'special requests', 'year', 'month', 'day']
    df = pd.DataFrame([form_values], columns=col_name)

    # Label encoding for non-numeric columns
    label_encoder = LabelEncoder()
    df['type of meal'] = label_encoder.fit_transform(df['type of meal'])
    df['room type'] = label_encoder.fit_transform(df['room type'])
    df['market segment type'] = label_encoder.fit_transform(df['market segment type'])

    prediction = model.predict(df)
    return render_template("index.html", prediction_text="Booking status is {}"
                           .format((lambda y: 'Not_Canceled' if prediction == 1 else 'Canceled')(prediction)))


if __name__ == "__main__":
    Booking_status_prediction_app.run(debug=True)
