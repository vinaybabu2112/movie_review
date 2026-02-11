import pandas as pd
from flask import Flask, request, jsonify, render_template
import joblib
from data_processing_and_features import text_data_cleaning, tfidf_features_transform


# Create Flask app
app = Flask(__name__)

# Load the pickle model
model = joblib.load('models/model_classifier.pkl')

# Load the TFIDF vectorizer
tfidf = joblib.load('models/tfidf.pkl')

@app.route("/")
def home():
    return render_template("index.html")

# Define a route for the prediction endpoint
@app.route('/predict', methods=["POST"])

def predict():
    review = request.form['Review']  # Get the review from the form
    data_pred = pd.DataFrame([review], columns=['review'])
    data_pred = text_data_cleaning(data_pred)
    data_pred_matrix = tfidf_features_transform(tfidf, data_pred)

    prediction = model.predict(data_pred_matrix)
    
    # Convert the prediction to sentiment labels
    sentiment = "positive" if prediction[0] == 1 else "negative"

    return render_template("index.html", prediction_text="The sentiment is {}".format(sentiment))

# Define a route for the API based prediction endpoint
@app.route('/sentiment', methods=["GET"])
#http://127.0.0.1:5000/sentiment?Review="nice movie"

def sentiment():
    review = request.args.get("Review")  # Get the review from the form
    data_pred = pd.DataFrame([review], columns=['review'])
    data_pred = text_data_cleaning(data_pred)
    data_pred_matrix = tfidf_features_transform(tfidf, data_pred)

    prediction = model.predict(data_pred_matrix)
    
    # Convert the prediction to sentiment labels
    sentiment = "positive" if prediction[0] == 1 else "negative"

    prediction_text = "The sentiment is {}".format(sentiment)
    return jsonify(prediction_text)


# Run the app if executed directly
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)
