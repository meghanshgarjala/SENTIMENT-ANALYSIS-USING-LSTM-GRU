from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = load_model('sentiment_model.h5')

# Define maximum length for padding
max_len = 300

# Route for homepage
@app.route('/')
def home():
    return render_template('index.html')

# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the review from the JSON request
    data = request.get_json(force=True)
    
    
    review = data['review']

    if not data or 'review' not in data:
     return jsonify({'error': 'No review provided'}), 400

    # Preprocess the review (e.g., tokenization)
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=max_len)

    # Make the prediction
    prediction = model.predict(padded_sequence)
    sentiment = 'Positive' if prediction > 0.5 else 'Negative'

    # Return the result as JSON
    return jsonify({'sentiment': sentiment})

# Run the app
if __name__ == '__main__':
    app.run(debug=False)


from waitress import serve
serve(app, host='0.0.0.0', port=8000)