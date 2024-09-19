from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import json

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load the trained model once when the app starts
model = tf.keras.models.load_model('./modalConfig/sarcasm_model.keras')

# Load the tokenizer's word index
with open("./modalConfig/tokenizer_word_index.json", 'r') as f:
    word_index = json.load(f)

# Create the tokenizer with the loaded word index
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.word_index = word_index

# Load the max sequence length
with open("./modalConfig/max_sequence_length.json", 'r') as f:
    max_length_data = json.load(f)
    max_length = max_length_data["max_sequence_length"]

@app.route("/")
def hello():
    return "Congratulations, Your Server is Working!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Ensure the input is provided and valid
        sentence = request.json.get('sentence', None)
        if not sentence:
            return jsonify({"error": "Please provide a valid sentence for prediction"}), 400

        # Preprocess the sentence (tokenize and pad)
        sequences = tokenizer.texts_to_sequences([sentence])
        padded = pad_sequences(sequences, maxlen=max_length)

        # Make a prediction
        prediction = model.predict(padded)[0][0]

        # Return whether it's sarcastic or not
        is_sarcastic = bool(round(prediction))
        result = "Sarcastic" if is_sarcastic else "Not Sarcastic"

        return jsonify({"sentence": sentence, "prediction": result, "confidence": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)}), 500
