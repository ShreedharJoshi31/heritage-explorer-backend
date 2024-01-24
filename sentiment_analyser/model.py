from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        text = data['text']

        tokens = tokenizer.encode(text, return_tensors='pt')
        result = model(tokens)
        logits = result.logits
        predicted_class = int(torch.argmax(logits)) + 1

        return jsonify({'sentiment_class': predicted_class})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
