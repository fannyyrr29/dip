from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2 as cv
import base64
from process import predict_result, preprocess_image

app = Flask(__name__)
CORS(app)


@app.route('/predict', methods=['POST'])
def predict():
    if 'base64Image' in request.json:
        base64Image = request.json['base64Image']

        # Decode base64 to image data
        try:
            decoded_data = base64.b64decode(base64Image)
            np_data = np.frombuffer(decoded_data, np.uint8)

            img = cv.imdecode(np_data, cv.IMREAD_COLOR)

            # Preprocess the image
            input_data = preprocess_image(img)

            # Make prediction
            label = predict_result(input_data)

            # Return prediction as JSON
            return jsonify({'label': label[0]})
        except Exception as e:
            return jsonify({'error': str(e)}), 400
    else:
        return jsonify({'error': 'No base64Image found in request'}), 400


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=True)
