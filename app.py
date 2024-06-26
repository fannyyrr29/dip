from flask import Flask, request, jsonify
import numpy as np
import cv2 as cv
import io
from process import predict_result, preprocess_image

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['input_image']
    img_stream = io.BytesIO(file.read())
    img = cv.imdecode(np.frombuffer(img_stream.getvalue(), np.uint8), cv.IMREAD_COLOR)

    # Preprocess the image
    input_data = preprocess_image(img)

    # Make prediction
    prediction = predict_result(input_data)

    # Return prediction as JSON
    return jsonify({'prediction': str(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)

