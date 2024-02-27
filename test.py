from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
from io import BytesIO
import numpy as np

app = Flask(__name__)

# Load your pre-trained model
model = load_model('car_types_model.keras')
print(model.summary())

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        # Read the file into a buffer
        img_buffer = BytesIO()
        file.save(img_buffer)
        img_buffer.seek(0)

        # Load the image from the buffer
        img = image.load_img(img_buffer, target_size=(150, 150))  # Adjust target_size as per your model's input
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array)

        # Assuming your classes are in the order mentioned in your question
        classes = ['Convertible', 'Coupe', 'Hatchback', 'Pick-up', 'Sedan', "SUV", "VAN"]

        # Get the index of the highest probability
        predicted_class_index = np.argmax(prediction[0])
        predicted_class = classes[predicted_class_index]

        # Create a response object with the predicted class
        response = {
            'prediction': predicted_class,
            'confidence': float(np.max(prediction[0]))
        }

        print(response)

        return _corsify_actual_response(jsonify(response))


if __name__ == '__main__':
    app.run(debug=True)
