import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('cnn_model.keras')

# Define the class names from the notebook
class_names = ['Glioma', 'Meninigioma', 'Notumor', 'Pituitary']


def predict_image(img):
    try:
        # Preprocess the image
        # Resize to the model's expected input shape (168, 168)
        img = image.smart_resize(img, (168, 168))
        # Convert PIL image to numpy array and ensure it's writable
        img_array = np.array(img)
        # Add channel dimension if it's a grayscale image
        if img_array.ndim == 2:
            img_array = np.expand_dims(img_array, axis=-1)
        # Add the batch dimension
        img_array = np.expand_dims(img_array, axis=0)

        # Normalize the image data
        img_array = img_array.astype('float32') / 255.0

        # Make a prediction
        prediction = model.predict(img_array)

        # Get the confidence scores for each class
        confidence_scores = {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}

        return confidence_scores
    except Exception as e:
        return str(e)

# Create the Gradio interface
iface = gr.Interface(fn=predict_image,
                     inputs=gr.Image(type="pil"),
                     outputs=gr.Label(num_top_classes=4),
                     title="Brain Tumor Classification",
                     description="Upload a brain MRI to classify the type of tumor.")

# Launch the app
iface.launch(share=True)