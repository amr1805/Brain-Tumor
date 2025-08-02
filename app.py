import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('brain_tumor_model.keras')

# Define the class names
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

def predict_image(img):
    # Preprocess the image
    img = image.smart_resize(img, (150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    # Make a prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)

    return {class_names[i]: float(prediction[0][i]) for i in range(len(class_names))}

# Create the Gradio interface
iface = gr.Interface(fn=predict_image,
                     inputs=gr.Image(type="pil"),
                     outputs=gr.Label(num_top_classes=4),
                     title="Brain Tumor Classification",
                     description="Upload a brain MRI to classify the type of tumor.")

# Launch the app
iface.launch()
