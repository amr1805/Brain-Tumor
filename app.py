import gradio as gr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model (updated path)
model = load_model('cnn_model.keras')

# Define the class names (matching training notebook order/casing)
class_names = ['Glioma', 'Meninigioma', 'Notumor', 'Pituitary']

def predict_image(img):
    try:
        # Preprocess the image to 168x168 grayscale as per training
        # Convert to grayscale ('L') and resize
        img = img.convert('L').resize((168, 168))
        img_array = image.img_to_array(img)  # shape: (168, 168, 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Create a writable copy of the array
        img_array = img_array.copy()
        
        img_array /= 255.

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