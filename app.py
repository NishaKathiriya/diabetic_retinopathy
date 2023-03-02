#from keras import models
from tensorflow.keras.models import load_model
import gradio as gr
import tensorflow as tf



model_load=load_model(r'D:\DR_detection\model.h5')

#model_load=mod=load_model(r"D:\DR_detection\model.h5")
class_names=['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

def prediction(image):
    image = image.reshape((1, 224, 224, 3))
    image=tf.keras.applications.mobilenet.preprocess_input(image)
    prediction = model_load.predict(image).flatten()
    return {class_names[i]: float(prediction[i]) for i in range(5)}

image1 = gr.inputs.Image(shape=(224,224))
label1 = gr.outputs.Label(num_top_classes=5)

# Gradio interface to input an image and see its prediction with percentage confidence
gr.Interface(fn=prediction, inputs=image1, outputs=label1,
             #theme="huggingface",
             title="Diabetic Retinopathy",
             allow_flagging=False,
             layout="vertical",
             live=True,
             capture_session=True,
             interpretation='default').launch(debug='True',share=True)