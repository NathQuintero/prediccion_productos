# streamlit_audio_recorder y whisper by Alfredo Diaz - version Mayo 2024

# En VsC seleccione la version de Python (recomiendo 3.9) 
#CTRL SHIFT P  para crear el enviroment (Escriba Python Create Enviroment) y luego venv 

#o puede usar el siguiente comando en el shell
#Vaya a "view" en el menú y luego a terminal y lance un terminal.
#python -m venv env

#Verifique que el terminal inicio con el enviroment o en la carpeta del proyecto active el env.
#cd D:\flores\env\Scripts\
#cd C:\flores\venv\Scripts\
#.\activate 

#Debe quedar asi: (.venv) D:\proyectos_ia\Flores>

#Puedes verificar que no tenga ningun libreria preinstalada con
#pip freeze
#Actualicie pip con pip install --upgrade pip

#pip install tensorflow==2.15 La que tiene instalada Google Colab o con la versión qu fué entrenado el modelo
#Verifique se se instaló numpy, no trate de instalar numpy con pip install numpy, que puede instalar una version diferente
#pip install streamlit
#Verifique se se instaló no trante de instalar con pip install pillow
#Esta instalacion se hace si la requiere pip install opencv-python

#Descargue una foto de una flor que le sirva de ícono 

# importing the libraries and dependencies needed for creating the UI and supporting the deep learning models used in the project
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import streamlit as st  
import tensorflow as tf # TensorFlow is required for Keras to work
from PIL import Image
import numpy as np
import pyttsx3
#import cv2

# hide deprication warnings which directly don't affect the working of the application
import warnings
warnings.filterwarnings("ignore")

# Inicializar el motor de síntesis de voz
engine = pyttsx3.init()

# set some pre-defined configurations for the page, such as the page title, logo-icon, page loading state (whether the page is loaded automatically or you need to perform some action for loading)
st.set_page_config(
  page_title="¿Que producto es?",
  page_icon="icono.ico",
  initial_sidebar_state='auto',
  menu_items={
        'Report a bug': 'http://www.unab.edu.co',
        'Get Help': "https://docs.streamlit.io/get-started/fundamentals/main-concepts",
        'About': "Nathalia Quintero & Angelly Cristancho. Inteligencia Artificial *Ejemplo de clase* Ingenieria de sistemas!"
    }
  )

# hide the part of the code, as this is just for adding some custom CSS styling but not a part of the main idea 
hide_streamlit_style = """
	<style>
  #MainMenu {visibility: hidden;}
	footer {visibility: hidden;}
  </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True) # Oculta el código CSS de la pantalla, ya que están incrustados en el texto de rebajas. Además, permita que Streamlit se procese de forma insegura como HTML

#st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache_resource
def load_model():
    model=tf.keras.models.load_model('./modelofinalizado.h5')
    return model
with st.spinner('Modelo está cargando..'):
    model=load_model()
    


# Título de la página
st.image("./videos/banner.png", use_column_width=True)
st.title("Bienvenido a Mi Página con Streamlit")


with st.sidebar:
    option = st.selectbox(
    "Que te gustaria usar para subir la foto?",
    ("Tomar foto", "Subir archivo", "URL"),
    index=None,
    placeholder="Selecciona como subir la foto",
    )
    st.markdown("Cómo poner el producto correctamente en la camara?") 
    # Ruta del archivo de video
    video_file_path = './videos/SI.mp4'
    # Lee el contenido del archivo de video
    try:
        with open(video_file_path, 'rb') as video_file:
            video_bytes = video_file.read()

        # Reproduce el video
        st.video(video_bytes)
    except FileNotFoundError:
        st.error(f"El archivo de video no se encontró en la ruta: {video_file_path}")


    # Ruta del archivo de video
    video_file_path = './videos/NO.mp4'
    # Lee el contenido del archivo de video
    try:
        with open(video_file_path, 'rb') as video_file:
            video_bytes = video_file.read()

        # Reproduce el video
        st.video(video_bytes)
    except FileNotFoundError:
        st.error(f"El archivo de video no se encontró en la ruta: {video_file_path}")


def import_and_predict(image_data, model, class_names):
    
    image_data = image_data.resize((180, 180))
    
    image = tf.keras.utils.img_to_array(image_data)
    image = tf.expand_dims(image, 0) # Create a batch

    
    # Predecir con el modelo
    prediction = model.predict(image)
    index = np.argmax(prediction)
    score = tf.nn.softmax(prediction[0])
    class_name = class_names[index]
    
    return class_name, score


class_names = open("./clases.txt", "r").readlines()

img_file_buffer = st.camera_input("Capture una foto para identificar una flor")    
if img_file_buffer is None:
    st.text("Por favor tome una foto")
else:
    image = Image.open(img_file_buffer)
    st.image(image, use_column_width=True)
    
    # Realizar la predicción
    class_name, score = import_and_predict(image, model, class_names)
    
    # Mostrar el resultado

    if np.max(score)>0.5:
        msj= "Tipo de producto: ", class_name
        st.subheader(msj)
        st.text(f"Puntuación de confianza: {100 * np.max(score):.2f}%")
        engine.say(msj)
        engine.runAndWait()
    else:
        st.text(f"No se pudo determinar el tipo de flor")

if option== "Tomar foto":
    engine.say("Haz seleccionado camara")
    engine.runAndWait()
else:
    engine.say("Hola! soy beimax, tu asistente neuronal personal, ¿como te sientes hoy?")
    engine.runAndWait()

