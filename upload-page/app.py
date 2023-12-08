
from flask import Flask, render_template, request
import pickle
import numpy as np
from PIL import Image
from io import BytesIO
# import tensorflow as tf

from keras.models import load_model

app = Flask(__name__)

# with open('plant_project.pkl', 'rb') as model_file:
#     model = pickle.load(model_file)

model = load_model('model_new.h5')

@app.route('/')
def index():
    return render_template('upload-page.html')
    

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    uploaded_file = request.files['image']
    img = Image.open(BytesIO(uploaded_file.read()))
    tree_names=['Aloevera','Amla','Amruta_Balli','Arali','Ashoka','Ashwagandha','Avacado','Bamboo','Basale','Betel','Betel_Nut','Brahmi','Castor','Curry_Leaf','Doddapatre','Ekka','Ganike','Guava','Geranium','Henna','Hibiscus','Honge','Insulin','Jasmine',
    'Lemon','Lemon_grass','Mango','Mint','Nagadali','Neem','Nithyapushpa','Nooni','Pappaya','Pepper','Pomegranate','Raktachandini','Rose','Sapota','Tulsi','Wood_sorel']
    print(tree_names[predict_img(img)])
    plant_name = tree_names[predict_img(img)]

    pathh = "static\\images\\uploaded_image.jpeg"
    img.save(pathh)
    # print(res)
    # Perform any necessary preprocessing steps here
    # img_array = np.array(img)
    # prediction = model.predict(img_array.(1, -1))  # Adjust input shape as needed
    # return str(prediction[0])
    return render_template('result.html' , image = pathh , plant_name = plant_name)

def predict_img(img):
    IMG_SIZE = (1,299,299,3)

    img = img.resize(IMG_SIZE[1:-1])
    img_arr = np.array(img)
    img_arr = img_arr.reshape(IMG_SIZE)

    pred_proba = model.predict(img_arr)
    pred = np.argmax(pred_proba)
    return pred

# def df(model):
#     file = request.files['image']
#     img = Image.open(BytesIO(file.read()))
#     # img = Image.open(file.read())
#     # img = Image.open(file)
#     img = tf.keras.preprocessing.image.load_img(img, target_size=(299, 299))
#     img_array = tf.keras.preprocessing.image.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)  # Create a batch


#     predictions = model.predict(img_array)
#     score = tf.nn.sigmoid(predictions[0])
#     # print(
#     # "This image most likely belongs to {} with a {:.2f} percent confidence."
#     # .format(tree_names[np.argmax(score)], 100 * np.max(score)))
#     return tree_names[np.argmax(score)]

if __name__ == '__main__':
    app.run(debug=True)