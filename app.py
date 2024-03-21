from flask import Flask, request, render_template, url_for, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

app = Flask(__name__)

def preprossing(image):
    image=Image.open(image)
    image = image.resize((224, 224))
    image_arr = np.array(image.convert('RGB'))
    image_arr.shape = (1, 224, 224, 3)
    image_arr=image_arr/255.0
    return image_arr

classes = ['AD' ,'CN', 'EMCI' ,'LMCI']
model=load_model("model_VGG16.h5")

@app.route('/')
def index():

    return render_template('index.html', appName="Alzhimer Image Classification")


@app.route('/predictApi', methods=["POST"])
def api():
    # Get the image from post request
    try:
        if 'fileup' not in request.files:
            return "Please try again. The Image doesn't exist"
        image = request.files.get('fileup')
        image_arr = preprossing(image)
        print("Model predicting ...")
        result = model.predict(image_arr)
        print("Model predicted")
        ind = np.argmax(result)
        prediction = classes[ind]
        print(prediction)
        return jsonify({'prediction': prediction,'Prob':round(result[0,ind],2)})
    except:
        return jsonify({'Error': 'Error occur'})


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    print("run code")
    if request.method == 'POST':
        # Get the image from post request
        print("image loading....")
        image = request.files['fileup']
        print("image loaded....")
        image_arr= preprossing(image)
        print("predicting ...")
        result = model.predict(image_arr)
        print("predicted ...")
        ind = np.argmax(result)
        prediction = classes[ind]+" Prob: "+str(round(result[0,ind],2))

        print(prediction)

        return render_template('index.html', prediction=prediction, image='static/IMG/', appName="Alzhimer Image Classification")
    else:
        return render_template('index.html',appName="Alzhimer Image Classification")


if __name__ == '__main__':
    app.run(debug=True)