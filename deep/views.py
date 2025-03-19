from django.shortcuts import render
from django.shortcuts import render
from .forms import ImageForm
from .models import Image
import joblib
from joblib import load
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout, Reshape, Concatenate, LeakyReLU
from keras.optimizers import Adam
from keras.models import Model
import joblib
from joblib import dump
from joblib import load
from django.shortcuts import render
from django.http import HttpResponse
import numpy as np
from keras.preprocessing import image
from io import BytesIO
import cv2



def upload(request):
    image_dimensions = {'height':256, 'width':256, 'channels':3}
    class Classifier:
        def __init__(self):
            self.model = 0

        def predict(self, x):
            return self.model.predict(x)

        def fit(self, x, y):
            return self.model.train_on_batch(x, y)

        def get_accuracy(self, x, y):
            return self.model.test_on_batch(x, y)

        def load(self, path):
            self.model.load_weights(path)
    
    class Meso4(Classifier):
        def __init__(self, learning_rate = 0.001):
            self.model = self.init_model()
            optimizer = Adam(learning_rate)
            self.model.compile(optimizer = optimizer,
                            loss = 'mean_squared_error',
                            metrics = ['accuracy'])

        def init_model(self):
            x = Input(shape = (image_dimensions['height'],
                            image_dimensions['width'],
                            image_dimensions['channels']))

            x1 = Conv2D(8, (3, 3), padding='same', activation = 'relu')(x)
            x1 = BatchNormalization()(x1)
            x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)

            x2 = Conv2D(8, (5, 5), padding='same', activation = 'relu')(x1)
            x2 = BatchNormalization()(x2)
            x2 = MaxPooling2D(pool_size=(2, 2), padding='same')(x2)

            x3 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x2)
            x3 = BatchNormalization()(x3)
            x3 = MaxPooling2D(pool_size=(2, 2), padding='same')(x3)

            x4 = Conv2D(16, (5, 5), padding='same', activation = 'relu')(x3)
            x4 = BatchNormalization()(x4)
            x4 = MaxPooling2D(pool_size=(4, 4), padding='same')(x4)

            y = Flatten()(x4)
            y = Dropout(0.5)(y)
            y = Dense(16)(y)
            y = LeakyReLU(alpha=0.1)(y)
            y = Dropout(0.5)(y)
            y = Dense(1, activation = 'sigmoid')(y)

            return Model(inputs = x, outputs = y)
        
    meso = Meso4()




   
    loaded_model=keras.models.load_model("C:/Users/ashwi/Downloads/Telegram Desktop/deepim/deep/deep/meso1.h5")
    def preprocess_image(image_file, target_size=(256, 256)):
        img = image.load_img(image_file, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Rescale pixel values between 0 and 1
        return img_array


      # Import OpenCV library for face detection

    def detect_face(image_stream):
        # Convert the image stream to numpy array
        nparr = np.frombuffer(image_stream.read(), np.uint8)
        img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # Convert bytes to image
        
        # Load the pre-trained face detection cascade
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # If no faces are detected, return False
        return len(faces) > 0

    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']
            image_stream = BytesIO(image_file.read())  # Convert to byte stream
            if not detect_face(image_stream):
                return render(request, 'invalid_image.html')  # Render a template for invalid image
            image_stream.seek(0)  # Reset stream position for further processing
            input_image = preprocess_image(image_stream)
            form.save()
            prediction = loaded_model.predict(input_image)[0][0]
            obj = form.instance
            threshold = 0.65
            prediction_label = "Real" if prediction >= threshold else "Fake"
            img = Image.objects.all()
            return render(request, 'result.html', {"img": img, 'prediction_label': prediction_label})
    else:
        form = ImageForm()
        img = Image.objects.all()
    return render(request, 'upload.html', {"img": img, "form": form})


       
    
def result(request):
    return(request,'result.html')

def invalid_image(request):
    return(request,'invalid_image.html')





