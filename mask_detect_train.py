# Import the required packages
# Preprocessing package
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img

# Application Package
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Model design imports
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Input
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense


from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from imutils import paths

import matplotlib.pyplot as plt
import numpy as np
import os


# Set the training parameters the initial learning  rate, number of epochs and  batch size
INIT_LR = 1e-4
EPOCHS = 15
BS = 32

DIRECTORY = r'D:\DK\Machine_Learning\FaceMaskDetection\FaceMaskLiveDetection\dataset'
CATEGORIES = ['with_mask', 'without_mask']

# get the list of images in our dataset directory and then init the list with data and labels

print('[INFO] Loading Images...')

data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path) :
        imagepath = os.path.join(path, img)
        image = load_img(imagepath, target_size=(224,224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)

# perform one hot encoding on the labels

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

# Convert to numpy array for deep learning processing
data = np.array(data, dtype='float32')
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.20, stratify=labels, random_state=42)

# construct the training image genrator for data  augmentation
aug = ImageDataGenerator(rotation_range=20, 
                        width_shift_range=0.2,
                        height_shift_range=0.2,
                        shear_range=0.15,
                        zoom_range=0.15,
                        horizontal_flip=True,
                        fill_mode='nearest')


# Load the base model as MobileNet network
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))

# construct the head model which will go before the base model

headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation='softmax')(headModel)

# Place the head model on top of base model
model = Model(inputs=baseModel.input, outputs=headModel)

# freeze the layers in the basemodel
# so that they are not updated in the inital training process

for layer in baseModel.layers:
    layer.trainable = False

# compile the model
print('[INFO] Compiling the model')
model.compile(optimizer=Adam(lr = INIT_LR, decay=INIT_LR/EPOCHS), loss='binary_crossentropy', metrics=['accuracy'])

# train the head of the network
print('[INFO] Training head.....')

H = model.fit(aug.flow(X_train,y_train, batch_size=BS),
    steps_per_epoch = len(X_train)//BS,
    validation_data = (X_test, y_test),
    validation_steps = len(X_test)//BS,
    epochs = EPOCHS)


print('[INFO].. Evaluation model...')
predictions = model.predict(X_test, batch_size=BS)

predictions = np.argmax(predictions, axis=1)

print(classification_report(y_test.argmax(axis=1), predictions, target_names=lb.classes_))

print('[INFO] saving mask detector model....')
model.save('mask_detector.model', save_format='h5')

# plot the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")