"""import cv2
import os
from PIL import Image
import numpy as np

from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils import to_categorical
image_directory = "datasets/"
no_tumor_images = os.listdir(image_directory+'no/')
yes_tumor_images = os.listdir(image_directory+'yes/')
dataset = []
label = []
INPUT_SIZE = 64

#print (no_tumor_images)
#path="no0.jpg'
#prin(path.split(.)[1])

for i, image_name in enumerate(no_tumor_images):

    if(image_name.split('.')[1]=='jpg'):

        image = cv2.imread(image_directory + 'no/'+image_name)
        image = Image.fromarray(image, 'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(0)

for i, image_name in enumerate(yes_tumor_images):
    if(image_name.split('.')[1]=='jpg'):
        image = cv2.imread(image_directory + 'yes/'+image_name)
        image = Image.fromarray(image,'RGB')
        image = image.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(image))
        label.append(1)

dataset=np.array(dataset)
label=np.array(label)
x_train, x_test, y_train, y_test = train_test_split(dataset,label,test_size=0.2,random_state=0)

# x_train=no of images for training purpose(2400,64,64,3)
#x_test=no of images for testing purpose
#Reshape=(n, image_width,imagr_height,n_channel)
#print(x_train.shape)
#print(y_train.shape)
#print(x_test.shape)
#print(y_test.shape)

x_train= normalize(x_train,axis=1)
x_test=normalize(x_test,axis=1)
y_train=to_categorical(y_train, num_classes=2)
y_test=to_categorical(y_test, num_classes=2)


#Model Builing

model = Sequential()
#model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(height, width, channels)))
model.add(Conv2D(32, (3, 3),input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3),kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

#Binary CrossEntropy=1,sigmoid
#Categorical CrossEntropy=2,softmax

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=16,verbose=1,epochs=10,validation_data=(x_test,y_test),shuffle=False)

model.save('BrainTumor10EpochsCategorical.h5')"""



"""import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

#define augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,  # randomly rotate images within 20 degrees
    width_shift_range=0.2,  # randomly shift images horizontally within 20% of the width
    height_shift_range=0.2,  # randomly shift images vertically within 20% of the height
    horizontal_flip=True  # randomly flip images horizontally
)

# Define the morphological operation parameters
kernel_size = (3, 3)
iterations = 1

# Custom implementation of Gaussian filter
def gaussian_filter(image, sigma):
    # Define the Gaussian kernel
    kernel_size = 3
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    sum = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            sum += kernel[i, j]

    # Normalize the kernel
    kernel /= sum

    # Convolve the image with the Gaussian kernel
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

# Custom implementation of Median filter
def median_filter(image, kernel_size):
    filtered_image = np.copy(image)
    rows, cols = image.shape
    padding = kernel_size // 2

    for i in range(padding, rows - padding):
        for j in range(padding, cols - padding):
            patch = image[i - padding:i + padding + 1, j - padding:j + padding + 1]
            filtered_image[i, j] = np.median(patch)

    return filtered_image

try:
    image_directory = "datasets/"

    no_tumor_images = os.listdir(image_directory+'no/')#no tumor images in this list
    yes_tumor_images = os.listdir(image_directory+'yes/')# tumor images in this list
    dataset = []
    label = []
    INPUT_SIZE = 64

    for i, image_name in enumerate(no_tumor_images):
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(image_directory + 'no/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            # Apply Gaussian filter
            image = gaussian_filter(np.array(image), sigma=1.0)
            dataset.append(np.array(image))
            label.append(0)

    for i, image_name in enumerate(yes_tumor_images):
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(image_directory + 'yes/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            # Apply Median filter
            image = median_filter(np.array(image), kernel_size=3)
            dataset.append(np.array(image))
            label.append(1)

    dataset = np.array(dataset)
    label = np.array(label)
    x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

    x_train = normalize(x_train, axis=1)
    x_test = normalize(x_test, axis=1)
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    # Apply data augmentation to x_train
    datagen.fit(x_train)

    # Model Building
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=150, validation_data=(x_test, y_test))

    # Remaining code for training, evaluation, and saving the model...

except Exception as e:
    print("An error occurred:", e)
apply an median filter without importing libraries and then use threshold segmentation and then apply morphological process"""

"""import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

#define augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,  # randomly rotate images within 20 degrees
    width_shift_range=0.2,  # randomly shift images horizontally within 20% of the width
    height_shift_range=0.2,  # randomly shift images vertically within 20% of the height
    horizontal_flip=True  # randomly flip images horizontally
)
# Custom implementation of Gaussian filter
def gaussian_filter(image, sigma):
    # Define the Gaussian kernel
    kernel_size = 3
    kernel = np.zeros((kernel_size, kernel_size))
    center = kernel_size // 2
    sum = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            x = i - center
            y = j - center
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            sum += kernel[i, j]

    # Normalize the kernel
    kernel /= sum

    # Convolve the image with the Gaussian kernel
    filtered_image = cv2.filter2D(image, -1, kernel)
    return filtered_image

# Custom implementation of Median filter
def median_filter(image, kernel_size):
    # Check if the image is already grayscale
    if len(image.shape) == 2:
        gray_image = image
    else:
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create a copy of the image for filtered output
    filtered_image = np.copy(gray_image)

    # Get the dimensions of the image
    rows, cols = gray_image.shape

    # Calculate the padding size
    padding = kernel_size // 2

    # Apply median filter
    for i in range(padding, rows - padding):
        for j in range(padding, cols - padding):
            patch = gray_image[i - padding:i + padding + 1, j - padding:j + padding + 1]
            filtered_image[i, j] = np.median(patch)

    return filtered_image


# Custom implementation of threshold segmentation
def threshold_segmentation(image, threshold):
    if len(image.shape) == 2:
        gray_image = image
    # Convert the image to grayscale
    else:
        # Convert image to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Apply thresholding
    _, segmented_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return segmented_image

 # Custom implementation of erode
def erode(image, kernel):
    rows, cols = image.shape
    padding = kernel.shape[0] // 2
    eroded_image = np.zeros_like(image)

    for i in range(padding, rows - padding):
        for j in range(padding, cols - padding):
            patch = image[i - padding:i + padding + 1, j - padding:j + padding + 1]
            eroded_image[i, j] = np.min(patch * kernel)

    return eroded_image


# Define the morphological operation parameters
kernel_size = (3, 3)
iterations = 1


image_directory = "datasets/"

no_tumor_images = os.listdir(image_directory+'no/')#no tumor images in this list
yes_tumor_images = os.listdir(image_directory+'yes/')# tumor images in this list
dataset = []
label = []
INPUT_SIZE = 64

for i, image_name in enumerate(no_tumor_images):
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(image_directory + 'no/' + image_name)
            if Image is not None:

             image = Image.fromarray(image, 'RGB')
             image = image.resize((INPUT_SIZE, INPUT_SIZE))
            # Apply Gaussian filter
             #image = gaussian_filter(np.array(image), sigma=1.0)
            # Apply Median filter
             image = median_filter(np.array(image), kernel_size=3)
            # Apply thresholding
             image = threshold_segmentation(image, threshold=127)
            # Convert the grayscale image to binary image
             image = np.where(image > 0, 255, 0)
            # Apply erode
             kernel = np.ones((3, 3), np.uint8)  # Define the erode kernel
             image = erode(image, kernel=kernel)
             dataset.append(np.array(image))
             label.append(0)

for i, image_name in enumerate(yes_tumor_images):
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(image_directory + 'yes/' + image_name)

            if Image is not None:

             image = Image.fromarray(image, 'RGB')
             image = image.resize((INPUT_SIZE, INPUT_SIZE))
            # Apply Gaussian filter
             #image = gaussian_filter(np.array(image), sigma=1.0)
            # Apply Median filter
             image = median_filter(np.array(image), kernel_size=3)
            #  Apply thresholding
             image = threshold_segmentation(image, threshold=127)
            # Convert the grayscale image to binary image
             image = np.where(image > 0, 255, 0)
             kernel = np.ones((3, 3), np.uint8)  # Define the erode kernel
             image = erode(image, kernel=kernel)
             dataset.append(np.array(image))
             label.append(1)

dataset = np.array(dataset)
label = np.array(label)
x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)
x_train=x_train.reshape(-1, 64, 64, 1)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)
y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)
# Apply data augmentation to x_train
datagen.fit(x_train)
    # Model Building
model = Sequential()

#conv 2d(no of filters,kernel size,size of image) 
model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
    #model.add(Activation('sigmoid'))

#model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
#model.add(Activation('sigmoid'))
#model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
#model.add(Activation('sigmoid'))
#model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=50, validation_data=(x_test, y_test))

    # Plot training and validation accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
# Predict labels for test data
# Predict labels for test data
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Convert categorical labels back to binary
y_test_labels = np.argmax(y_test, axis=1)

# Generate confusion matrix
cm = confusion_matrix(y_test_labels, y_pred)

# Create confusion matrix plot
labels = ['No Tumor', 'Tumor']
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()

    # Add title and axis labels
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Show the plot
plt.show()
# Model Training
model.fit(datagen.flow(x_train, y_train, batch_size=16), verbose=1, epochs=50, validation_data=(x_test, y_test))

# Train the model and store the history

model.save('BrainTumor1EpochsCategorical.h5')"""
import cv2
import os
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

#define augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,  # randomly rotate images within 20 degrees
    width_shift_range=0.2,  # randomly shift images horizontally within 20% of the width
    height_shift_range=0.2,  # randomly shift images vertically within 20% of the height
    horizontal_flip=True  # randomly flip images horizontally
)

# Define the morphological operation parameters
kernel_size = (3, 3)
iterations = 1

try:
    image_directory = "datasets/"

    no_tumor_images = os.listdir(image_directory+'no/')#no tumor images in this list
    yes_tumor_images = os.listdir(image_directory+'yes/')# tumor images in this list
    dataset = []
    label = []
    INPUT_SIZE = 64

    for i, image_name in enumerate(no_tumor_images):
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(image_directory + 'no/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            # Apply morphological operation (e.g., erosion or dilation)
            image = cv2.erode(np.array(image), kernel_size, iterations=iterations)
            dataset.append(np.array(image))
            label.append(0)

    for i, image_name in enumerate(yes_tumor_images):
        if image_name.split('.')[1] == 'jpg':
            image = cv2.imread(image_directory + 'yes/' + image_name)
            image = Image.fromarray(image, 'RGB')
            image = image.resize((INPUT_SIZE, INPUT_SIZE))
            # Apply morphological operation (e.g., erosion or dilation)
            image = cv2.erode(np.array(image), kernel_size, iterations=iterations)
            dataset.append(np.array(image))
            label.append(1)

    dataset = np.array(dataset)
    label = np.array(label)
    x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

    x_train = normalize(x_train, axis=1)
    x_test = normalize(x_test, axis=1)
    y_train = to_categorical(y_train, num_classes=2)
    y_test = to_categorical(y_test, num_classes=2)
    # Apply data augmentation to x_train
    datagen.fit(x_train)




    # Model Building
    model = Sequential()

#conv 2d(no of filters,kernel size,size of image)
    model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    model.add(Dense(2))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(x_train, y_train, batch_size=16, verbose=1, epochs=10, validation_data=(x_test, y_test))

    # Plot training and validation accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    # Plot training and validation loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    # Predict labels for test data
    # Predict labels for test data
    y_pred_probs = model.predict(x_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Convert categorical labels back to binary
    y_test_labels = np.argmax(y_test, axis=1)

    # Generate confusion matrix
    cm = confusion_matrix(y_test_labels, y_pred)

    # Create confusion matrix plot
    labels = ['No Tumor', 'Tumor']
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()

    # Add title and axis labels
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Show the plot
    plt.show()
    # Model Training
    model.fit(datagen.flow(x_train, y_train, batch_size=16), verbose=1, epochs=10, validation_data=(x_test, y_test))

    # Train the model and store the history

    model.save('BrainTumor10EpochsCategorical.h5')

except Exception as e:
    print("An error occurred:", e)



