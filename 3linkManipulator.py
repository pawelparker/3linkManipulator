#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import csv
import random
from matplotlib import pyplot as plt


# In[2]:


# Lengths for three links of the Robotic Arm
LINK1 = 14.0
LINK2 = 22.0
LINK3 = 6.0

# No of rows/samples to generate for the dataset
SAMPLES = 1500

# Number of digits after decimal point to keep in the data points
PRECISION = 3

# Number of epochs to iterate while training the model
NUMBER_OF_EPOCHS = 100

ARROW_LENGTH = 0.7


# <h3><u>Generating Dataset</u></h3>

# - Creating the Helper Functions:

# In[3]:


#Angles are in Radian
def getXCoordinate(AngQ1, AngQ2, AngQ3):
    return LINK1*math.cos(AngQ1) + LINK2*math.cos(AngQ1 + AngQ2) + LINK3*math.cos(AngQ1 + AngQ2 + AngQ3)

def getYCoordinate(AngQ1, AngQ2, AngQ3):
    return LINK1*math.sin(AngQ1) + LINK2*math.sin(AngQ1 + AngQ2) + LINK3*math.sin(AngQ1 + AngQ2 + AngQ3)

def getTotalAngle(AngQ1, AngQ2, AngQ3):
    return math.degrees(AngQ1 + AngQ2 + AngQ3)


# In[4]:


def generateDataset():
    with open('training_data.csv', mode='w', newline='') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames = ["Q1", "Q2", "Q3", "XCoordEndEff", "YCoordEndEff", "TotalAngle"])
        writer.writeheader()

        for i in range (0,SAMPLES):
            #Randomly generating joint angles 
            q1= round(random.uniform(0, math.pi), PRECISION)
            q2= round(random.uniform(-math.pi, 0), PRECISION)
            q3= round(random.uniform(-math.pi/2, math.pi/2), PRECISION)

            writer.writerow({
                'Q1':q1,
                'Q2':q2,
                'Q3':q3,
                'XCoordEndEff':str(round(getXCoordinate(q1, q2, q3), PRECISION)),
                'YCoordEndEff':str(round(getYCoordinate(q1, q2, q3), PRECISION)),
                'TotalAngle':str(round(getTotalAngle(q1, q2, q3), PRECISION))
            })


# - Generate Dataset:

# In[5]:


#Uncomment the next line if to generate the dataset
#generateDataset()


# In[6]:


dataset = pd.read_csv('training_data.csv')
dataset


# In[7]:


coordX = dataset.iloc[:, [3]].values
coordY = dataset.iloc[:, [4]].values
thetaAng = dataset.iloc[:, [5]].values


# In[8]:


#function to plot data points
def plotDataset():
    plt.figure(figsize=(9, 7))
    for i in range(len(coordX)):
        posx = coordX[i][0]
        posy = coordY[i][0]
        theta = thetaAng[i][0]
        plt.plot([posx, posx + ARROW_LENGTH * math.cos(math.radians(theta))], [posy, posy + ARROW_LENGTH * math.sin(math.radians(theta))],'k-')

    plt.scatter(coordX, coordY)
    plt.xlabel("X Axis")
    plt.ylabel("Y Axis")
    plt.title("Data set of " + str(SAMPLES) + " possible endeffector positions and orientations")
    dataset_fig = plt.gcf()
    plt.show()
    plt.draw()
    #dataset_fig.savefig('Plot of Data Points.png', dpi=100, bbox_inches='tight')


# In[9]:


plotDataset()


# In[10]:


# Removing duplicated end effector positions
dups = dataset.duplicated(subset=["XCoordEndEff", "YCoordEndEff"])

# duplicated() method returns a series which contain false for all the unique values/rows
# Since the deduplicated dataset must contain only unique values we invert "dups" and select 
# all the rows which are true in "dups" 
dataset = dataset[~dups]
dataset


# In[11]:


#The X and Y Coordinates and the end effector orientation is the input to the model
X = dataset.iloc[:, 3:].values

#The joint angles are the output from the model
y = dataset.iloc[:, :3].values


# In[12]:


#Splitting dataset into training and test data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, shuffle=1, random_state=0)


# In[13]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
X_train = scaler.fit_transform(X_train)
scaler = MinMaxScaler(feature_range=(-1, 1))
X_test = scaler.fit_transform(X_test)


# In[14]:


# 70% - Training, 15% - Validation, 15% - Testing
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.18, random_state=0)#0.25


# In[15]:


import tensorflow as tf
from tensorflow import keras


# In[16]:


#Function to build the neural network model
def build_model():
     model = keras.Sequential()
     model.add(keras.layers.Dense(3, activation='linear', input_shape=(3, )))
     model.add(keras.layers.Dense(64, activation='relu'))
     model.add(keras.layers.Dense(100, activation='relu'))
     model.add(keras.layers.Dropout(0.3))
     model.add(keras.layers.Dense(3, activation='linear'))
     model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss=tf.losses.mean_squared_error, metrics=['accuracy'])
     return model


# In[17]:


#Building the model
model = build_model()


# In[18]:


#Training the model
history = model.fit(
    x=X_train, 
    y=y_train,
    epochs=NUMBER_OF_EPOCHS,
    verbose=2,
    validation_data=(X_val, y_val)
)


# In[19]:


#Evaluating model on testing data 
print('Testing the model:\n')

test_loss, test_acc = model.evaluate(X_test, y_test)

print('\nTesting Loss = ', test_loss) 
print('Testing Accuracy = ', test_acc)


# In[31]:


# plot the training loss and accuracy
N = 100
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), history.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), history.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), history.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")


# In[21]:


# Summarize history for accuracy

plt.figure(figsize=(15, 10))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.xticks(np.arange(0, NUMBER_OF_EPOCHS+1, 10))

model_acc_fig = plt.gcf()
plt.show()
plt.draw()


# In[22]:


# Summarize history for loss
plt.figure(figsize=(15, 10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss (Mean Squared Error)')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.xticks(np.arange(0, NUMBER_OF_EPOCHS+1, 10))
model_loss_fig = plt.gcf()
plt.show()
plt.draw()


# In[23]:


# Make Predictions
predictions = model.predict(X_test)


# In[24]:


# Scatter Plot for True Value vs Predictions of Joint Angles
plt.figure(figsize=(9, 9))
plt.scatter(y_test[:, 0], predictions[:, 0], label="Joint Angle 1 (Q1)", c="red",alpha=0.5)
plt.scatter(y_test[:, 1], predictions[:, 1], label="Joint Angle 2 (Q2)", c="blue",alpha=0.5)
plt.scatter(y_test[:, 2], predictions[:, 2], label="Joint Angle 3 (Q3)", c="green",alpha=0.5)

plt.title("True Value vs Predictions")
plt.xlabel('True Value of angle (rad)')
plt.ylabel('Predictions of angle (rad)')
plt.legend(loc='upper left')
actual_prediction_fig = plt.gcf()
plt.show()
plt.draw()


# In[25]:


# Get Predicted Coordinates

# We need to derive the predicted coordinates from the predicted joint angles.
predicted_coord = np.empty((len(predictions), 3))
for i in range(len(predictions)):
    ang = predictions[i, :]
    predicted_coord[i, 0] = getXCoordinate(ang[0], ang[1], ang[2])
    predicted_coord[i, 1] = getYCoordinate(ang[0], ang[1], ang[2])
    predicted_coord[i, 2] = getTotalAngle(ang[0], ang[1], ang[2])


# In[26]:


# Getting the Actual Coordinates
# We need to inverse transform to get the coordinates back to the original range
# as they we transformed to range [-1, 1] during feature scaling.
actual_coord = scaler.inverse_transform(X_test)


# In[27]:


# Plotting the Actual Position and the Predicted position of the end effector
plt.figure(figsize=(9, 7))
plt.scatter(actual_coord[:, 0], actual_coord[:, 1], label="Actual Value", c="blue",alpha=0.5)
plt.scatter(predicted_coord[:, 0], predicted_coord[:, 1], label="Predicted Value", c="green",alpha=0.5)

for i in range(len(actual_coord)):
        posx = actual_coord[i][0]
        posy = actual_coord[i][1]
        theta = actual_coord[i][2]
        plt.plot([posx, posx + ARROW_LENGTH * math.cos(math.radians(theta))], [posy, posy + ARROW_LENGTH * math.sin(math.radians(theta))],'k-')
        
        posx = predicted_coord[i][0]
        posy = predicted_coord[i][1]
        theta = predicted_coord[i][2]
        plt.plot([posx, posx + ARROW_LENGTH * math.cos(math.radians(theta))], [posy, posy + ARROW_LENGTH * math.sin(math.radians(theta))],'k-')

plt.legend(loc='upper left')
plt.xlabel("X Axis")
plt.ylabel("Y Axis")
plt.title("Actual Position and the Predicted position of the end effector")
actual_predicted_fig = plt.gcf()
plt.show()
plt.draw()


# In[28]:


model.save('model')


# In[ ]:




