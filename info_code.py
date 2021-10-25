Predicting wind speed and direction is one of the most crucial as well as critical tasks in a wind farm because wind turbine blades motion and energy production is closely related to the behavior of the wind flow. We will design a Machine learning Neural Network-based model to predict the speed and the direction of the wind.

Wind direction prediction
The dataset we have taken here is an open dataset from ‘meteoblue’ consisting of Date, Minimum temperature, Maximum temperature, Mean temperature, Minimum speed, Maximum speed, Mean speed, and direction. These values are from Jan 1st, 2019 to Dec 31st, 2019.

We will use pandas to extract the data frame from the CSV file, keras.models to build the model which uses TensorFlow at the backend, scikit-learn for preprocessing and matplotlib to plot prediction values.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import model_selection
Now, we will read the data frame from the CSV file using pandas.

df = pd.read_csv('data_wind.csv')
The resultant data-frame is shown below:

dataframe

We will drop the precipitation column as it has many sparse values that can randomly affect our model.

df = df.drop('precipitation',1)
Then take Minimum temperature, Maximum temperature, Mean temperature, Minimum speed, Maximum speed, and Mean speed as input and direction as the required output.

x = df.iloc[:,1:-1]
y = df.iloc[:,-1]
Split the input and output as training & testing data where testing data consists of 90% and rest is testing data, both being randomly sampled out the dataset.

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.1, random_state=4)
Now, use keras.models to build the model which uses TensorFlow at the backend. We will use Standard scalar to normalize the data.

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
xnorm = StandardScaler();
ynorm = StandardScaler();
x_train=xnorm.fit_transform(x_train)
x_test=xnorm.transform(x_test)
y_train=ynorm.fit_transform(np.array(y_train).reshape(-1,1))
y_test=ynorm.transform(np.array(y_test).reshape(-1,1))
Now, we will use 7 dense layers consisting of 512, 256, 128, 64, 32, 16, & 1 neurons respectively. Activation function, loss & optimizer used are relu, Mean Square Error, and adam, respectively. The model is trained in 30 epochs with batch-size 32.

model = Sequential()
model.add(Dense(512, input_shape=(6,), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
print(model.summary())
model.fit(x_train,y_train, epochs=30, batch_size=32)
# predictions 
trainPredict = model.predict(x_train)
testPredict = model.predict(x_test)
Now, plot the predicted values using matplotlib to visualize.

plt.plot(range(0,y_train.shape[0]),ynorm.inverse_transform(y_train), label='y_train')
plt.plot(range(y_train.shape[0],y_train.shape[0]+y_test.shape[0]),ynorm.inverse_transform(y_test), label='y_test')
plt.xlabel('Day')
plt.ylabel('Mean Direction')
plt.title('Wind Direction Prediction')
plt.legend()
plt.show()
Dir 1 graph

plt.plot(range(0,y_train.shape[0]),ynorm.inverse_transform(y_train), label='y_train')
plt.plot(range(y_train.shape[0],y_train.shape[0]+y_test.shape[0]),ynorm.inverse_transform(testPredict), label='testPredict')
plt.xlabel('Day')
plt.ylabel('Mean Direction')
plt.title('Wind Direction Prediction')
plt.legend()
plt.show()
Dir 2 grp

Now, we will find the training & testing of MSE and RMSE scores.

import math
# Estimate model performance
trainingScore = model.evaluate(x_train, y_train)
print('Training Score is : %.2f MSE (%.2f RMSE)' % (trainingScore, math.sqrt(trainingScore)))
testingScore = model.evaluate(x_test, y_test)
print('Testing Score is  : %.2f MSE (%.2f RMSE)' % (testingScore, math.sqrt(testingScore)))
328/328 [==============================] - 0s 86us/step
Training Score is : 0.55 MSE (0.74 RMSE)
37/37 [==============================] - 0s 155us/step
Testing Score is  : 0.96 MSE (0.98 RMSE)
Wind speed prediction
Similarly, we will train our model for speed prediction. Now we will take Minimum temperature, Maximum temperature, Mean temperature, Minimum speed, Maximum speed, and direction as input and Mean speed as required output.

x2 = df[['min_temp','max_temp','mean_temp','min_speed','max_speed','direction']]
y2 = df[['mean_speed']]
Split data into training and testing samples as 90% and 10%, respectively.

x2_train, x2_test, y2_train, y2_test = model_selection.train_test_split(x2, y2, test_size=0.1, random_state=4)
Normalize data as follows:

from keras.models import Sequential
from keras.models import Model
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
xnorm = StandardScaler();
ynorm = StandardScaler();
x2_train=xnorm.fit_transform(x2_train)
x2_test=xnorm.transform(x2_test)
y2_train=ynorm.fit_transform(np.array(y2_train).reshape(-1,1))
y2_test=ynorm.transform(np.array(y2_test).reshape(-1,1))
Design model as the previous one.

model = Sequential()
model.add(Dense(512, input_shape=(6,), activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
print(model.summary())
model.fit(x2_train,y2_train, epochs=30, batch_size=32)
# predictions
trainPredict2 = model.predict(x2_train)
testPredict2 = model.predict(x2_test)
Plot the resultant predictions using matplotlib.

plt.plot(range(0,y2_train.shape[0]),ynorm.inverse_transform(y2_train), label='y2_train')
plt.plot(range(y2_train.shape[0],y2_train.shape[0]+y2_test.shape[0]),ynorm.inverse_transform(y2_test), label='y2_test')
plt.xlabel('Day')
plt.ylabel('Mean Speed')
plt.title('Wind Speed Prediction')
plt.legend()
plt.show()
Speed 1 grp

plt.plot(range(0,y2_train.shape[0]),ynorm.inverse_transform(y2_train),label='y2_train')
plt.plot(range(y2_train.shape[0],y2_train.shape[0]+y2_test.shape[0]),ynorm.inverse_transform(testPredict2), label='testPredict2')
plt.xlabel('Day')
plt.ylabel('Mean Speed')
plt.title('Wind Speed Prediction')
plt.legend()
plt.show()
Speed 1 grp

Finally, we will find the training & testing scores.

import math
trainingScore = model.evaluate(x2_train, y2_train)
print('Training Score is : %.3f MSE (%.3f RMSE)' % (trainingScore, math.sqrt(trainingScore)))
testingScore = model.evaluate(x2_test, y2_test)
print('Testing Score is  : %.3f MSE (%.3f RMSE)' % (testingScore, math.sqrt(testingScore)))
328/328 [==============================] - 0s 69us/step
Training Score is : 0.027 MSE (0.164 RMSE)
37/37 [==============================] - 0s 96us/step
Testing Score is  : 0.108 MSE (0.328 RMSE)
