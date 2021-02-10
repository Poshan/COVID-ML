from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.regularizers import L1L2

import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from pandas import read_csv
from pandas import DataFrame
from pandas import concat

from keras.optimizers import schedules
from keras.optimizers import SGD
from numpy import concatenate

from sklearn.metrics import mean_squared_error
from math import sqrt



# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)

	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg


def scaleData(values, scaler_type):
  values = values.astype("float32")
  if (scaler_type == "min_max"):
    print("working on min max")
  elif (scaler_type == "standard"):
    # print(values.shape)
    # normalize features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(values)
    print(scaled.shape)
  else:
    print("scaling method not avialable")
  
  return (scaler,scaled)


def lstm(train_X, train_y, test_X, test_y, params):
  epochs = params["epochs"]
  batch_size = params["batch_size"]
  loss_function = params["loss_function"]
  optimizer = params["optimizer"]
  lstm_units = params["lstm_units"]
  # design network
  model = Sequential()

  # model - bidirectional
  # forward_layer = LSTM(lstm_units, return_sequences=True)
  # backward_layer = LSTM(lstm_units, activation='relu', return_sequences=True,
  #                      go_backwards=True)
  # model.add(Bidirectional(forward_layer, backward_layer=backward_layer,
  #                        input_shape=(train_X.shape[1], train_X.shape[2])))
  
  ###model - one layered
  # model.add(Bidirectional(LSTM(lstm_units, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2]))))

  ####model - 2 layererd lstms
  # model.add(LSTM(lstm_units, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
  # model.add(LSTM(lstm_units, return_sequences=True))


  ####model - 3 layererd lstms
  model.add(LSTM(128, return_sequences=True, input_shape=(train_X.shape[1], train_X.shape[2])))
  model.add(LSTM(64, return_sequences=True))
  model.add(LSTM(32))

  model.add(Dense(1))
  model.compile(loss=loss_function,optimizer=optimizer)
  # fit network
  history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(test_X, test_y), verbose=2, shuffle=False)
  print(history.history.keys())
  # plot history
  # plt.plot(history.history['loss'], label='train')
  # plt.plot(history.history['val_loss'], label='test')
  # plt.legend()
  
  # create figure and axis objects with subplots()
  fig,ax=plt.subplots()
  ax.plot(history.history['loss'], label = "training" )
  ax.set_ylabel("loss")
  ax.set_xlabel("epoches")
  ax.plot(history.history['val_loss'], label = "test")
  ax.legend()
  
  # plt= "test"
  return (model, history, fig)



def run_lstm(df, params):
  dates = params["dates"]
  training_size = params["training_size"]
  test_size = params["test_size"]
  values = df.values
  no_of_predictors = params["no_of_predictors"]
  norm_method = params["norm_method"]
  remove_cols = []
  for i in range(no_of_predictors + 1, no_of_predictors*2):
    remove_cols.append(i)

  # print(values.shape)

  #scaling the data
  scaler, scaled = scaleData(values, norm_method)

  # frame as supervised learning
  reframed = series_to_supervised(scaled, 1, 1)
  # print(reframed)
  # do this manually 
  # drop columns we don't want to predict
  reframed.drop(reframed.columns[remove_cols], axis=1, inplace=True)
  print(reframed.head(5))

  values = reframed.values
  # print(values.shape[0])

  # split into train and test sets
  train = values[:training_size, :]
  test = values[training_size:training_size+test_size, :]
  validation = values[training_size+test_size:,:]

  # split into input and outputs
  train_X, train_y = train[:, :-1], train[:, -1]
  test_X, test_y = test[:, :-1], test[:, -1]
  validation_X, validation_y = validation[:, :-1], validation[:, -1]
  # reshape input to be 3D [samples, timesteps, features]
  train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
  test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
  validation_X = validation_X.reshape(validation_X.shape[0], 1, validation_X.shape[1])


  # train_X.shape, train_y.shape, test_X.shape, test_y.shape = divide_test_train(values, 0.3)
  # print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

  #running the model
  model, history, plt_history = lstm(train_X, train_y, test_X, test_y, params)

  
  whole_X = values[:, :-1]
  whole_y = values[:, -1]

  # Make a prediction
  whole_X = whole_X.reshape(whole_X.shape[0], 1, whole_X.shape[1])
  yhat = model.predict(whole_X)
  print(yhat.shape)
  whole_X = whole_X.reshape((whole_X.shape[0], whole_X.shape[2]))
  # invert scaling for actual
  yhat = yhat.reshape(yhat.shape[0], 1)

  print(yhat.shape)

  inv_yhat = concatenate((yhat, whole_X[:, 1:]), axis=1)
  inv_yhat = scaler.inverse_transform(inv_yhat)
  inv_yhat = inv_yhat[:,0]

  whole_y = whole_y.reshape((len(whole_y), 1))
  inv_y = concatenate((whole_y, whole_X[:, 1:]), axis=1)
  inv_y = scaler.inverse_transform(inv_y)
  inv_y = inv_y[:,0]

  # calculate RMSE
  rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
  print('RMSE: %.3f' % rmse)

  # for plotting the predictions with the actual validation set

  
  print("printing the lenghts againa")
  
  b = dates[1:]
  print(len(inv_y))
  print(len(b))

  d = {"dt":b,
      "real":inv_y,
      "predicted":inv_yhat
      }
  prediction_df1 = pd.DataFrame(d)
  # prediction_df1.dt = pd.to_datetime(prediction_df1.dt)
  print(prediction_df1.info())
  # prediction_df1["date"] = pd.to_datetime(prediction_df1["dt"])
  # prediction_df['dt'] = pd.to_datetime(prediction_df['dt'], format = '%Y-%m-%d')

  fig1,ax1 = plt.subplots(figsize=(20,10))
  sns.set(style='darkgrid')
  sns.lineplot(x='dt', y='predicted', data=prediction_df1, alpha=0.5, color='green', ax=ax1, label= "prediction")
  sns.lineplot(x='dt', y='real', data=prediction_df1, alpha = 0.3, color ='black', ax=ax1, label= "real")
  # real.set_xticklabels(real.get_xticklabels(), rotation = 45)
  # ax1.tick_params(axis='x', which='major', labelsize=5, rotation = 45)
  # ax1.tick_params(axis='both', which='minor', labelsize=6)

  training_label_x = training_size/2.0
  test_label_x = training_size+(test_size/2.0)
  validation_label_x = training_size+test_size+(test_size/2.0)


  plt.axvline(x=training_size, ymin=0, ymax=1, color = "red")
  plt.text(0.35, 0.85, 'Training', transform=ax1.transAxes, fontsize = 18)
  plt.text(0.3, 0.9, 'March 1, 2020 - October 22, 2020',transform=ax1.transAxes)

  plt.axvline(x=training_size + test_size, ymin=0, ymax=1, color = "red")
  plt.text(0.7, 0.9, 'October 23, 2020 - November 06, 2020', transform=ax1.transAxes, fontsize = 8)
  plt.text(0.72, 0.85,'Test', transform=ax1.transAxes, fontsize = 18)

  plt.text(0.87, 0.85, "Validation", transform=ax1.transAxes, fontsize = 18)
  plt.text(0.85, 0.9, 'November 07, 2020 - November 13, 2020', transform=ax1.transAxes, fontsize = 6)
  rmse_txt = 'RMSE: ' + str(round(rmse,2))
  plt.text(0.99, 0.99, rmse_txt, transform=ax1.transAxes) 

  plt.ylabel("no of cases")
  plt.xlabel("date")
  plt.suptitle(params["plot_title"])
  plt.legend()

  return (rmse, plt_history, fig1, prediction_df1, model) 
