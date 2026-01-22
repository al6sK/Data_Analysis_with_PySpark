import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score

plots_path = 'PLOTS/04_RNN'
os.makedirs(plots_path, exist_ok=True)
sns.set_theme(style="darkgrid")

data = pd.read_csv("DATA/log_data_Downsampled.csv", index_col='Timestamp', parse_dates=True)

# ===========================================================================================
# Data preparacion
# ===========================================================================================

# Split the dataset to : input_features , target_features
r_columns = ["R" + str(i) for i in range(1, 15)]
target_features = data[r_columns]
input_features = data.drop(columns = r_columns)

# Normalization of features
features_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaler = MinMaxScaler(feature_range=(0, 1))

scaled_input_features = features_scaler.fit_transform(input_features)
scaled_target_features = target_scaler.fit_transform(target_features)
# ===========================================================================================
# create_3d_dataset
# ===========================================================================================
LOOK_BACK = 10
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
def create_3d_dataset(features_df, targets_df,time_steps):
    X = []
    Y = []
    for i in range(len(features_df) - time_steps):
        batch_of_features = features_df[ i : (i + time_steps)]
        X.append(batch_of_features)
        Y.append(targets_df[i + time_steps])
    return np.array(X), np.array(Y)

X , Y = create_3d_dataset(scaled_input_features,scaled_target_features,LOOK_BACK)

# Split data to 70% - 15% - 15% for train, val and test 
data_size = len(X)
train_end = int(data_size * 0.7)
val_end = int(data_size * 0.85)

X_train = X[:train_end]
Y_train = Y[:train_end ]

X_val = X[train_end : val_end]
Y_val = Y[train_end : val_end]

X_test = X[val_end : ]
Y_test = Y[val_end : ] 

print("X_train shape:", X_train.shape, "Y_train shape:", Y_train.shape)
print("X_test shape:", X_test.shape, "Y_test shape:", Y_test.shape)

# =======================================================================================================================================
# MODEL CONFIGURATION!
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# [ LOOK_BACK ] --- | The window of time steps that will go as input to the model to predict Y the target (which Y is 14 values at once!).
# [ EPOCHS ] ------ | How many times will the model see COMPLETELY all the dataset.
# [ PATIENCE ] ---- | How many epochs will run without getting a better val_loss.
# [ BATCH ] ------- | How many "rows" of data the model will have to see before adjusting the weights.
# [ LSTM_NEURONS ]- | LSTM memory space
# [ N_FEATURES ] -- | The number of input features
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 

LOOK_BACK       # This has been set before in the Data preparacion, so that the data shape is the same as the models input shape
EPOCHS          = 3000
PATIENCE        = 50
BATCH           = 5
LSTM_NEURONS    = 128
N_FEATURES      = X_train.shape[2]
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,       
    patience=35,       
    min_lr=0.00001,  
    verbose=1
)
# ===========================================================================================
# Helper to build an LSTM model
# ===========================================================================================
def build_lstm(
    look_back: int,
    n_features: int,
    dropout: float = 0.0,
    recurrent_dropout: float = 0.0,
    ) -> keras.Model:

    inputs = keras.Input(shape=(look_back,n_features))
    
    x = keras.layers.LSTM(
        LSTM_NEURONS,
        dropout=dropout,
        recurrent_dropout=recurrent_dropout,
    )(inputs)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(1048, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)

    x = keras.layers.Dense(28, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)


    outputs = keras.layers.Dense(14, activation="linear")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model

# ===========================================================================================
# Building the LSTM model
# ===========================================================================================
model = build_lstm(
    look_back= LOOK_BACK, 
    n_features= N_FEATURES, 
    dropout= 0.0, 
    recurrent_dropout= 0.0
)
# ===========================================================================================
# Model training with EarlyStopping
# ===========================================================================================
callback = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

history = model.fit(
    X_train,
    Y_train,
    epochs= EPOCHS,
    batch_size= BATCH,
    validation_data=(X_val, Y_val),
    callbacks=[callback,reduce_lr],
    shuffle=False,
)
# ===========================================================================================
# Model predictions
# ===========================================================================================
predictions = model.predict(X_test)

# Invert the scaled values
predictions_log = target_scaler.inverse_transform(predictions)
Y_test_log = target_scaler.inverse_transform(Y_test)

# Invert Log values
predictions_real = np.expm1(predictions_log)
Y_test_real = np.expm1(Y_test_log)

# ===========================================================================================
# Evaluation
# ===========================================================================================
mape = mean_absolute_percentage_error(Y_test_real, predictions_real) * 100
rmse = np.sqrt(mean_squared_error(Y_test_real, predictions_real))
mae  = mean_absolute_error(Y_test_real, predictions_real)
r2   = r2_score(Y_test_real, predictions_real)

print("\n" + "="*40)
print(" LSTM FINAL RESULTS")
print("="*40)
print(f"MAPE: {mape:.4f} %")  
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R2:   {r2:.4f}")
print("="*40)