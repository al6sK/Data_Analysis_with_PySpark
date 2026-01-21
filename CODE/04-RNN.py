import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import seaborn as sns
import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler


plots_path = 'PLOTS/04_FEATURE ENGINEERING'
os.makedirs(plots_path, exist_ok=True)
sns.set_theme(style="darkgrid")

data = pd.read_csv("DATA/log_data_Downsampled.csv", index_col='Timestamp', parse_dates=True)
# ===========================================================================================
# Data preparacion
# ===========================================================================================


# Normalization

LOOK_BACK = 5
def create_sequance(dataset,seq_len):
    X = []
    Y = []
    for i in range(len(dataset) - seq_len):
        seq = dataset[ i : (i + seq_len)]
        






# =======================================================================================================================================
# MODEL CONFIGURATION!
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
# [ LOOK_BACK ] --- | The window of time steps that will go as input to the model to predict Y the target (which Y is 14 values at once!).
# [ EPOCHS ] ------ | How many times will the model see COMPLETELY all the dataset.
# [ PATIENCE ] ---- | How many epochs will run without getting a better val_loss.
# [ BATCH ] ------- | How many "rows" of data the model will have to see before adjusting the weights.
# [ VALID_SPLIT ] - | What % (in decimal) of the train data, will go for validation. 
# [ LSTM_UNITS ] -- | LSTM memory space
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 

LOOK_BACK       # This has been set before in the Data preparacion, so that the data shape is the same as the models input shape
EPOCHS          = 30
PATIENCE        = 10
BATCH           = 32
VALID_SPLIT     = 0.2
LSTM_NEURONS    = 128
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 

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
    # x = keras.layers.Dropout(0.2)(x)

    # x = keras.layers.Dense(64, activation="relu")(x)
    # x = keras.layers.Dropout(0.2)(x)

    # x = keras.layers.Dense(32, activation="relu")(x)
    # x = keras.layers.Dropout(0.2)(x)

    outputs = keras.layers.Dense(14, activation="linear")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model

# ===========================================================================================
# Building the LSTM model
# ===========================================================================================
model = build_lstm(
    look_back= LOOK_BACK, 
    n_features= 22, 
    dropout= 0.0, 
    recurrent_dropout= 0.0
)
# ===========================================================================================
# Model training with EarlyStopping
# ===========================================================================================
callback = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)

# history = model.fit(
#     X,
#     Y,
#     epochs= EPOCHS,
#     batch_size= BATCH,
#     validation_split= VALID_SPLIT,
#     callbacks=[callback],
# )
