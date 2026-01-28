import keras
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error, r2_score
import datetime

plots_path = 'PLOTS/04_RNN'
os.makedirs(plots_path, exist_ok=True)
sns.set_theme(style="whitegrid")

data = pd.read_csv("DATA/log_data_Downsampled.csv", index_col='Timestamp', parse_dates=True)

# ===========================================================================================
# Data preparacion
# ===========================================================================================

# Split the dataset to : input_features , target_features
r_columns = ["Target_R" + str(i) for i in range(1, 15)]
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
LOOK_BACK = 30 #30
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

# Split data to 70% - 20% - 10% for train, val and test 
data_size = len(X)
train_end = int(data_size * 0.7)
val_end = int(data_size * 0.9)

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
# [ LOOK_BACK ] -------- | The window of time steps that will go as input to the model to predict Y the target (which Y is 14 values at once!).
# [ EPOCHS ] ----------- | How many times will the model see COMPLETELY all the dataset.
# [ EPOCHS_PATIENCE ] -- | How many epochs will run without getting a better val_loss.
# [ LR_PATIENCE ] ------ | Reduces the lerning rate after a number of EPOCHS without a change.
# [ BATCH ] ------------ | How many "rows" of data the model will have to see before adjusting the weights.
# [ N_FEATURES ] ------- | The number of input features
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 

LOOK_BACK       # This has been set before in the Data preparacion, so that the data shape is the same as the models input shape
EPOCHS          = 1500 
EPOCHS_PATIENCE = 50    # 50 
LR_PATIENCE     = 25    # 25 
BATCH           = 8    # 8 
LONG_LSTM       = 256   # 256 
N_FEATURES      = X_train.shape[2]

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.55,   # 0.55 
    patience=LR_PATIENCE,       
    min_lr=0.00001,  
    verbose=1
)
# ===========================================================================================
# Helper to build an LSTM model
# ===========================================================================================
def build_lstm(
    look_back: int,
    n_features: int,
    ) -> keras.Model:

    inputs = keras.Input(shape=(look_back,n_features))

    x = keras.layers.LSTM(
        LONG_LSTM,
        return_sequences=False, # 
    )(inputs)

    x = keras.layers.Dropout(0.1)(x) # 0.1

    # x = keras.layers.LSTM(
    #     64, #64
    #     return_sequences=False,
    # )(x)

    # only 1 lstm 256 + 4 hidden layers 128-64-32-16

    x = keras.layers.Dense(128, activation="swish", kernel_initializer="he_normal")(x)  
    x = keras.layers.Dense(64, activation="swish", kernel_initializer="he_normal")(x)   
    x = keras.layers.Dense(32, activation="swish", kernel_initializer="he_normal")(x) 
    x = keras.layers.Dense(16, activation="swish", kernel_initializer="he_normal")(x)
    

    outputs = keras.layers.Dense(14, activation="linear")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="nadam", loss="mse") # optimizer="nadam"
    return model

# ===========================================================================================
# Building the LSTM model
# ===========================================================================================
model = build_lstm(
    look_back= LOOK_BACK, 
    n_features= N_FEATURES, 
)
# ===========================================================================================
# Model training with EarlyStopping
# ===========================================================================================
callback = EarlyStopping(
    monitor='val_loss', 
    patience=EPOCHS_PATIENCE, 
    restore_best_weights=True,
    )

start = datetime.datetime.now()

history = model.fit(
    X_train,
    Y_train,
    epochs= EPOCHS,
    batch_size= BATCH,
    validation_data=(X_val, Y_val),
    callbacks=[callback,reduce_lr],
    shuffle=True, # True 
)
train_time = datetime.datetime.now() - start 
train_time = str(train_time).split('.')[0]
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
# mape = mean_absolute_percentage_error(Y_test_real, predictions_real) * 100
# rmse = np.sqrt(mean_squared_error(Y_test_real, predictions_real))
# mae  = mean_absolute_error(Y_test_real, predictions_real)
# r2   = r2_score(Y_test_real, predictions_real)

# print("\n" + "="*40)
# print(" LSTM FINAL RESULTS")
# print("="*40)
# print(f"MAPE: {mape:.4f} %")  
# print(f"RMSE: {rmse:.4f}")
# print(f"MAE:  {mae:.4f}")
# print(f"R2:   {r2:.4f}")
# print("="*40)
# ===========================================================================================
# Evaluation per Sensor
# ===========================================================================================
metrics_data = []

import time
start_inf = time.time()
_ = model.predict(X_test, verbose=0)
end_inf = time.time()
inference_time = end_inf - start_inf

print(f"Total Training Time: {train_time} seconds")
print(f"Total Inference Time: {inference_time:.4f} seconds")

for i in range(14):
    sensor_name = f"Sensor_R{i+1}"
   
    y_true_s = Y_test_real[:, i]
    y_pred_s = predictions_real[:, i]

    mape = round((mean_absolute_percentage_error(y_true_s, y_pred_s) * 100), 4)
    rmse = round(np.sqrt(mean_squared_error(y_true_s, y_pred_s)), 4)
    mae  = round(mean_absolute_error(y_true_s, y_pred_s), 4)
    r2   = round(r2_score(y_true_s, y_pred_s), 4)

    errors = y_true_s - y_pred_s
    mean_error = round( np.mean(errors), 4)
    std_error  = round( np.std(errors), 4)
    
    metrics_data.append({
        "Sensor_ID": sensor_name,
        "MAPE (%)": mape,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "Mean_Error": mean_error, 
        "Std_Error": std_error    
    })

metrics_df = pd.DataFrame(metrics_data)
metrics_df.set_index("Sensor_ID", inplace=True)

print("\n" + "="*60)
print(" LSTM PER SENSOR RESULTS")
print("="*60)
print(metrics_df)
print("="*60)

metrics_df.to_csv(
    os.path.join(plots_path, "LSTM_Metrics_Per_Sensor.csv"), 
    sep=';',     
    decimal=','
)

# ===========================================================================================
# Plot predictions vs real
# ===========================================================================================
test_start_index = val_end + LOOK_BACK
test_timestamps = data.index[test_start_index : test_start_index + len(Y_test)]

results_list = []

for i in range(14): 
    sensor_id = i + 1 

    temp_df = pd.DataFrame({
        'Timestamp': test_timestamps,
        'Sensor_ID': sensor_id,
        'R_real': Y_test_real[:, i],          
        'prediction_real': predictions_real[:, i]
    })
    
    results_list.append(temp_df)

final_predictions_df = pd.concat(results_list, ignore_index=True)
print(final_predictions_df.head())


data = final_predictions_df.sort_values(by=['Sensor_ID', 'Timestamp'])

fig , axes = plt.subplots(nrows=5, ncols=3, figsize=(20, 26), sharex=True)
axes = axes.flatten()
sensor_ids = sorted(data['Sensor_ID'].unique())
for i, sensor_id in enumerate(sensor_ids):
    ax = axes[i]
    sensor_data = data[data['Sensor_ID'] == sensor_id]
    
    l1, = ax.plot(sensor_data['Timestamp'], sensor_data['R_real'], label='Actual', color="black", alpha=0.7, linewidth=1)
    l2, = ax.plot(sensor_data['Timestamp'], sensor_data['prediction_real'],label='Predicted', color="red", alpha=0.7, linewidth=1, linestyle='--')
    
    ax.set_title(f"Sensor R{sensor_id}", fontsize=11, fontweight='bold', loc='left', color='black')
    
    if i == 0:
        handles = [l1, l2]
        labels  = [l1.get_label(), l2.get_label()]
if len(sensor_ids) < len(axes):
    fig.delaxes(axes[-1])
plt.tight_layout(rect=[0, 0.02, 1, 0.93])

fig.legend(handles, labels, 
            loc='lower center',           
            bbox_to_anchor=(0.5, 0.93),   
            ncol=2, 
            frameon=True, 
            fontsize=13,
            facecolor='white',
            edgecolor='lightgray',
            borderpad=0.6)
    
plt.suptitle(f"Model Accuracy Analysis: RNN/LSTM", fontsize=25, fontweight='bold', y=0.98, color='#2C3E50')

save_path = os.path.join(plots_path, f"RNN_LSTM_Performance_per_Sensor.png")
plt.savefig(save_path, dpi=300)