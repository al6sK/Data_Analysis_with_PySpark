import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plots_path = 'PLOTS/02_FEATURE ENGINEERING'
os.makedirs(plots_path, exist_ok=True)
sns.set_theme(style="darkgrid")

data = pd.read_csv("DATA/data_Downsampled.csv", index_col='Timestamp', parse_dates=True)

# ===========================================================================================
# FEATURE ENGINEERING
# ===========================================================================================
print(data.head())
print(data.shape)

# ===========================================================================================
# Creating 3 new contextual features.
# ===========================================================================================
# The first contextual feature is equal to the Temperature difference between consecutive time points.
data["Temperature_diff"] = data["Temperature"].shift(1) - data["Temperature"]
# data.dropna(inplace = True)

# The second one is a binary option 0/1 characterizing the Heater_voltage to low or high.
mid_point = (data['Heater_voltage'].min() + data['Heater_voltage'].max())/2
data["Heater_voltage_state"] = [1 if v > mid_point else 0 for v in data['Heater_voltage']]

print(f"{(data['Heater_voltage_state'] == 1).sum()} /{len(data)}") 

# The third one is equal to the mean of the 14 sensors
sensors = ["R" + str(i) for i in range(1,15)]
data["Sensors_Mean"] = data[sensors].mean(axis=1)

print(data[["Temperature_diff","Heater_voltage_state","Sensors_Mean"]].head(15))

# save to csv file for later
data.to_csv("DATA/data_Downsampled.csv", index=True)

# ===========================================================================================
# plot CO_ppm_vs_Sensors_Mean
# ===========================================================================================

# normalized values to 0-1
columns = ["CO_ppm","Sensors_Mean"]
norm_data = data[columns].copy()

for feature in columns:
    norm_data[feature] = (norm_data[feature] - norm_data[feature].min()) / (norm_data[feature].max() - norm_data[feature].min())

colors = sns.color_palette("bright", 2)

data = norm_data[:100]
plt.figure(figsize=(15,8))
plt.plot(data['CO_ppm'], label='CO_ppm', color=colors[0], linewidth=1.5,alpha=0.8)
plt.plot(data['Sensors_Mean'], label='Sensors_Mean', color=colors[1], linewidth=1.5,alpha=0.8)
plt.title('CO_ppm vs Sensors_Mean')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.grid("x")
plt.tight_layout()
plt.savefig(os.path.join(plots_path, 'CO_ppm_vs_Sensors_Mean5.png'), dpi=300)
plt.close()

