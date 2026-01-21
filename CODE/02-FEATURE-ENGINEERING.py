import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

plots_path = 'PLOTS/02_FEATURE ENGINEERING'
os.makedirs(plots_path, exist_ok=True)
sns.set_theme(style="darkgrid")

data = pd.read_csv("DATA/log_data_Downsampled.csv", index_col='Timestamp', parse_dates=True)

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

R1to7 = ["R" + str(i) for i in range(1,8)]
R8to14 = ["R" + str(i) for i in range(8,15)]

data["R1to7_Mean"] = data[R1to7].mean(axis=1)
data["R8to14_Mean"] = data[R8to14].mean(axis=1)

# ===========================================================================================
# Creating extra contextual features using the Moving Average
# ===========================================================================================

def finding_MA_pair(short_windows, long_windows, target_feature):
    results = []
    # target = data[target_feature].shift(-1)
    target = data['CO_ppm']

    for short_w in short_windows:
        for long_w in long_windows:
            if short_w >= long_w:
                continue
            rolling_mean = data[target_feature].rolling(window=long_w).mean()
            rolling_std  = data[target_feature].rolling(window=long_w).std()

            current_val_or_short_mean = data[target_feature].rolling(window=short_w).mean()
            z_score = (current_val_or_short_mean - rolling_mean) / rolling_std 

            temp_df = pd.DataFrame({'Z_Score': z_score,'Target': target}).dropna()
            
            X = temp_df[['Z_Score']] 
            y = temp_df['Target']

            corr =  mutual_info_regression(X, y)[0]

            results.append({
                'Short_Window': short_w,
                'Long_Window': long_w,
                'mutual_info_regression': corr,
                'Real_Corr': corr
            })

    res_df = pd.DataFrame(results)

    best_params = res_df.sort_values(by='mutual_info_regression', ascending=False).iloc[0]

    best_short = int(best_params['Short_Window'])
    best_long = int(best_params['Long_Window'])
    print(f"\nBest rolling window pair for {target_feature}:")
    print(f"Short Window: {int(best_params['Short_Window'])} (x5 mins)")
    print(f"Long Window:  {int(best_params['Long_Window'])} (x5 mins)")
    print(f"mutual_info_regression:  {best_params['mutual_info_regression']:.4f}")
    return best_short,best_long

def add_zscore_as_feature(col_name, best_short, best_long, target_feature):
    long_rolling_mean = data[target_feature].rolling(window = best_long).mean()
    long_rolling_std  = data[target_feature].rolling(window = best_long).std()
    short_mean = data[target_feature].rolling(window=best_short).mean()

    z_score = (short_mean - long_rolling_mean) / (long_rolling_std + 1e-6) 
    z_score = z_score.fillna(0).clip(-5, 5)
    data[col_name] = z_score

def plot_zscore(col_name, file_name):
    plt.figure(figsize=(15,8))
    plt.plot(data[col_name], label=col_name, color="blue", linewidth=1.5,alpha=0.8)
    plt.title(f'{col_name} z_score')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.grid("x")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_path, f"{file_name}_z_score.png"), dpi=300)
    plt.close()

def create_short_medium_and_long_zscores_for(target_feature):
    short_windows = range(1,13)  
    
    # short z score
    feature_name = f"{target_feature}_short_zscore"
    long_windows = range(2,13)  
    best_short , best_long = finding_MA_pair(short_windows, long_windows,target_feature)
    add_zscore_as_feature(feature_name, best_short, best_long, target_feature)
    plot_zscore(feature_name, f"{target_feature}_short")

    # medium z score
    feature_name = f"{target_feature}_medium_zscore"
    long_windows = range(200,301)  
    best_short , best_long = finding_MA_pair(short_windows, long_windows,target_feature)
    add_zscore_as_feature(feature_name, best_short, best_long, target_feature)
    plot_zscore(feature_name, f"{target_feature}_medium")

    # long z score
    feature_name = f"{target_feature}_long_zscore"
    long_windows = range(400,801)  
    best_short , best_long = finding_MA_pair(short_windows, long_windows,target_feature)
    add_zscore_as_feature(feature_name, best_short, best_long, target_feature)
    plot_zscore(feature_name, f"{target_feature}_long")

create_short_medium_and_long_zscores_for("R1to7_Mean")
create_short_medium_and_long_zscores_for("R8to14_Mean")

# ===========================================================================================
# Add lag to mean and target features
# ===========================================================================================
cols_to_lag = ["Sensors_Mean","R1to7_Mean","R8to14_Mean","R1to7_Mean_short_zscore",
              "R1to7_Mean_medium_zscore","R1to7_Mean_long_zscore","R8to14_Mean_short_zscore",
              "R8to14_Mean_medium_zscore","R8to14_Mean_long_zscore"]

for col in cols_to_lag:
    data[col] = data[col].shift(1)
   
r_columns = ["R" + str(i) for i in range(1, 15)]
for col in r_columns:
    data["Prev_" + col] = data[col].shift(1)

data.dropna(inplace = True) 

# save to csv file for later
data.to_csv("DATA/log_data_Downsampled.csv", index=True)
