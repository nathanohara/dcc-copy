#import all of the packages
import neurokit2 as nk
import pandas as pd
import numpy as np
from random import sample
import math
from sklearn import *
import numpy as np
import pandas as pd
import os
from functools import reduce
from tqdm import tqdm
import gc
from scipy.integrate import simps
np.random.seed(440)
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from collections import defaultdict
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import StackingClassifier
from sklearn.model_selection import cross_val_predict


subject_names = next(os.walk("/hpc/group/sta440-f20/WESAD/WESAD"))[1]
all_data = pd.DataFrame()

for subjname in tqdm(subject_names):
    subject_pickle = pd.read_pickle(f"/hpc/group/sta440-f20/WESAD/WESAD/{subjname}/{subjname}.pkl")
    labels = subject_pickle["label"]
    chest = subject_pickle["signal"]["chest"]
    wrist = subject_pickle["signal"]["wrist"]
 
    # Overall reference to the chest data
    max_obs = chest["ECG"].shape[0]
    t_array = np.arange(0, max_obs)*(1/700)
 
    # Making time masks for all other metrics from the wrist sensor
    bvp_obs = wrist["BVP"].shape[0]
    bvp_mask = np.arange(0,bvp_obs)*(1/64)
 
    acc_obs = wrist["ACC"].shape[0]
    acc_mask = np.arange(0,acc_obs)*(1/32)
 
    eda_temp_obs = wrist["EDA"].shape[0]
    eda_temp_mask = np.arange(0,eda_temp_obs)*(1/4)
 
    # Dataframe of all the data from the chest censor at 700hz
    chest_df = pd.DataFrame({"ACC_x":chest["ACC"][:,0].reshape(-1), "ACC_y":chest["ACC"][:,1].reshape(-1), "ACC_z":chest["ACC"][:,2].reshape(-1), "ECG":chest["ECG"].reshape(-1),
                         "EMG":chest["EMG"].reshape(-1), "EDA":chest["EDA"].reshape(-1), "Temp":chest["Temp"].reshape(-1), "Resp":chest["Resp"].reshape(-1)})
    chest_df["time"] = t_array
    chest_df["label"] = labels
 
    wrist_acc_df = pd.DataFrame({"time":acc_mask, "wr_ACC_x":wrist["ACC"][:,0].reshape(-1), "wr_ACC_y":wrist["ACC"][:,1].reshape(-1),
                                "wr_ACC_z":wrist["ACC"][:,2].reshape(-1)})
 
    bvp_df = pd.DataFrame({"time":bvp_mask, "BVP":wrist["BVP"].reshape(-1)})
 
    eda_temp_df = pd.DataFrame({"time":eda_temp_mask, "EDA":wrist["EDA"].reshape(-1), "wr_Temp":wrist["TEMP"].reshape(-1)})
 
    data_frames = [chest_df, wrist_acc_df, bvp_df, eda_temp_df]
    df_merged = reduce(lambda  left,right: pd.merge(left,right,on=['time'],
                                                how='outer'), data_frames)
 
    df_merged["subject"] = subjname
    all_data = pd.concat([all_data, df_merged])
    del df_merged
    gc.collect()

all_data = all_data[all_data['label'] != 1.0]
all_data = all_data[all_data['label'] != 4.0]
all_data = all_data[all_data['label'] != 5.0]
all_data = all_data[all_data['label'] != 6.0]
all_data = all_data[all_data['label'] != 7.0]
all_data = all_data[all_data['label'].notna()]


#function that calculates the resp features, utilizing the neurokit2 package
#returns a dataframe object that has the resp rate, mean inhale duration, standard deviation of the inhale duration,
#mean exhale duration, standard deviation of the exhale duration, ie ration, and the resp stretch
def resp_features(rsp):
    #print("rsp", rsp.dropna().reset_index(drop=True))
    cleaned = nk.rsp_process(rsp.dropna().reset_index(drop = True))
    mean_rsp_rate = np.mean(cleaned[0]["RSP_Rate"])
    peak_idx = cleaned[1]["RSP_Peaks"]
    trough_idx = cleaned[1]["RSP_Troughs"]
    inhale_time = 0
    exhale_time = 0
    for i, (p_idx, t_idx) in enumerate(zip(peak_idx, trough_idx)):
        if i == 0 or i == len(peak_idx)-1:
            continue
        inhale_time = (p_idx - t_idx)*(1/700)
        exhale_time = (cleaned[1]["RSP_Troughs"][i+1]-p_idx)*(1/700)
    mean_inhale_duration = np.mean(inhale_time)
    std_inhale_duration = np.std(inhale_time)
    mean_exhale_duration = np.mean(exhale_time)
    std_exhale_duration = np.std(exhale_time)
    ie_ratio = np.sum(inhale_time)/np.sum(exhale_time)
    stretch = np.max(rsp) - np.min(rsp)
    return pd.DataFrame({"resp_rate":mean_rsp_rate, "mean_inhale_duration":mean_inhale_duration,"std_inhale_duration":std_inhale_duration,
                        "mean_exhale_duration":[mean_exhale_duration], "std_exhale_duration":[std_exhale_duration], "ie_ratio":[ie_ratio],
                        "resp_stretch":[stretch]})
                        
#function that calculates the emg features. it returns a dataframe containing the mean emg, standard deviation of
#the emg, the number of peaks in the emg, the tenth quantile of the emg, the nintieth quantile of the emg, and
#the range of the emg.
def process_emg(df):
    features = pd.DataFrame(columns=['emg_mean', 'emg_standard_deviation', 'emg_num_peaks', "emg_tenth_quantile", "emg_nintieth_quantile", "emg_range"])
    emg = df['EMG']
    peaks, _ = find_peaks(emg)
    num_peaks = len(peaks)
    mean = emg.mean()
    standard_deviation = emg.std()
    tenth_quantile = emg.quantile(.1)
    nintieth_quantile = emg.quantile(.9)
    range_e = emg.max() - emg.min()
    features.loc[len(features)] = [mean, standard_deviation, num_peaks, tenth_quantile, nintieth_quantile, range_e]
    return features

#this function processes the temperature and returns the features concerning temperature.
#it returns a dataframe containing the mean temperature, the standard deviation of the temperature, the tenth quantile,
#the nintieth quantile, and the range of the temperature values.
def process_temp(df, wrist):
    features = pd.DataFrame(columns=['temp_mean', 'temp_standard_deviation', 'temp_tenth_quantile', 'temp_nintieth_quantile', 'temp_range'])
    if wrist:
        temp = df['wr_Temp']
    else:
        temp = df['Temp']
    mean = temp.mean()
    standard_deviation = temp.std()
    tenth_quantile = temp.quantile(.1)
    nintieth_quantile = temp.quantile(.9)
    range_t = temp.max() - temp.min()
    features.loc[len(features)] = [mean, standard_deviation, tenth_quantile, nintieth_quantile, range_t]
    return features
    
#this function processes the bvp and returns the features concerning bvp.
#it returns a dataframe containing the mean bvp, standard deviation of bvp, the tenth quantile of bvp, the nintieth
#quantile of bvp, and the range of values of bvp.
def process_bvp(df):
    features = pd.DataFrame(columns=['bvp_mean', 'bvp_standard_deviation', 'bvp_tenth_quantile', 'bvp_nintieth_quantile', 'bvp_range'])
    bvp = df['BVP']
    bvp = bvp.dropna()
    mean = bvp.mean()
    standard_deviation = bvp.std()
    tenth_quantile = bvp.quantile(.1)
    nintieth_quantile = bvp.quantile(.9)
    range_b = bvp.max() - bvp.min()
    features.loc[len(features)] = [mean, standard_deviation, tenth_quantile, nintieth_quantile, range_b]
    return features

#function returns the processes ecg. it utilizes the neurokit2 package to process the results to calculate
#features associated with the heart rate variability. returns a dataframe object.
def process_ecg(df):
    df = df['ECG']
    df = df.dropna()
    processed_data, info = nk.bio_process(ecg=df, sampling_rate=700)
    results = nk.bio_analyze(processed_data, sampling_rate=700)
    return results

#function that returns the features associated with eda. this also utilizes the neurokit2 package to process the
#raw eda signals. The sample rate input depends on whether the eda signals are from the wrist or the chest device.
def get_eda_features(eda, sample_rate=700, windex = (0, -1)):
    if sample_rate == 4:
        sample_rate = 8
        eda = eda.dropna().reset_index(drop = True)
        temp_eda = np.zeros((len(eda)*2))
        for i in range(len(eda)-1):
            temp_eda[i*2] = eda[i]
            temp_eda[i*2+1] = (eda[i] + eda[i+1])/2
        eda = temp_eda
    else:
        eda = eda.dropna().reset_index(drop = True)
    eda = eda[windex[0]:windex[1]]
    t = (np.arange(0,len(eda)+1)*(1/sample_rate))[windex[0]:windex[1]]
    signals, info = nk.eda_process(eda, sampling_rate=sample_rate)
    cleaned = signals["EDA_Clean"]
    eda_features = nk.eda_phasic(nk.standardize(eda), sampling_rate=sample_rate)
    scr = eda_features["EDA_Phasic"]
    scl = eda_features["EDA_Tonic"]
    scl_corrcoeff = np.corrcoef(scl, t)[0,1]
    num_scr_segments = len(np.nan_to_num(info["SCR_Onsets"]))
    sum_startle_magnitudes = sum(np.nan_to_num(info["SCR_Amplitude"]))
    sum_response_time = sum(np.nan_to_num(info["SCR_RiseTime"]))
    peak_integrals = []
    for onset_idx, peak_idx in zip(info["SCR_Onsets"], info["SCR_Peaks"]):
        if np.isnan(onset_idx) or np.isnan(peak_idx):
            continue
        onset_idx, peak_idx = int(onset_idx), int(peak_idx)
        cur_y = cleaned[onset_idx:peak_idx+1]
        cur_integral = simps(cur_y, dx = 1/sample_rate)
        peak_integrals.append(cur_integral)
    sum_response_areas = np.sum(peak_integrals)
    mean_eda = np.mean(eda)
    std_eda = np.std(eda)
    min_eda = np.min(eda)
    max_eda = np.max(eda)
    slope_eda = (eda[len(eda)-1] - eda[0])/len(eda)
    range_eda = max_eda - min_eda
    mean_scl = np.mean(scl)
    std_scl = np.std(scl)
    std_scr = np.std(scr)
    return pd.DataFrame({"mean_eda":[mean_eda], "std_eda":[std_eda], "min_eda":[min_eda], "max_eda":[max_eda], "slope_eda":[slope_eda],
                       "range_eda":[range_eda], "mean_scl":[mean_scl], "std_scl":[std_scl], "std_scr":[std_scr], "scl_corr":[scl_corrcoeff],
                       "num_scr_seg":[num_scr_segments], "sum_startle_mag":[sum_startle_magnitudes],"sum_response_time":[sum_response_time],
                       "sum_response_areas":[sum_response_areas]})

#function that creates the overlapping windows per each subject, and for amusement and stress separately.
#the windows are spaced out approximately per 60 seconds. there are some windows that end up smaller than the others,
#just by the fact that there are not even increments of 60 seconds for both stress and amusement conditions for each of
#the subjects.
def create_windows(df, initial_time):
    indices = []
    samples = []
    num_of_separation = 70000
    samples = []
    size_of_window = 70000
    counter = 0
    while(counter + 70000 <= len(df)):
        s = sub_stress.loc[counter:counter+70000]
        samples.append(s)
    return samples

'''
def create_windows(df, initial_time):
    indices = []
    samples = []
    num_of_separation = 70000
    length = math.floor(len(df) / num_of_separation) #70,000 == 100 second windows. 10500 = 15 second windows
    counter = 0
    while(length != 0):
        counter = counter + num_of_separation
        indices.append(counter)
        length = length - 1
        
    samples.append(df.loc[0:indices[0]]) #from the first row of the dataframe to the first 60 seconds
    diff = int(indices[1]-indices[0]) #= to 10500 = length of the window
    for x in range(len(indices)-1):
        s = df.loc[indices[x]:indices[x+1]]
        samples.append(s)
        temp = int(indices[x+1] - indices[x])
        #temp_2 = int(temp/2)
        temp_2 = 700
        if temp_2 + indices[x] <= len(df):
            ss = df.loc[temp_2+indices[x] - num_of_separation:temp_2+indices[x]]
            samples.append(ss)
        else:
            ss = df.loc[temp_2+indices[x]:len(df)-1]
            samples.append(ss)
    final_s = df.loc[indices[len(indices)-1]:len(df)] #final
    if not final_s.empty:
        samples.append(final_s)
    return samples
  '''
    
#read in the csv file containing only the amusement and stress conditions. this calls the function to create the
#windows.
df = all_data
labels = ['S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11', 'S13', 'S14', 'S15', 'S16', 'S17']
windows = []
for i in labels:
    sub_stress = df[(df['subject'] == i) & (df['label'] == 2.0)]
    sub_stress = sub_stress.reset_index()
    
    sub_amusement = df[(df['subject'] == i) & (df['label'] == 3.0)]
    sub_amusement = sub_amusement.reset_index()
    
    sub_stress_temp = sub_stress['time'].head(1)
    stress_initial_time = sub_stress_temp.values[0]
    
    sub_amusement_temp = sub_amusement['time'].head(1)
    amusement_initial_time = sub_amusement_temp.values[0]
    
    temp = create_windows(sub_stress, stress_initial_time)
    for t in temp:
        windows.append(t)
    temp2 = create_windows(sub_amusement, amusement_initial_time)
    for t in temp2:
        windows.append(t)

#create the dataframe that will be the final output ocntaining all of the features for the particular window.
y = pd.DataFrame(columns = ['ECG_Rate_Mean', 'HRV_RMSSD', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_SDSD','HRV_CVNN', 'HRV_CVSD', 'HRV_MedianNN', 'HRV_MadNN', 'HRV_MCVNN', 'HRV_IQRNN', 'HRV_pNN50', 'HRV_pNN20', 'HRV_TINN', 'HRV_HTI', 'HRV_ULF',
'HRV_VLF', 'HRV_LF', 'HRV_HF', 'HRV_VHF', 'HRV_LFHF', 'HRV_LFn',
'HRV_HFn', 'HRV_LnHF', 'HRV_SD1', 'HRV_SD2', 'HRV_SD1SD2', 'HRV_S',
'HRV_CSI', 'HRV_CVI', 'HRV_CSI_Modified', 'HRV_PIP', 'HRV_IALS',
'HRV_PSS', 'HRV_PAS', 'HRV_GI', 'HRV_SI', 'HRV_AI', 'HRV_PI', 'HRV_C1d',
'HRV_C1a', 'HRV_SD1d', 'HRV_SD1a', 'HRV_C2d', 'HRV_C2a', 'HRV_SD2d',
'HRV_SD2a', 'HRV_Cd', 'HRV_Ca', 'HRV_SDNNd', 'HRV_SDNNa', 'HRV_ApEn',
'HRV_SampEn','eda_mean_chest',
'eda_std_chest',
'eda_min_chest',
'eda_max_chest',
'eda_slope_chest',
'eda_range_chest',
'eda_mean_scl_chest',
'eda_std_scl_chest',
'eda_std_scr_chest',
'eda_scl_corr_chest',
'eda_num_scr_seg_chest',
'eda_sum_startle_mag_chest',
'eda_sum_response_time_chest',
'eda_sum_response_areas_chest',
'eda_mean_wr',
'eda_std_wr',
'eda_min_wr',
'eda_max_wr',
'eda_slope_wr',
'eda_range_wr',
'eda_mean_scl_wr',
'eda_std_scl_wr',
'eda_std_scr_wr',
'eda_scl_corr_wr',
'eda_num_scr_seg_wr',
'eda_sum_startle_mag_wr',
'eda_sum_response_time_wr',
'eda_sum_response_areas_wr',
'resp_rate',
  'mean_inhale_duration',
  'std_inhale_duration',
  'mean_exhale_duration',
  'std_exhale_duration',
  'ie_ratio',
  'resp_stretch',
  'temp_wr_mean',
  'temp_wr_standard_deviation',
  'temp_wr_tenth_quantile',
  'temp_wr_nintieth_quantile',
  'temp_wr_range',
  'temp_chest_mean',
  'temp_chest_standard_deviation',
  'temp_chest_tenth_quantile',
  'temp_chest_nintieth_quantile',
  'temp_chest_range',
  'acc_x_mean',
  'acc_y_mean',
  'acc_z_mean',
  'acc_square_root',
  'bvp_mean',
  'bvp_standard_deviation',
  'bvp_tenth_quantile',
  'bvp_nintieth_quantile',
  'bvp_range',
  'subject', 'label'])



#for each of the windows, calculate the features for each of the raw signals.
for i in range(len(windows)):
    
    subject = windows[i]['subject'].head(1).values[0]
    label = windows[i]['label'].head(1).values[0]
    ecg_data = process_ecg(windows[i])
    #ecg_data['subject'] = windows[i]['subject'].head(1).values[0]
    #ecg_data['window_trial'] = i
    #ecg_data['label'] = windows[i]['label'].head(1).values[0]
    
    ECG_Rate_Mean = ecg_data.ECG_Rate_Mean.values[0]
    HRV_RMSSD = ecg_data.HRV_RMSSD.values[0]
    HRV_MeanNN= ecg_data.HRV_MeanNN.values[0]
    HRV_SDNN= ecg_data.HRV_SDNN.values[0]
    HRV_SDNN= ecg_data.HRV_SDNN.values[0]
    HRV_CVNN= ecg_data.HRV_CVNN.values[0]
    HRV_CVSD= ecg_data.HRV_CVSD.values[0]
    HRV_MedianNN= ecg_data.HRV_MedianNN.values[0]
    HRV_MadNN= ecg_data.HRV_MadNN.values[0]
    HRV_MCVNN= ecg_data.HRV_MCVNN.values[0]
    HRV_IQRNN= ecg_data.HRV_IQRNN.values[0]
    HRV_pNN50= ecg_data.HRV_pNN50.values[0]
    HRV_pNN20= ecg_data.HRV_pNN20.values[0]
    HRV_TINN= ecg_data.HRV_TINN.values[0]
    HRV_HTI= ecg_data.HRV_HTI.values[0]
    HRV_ULF= ecg_data.HRV_ULF.values[0]
    HRV_VLF= ecg_data.HRV_VLF.values[0]
    HRV_LF= ecg_data.HRV_LF.values[0]
    HRV_HF= ecg_data.HRV_HF.values[0]
    HRV_VHF= ecg_data.HRV_VHF.values[0]
    HRV_LFHF= ecg_data.HRV_LFHF.values[0]
    HRV_LFn= ecg_data.HRV_LFn.values[0]
    HRV_HFn= ecg_data.HRV_HFn.values[0]
    HRV_LnHF= ecg_data.HRV_LnHF.values[0]
    HRV_SD1= ecg_data.HRV_SD1.values[0]
    HRV_SD2= ecg_data.HRV_SD2.values[0]
    HRV_SD1SD2= ecg_data.HRV_SD1SD2.values[0]
    HRV_S= ecg_data.HRV_S.values[0]
    HRV_CSI= ecg_data.HRV_CSI.values[0]
    HRV_CVI= ecg_data.HRV_CVI.values[0]
    HRV_CSI_Modified= ecg_data.HRV_CSI_Modified.values[0]
    HRV_PIP= ecg_data.HRV_PIP.values[0]
    HRV_IALS= ecg_data.HRV_IALS.values[0]
    HRV_PSS= ecg_data.HRV_PSS.values[0]
    HRV_PAS= ecg_data.HRV_PAS.values[0]
    HRV_GI= ecg_data.HRV_GI.values[0]
    HRV_SI= ecg_data.HRV_SI.values[0]
    HRV_AI= ecg_data.HRV_AI.values[0]
    HRV_PI= ecg_data.HRV_PI.values[0]
    HRV_C1d= ecg_data.HRV_C1d.values[0]
    HRV_C1a= ecg_data.HRV_C1a.values[0]
    HRV_SD1d= ecg_data.HRV_SD1d.values[0]
    HRV_SD1a= ecg_data.HRV_SD1a.values[0]
    HRV_C2d= ecg_data.HRV_C2d.values[0]
    HRV_C2a= ecg_data.HRV_C2a.values[0]
    HRV_SD2d= ecg_data.HRV_SD2d.values[0]
    HRV_SD2a= ecg_data.HRV_SD2a.values[0]
    HRV_Cd= ecg_data.HRV_Cd.values[0]
    HRV_Ca= ecg_data.HRV_Ca.values[0]
    HRV_SDNNd= ecg_data.HRV_SDNNd.values[0]
    HRV_SDNNa= ecg_data.HRV_SDNNa.values[0]
    HRV_ApEn= ecg_data.HRV_ApEn.values[0]
    HRV_SampEn= ecg_data.HRV_SampEn.values[0]
    HRV_SDSD = ecg_data.HRV_SDSD.values[0]
    
    #EDA Features ~~
    
    eda_data = get_eda_features(windows[i]['EDA_x'])
    #eda_data['subject'] = windows[i]['subject'].head(1).values[0]
    #eda_data["window_trial"] = i
    #eda_data['label'] = windows[i]['label'].head(1).values[0]
    #eda_data['wrist_or_chest'] = 'chest'
    
    eda_mean_chest = eda_data.mean_eda.values[0]
    eda_std_chest = eda_data.std_eda.values[0]
    eda_min_chest = eda_data.min_eda.values[0]
    eda_max_chest = eda_data.max_eda.values[0]
    eda_slope_chest = eda_data.slope_eda.values[0]
    eda_range_chest = eda_data.range_eda.values[0]
    eda_mean_scl_chest = eda_data.mean_scl.values[0]
    eda_std_scl_chest = eda_data.std_scl.values[0]
    eda_std_scr_chest = eda_data.std_scr.values[0]
    eda_scl_corr_chest = eda_data.scl_corr.values[0]
    eda_num_scr_seg_chest = eda_data.num_scr_seg.values[0]
    eda_sum_startle_mag_chest = eda_data.sum_startle_mag.values[0]
    eda_sum_response_time_chest = eda_data.sum_response_time.values[0]
    eda_sum_response_areas_chest = eda_data.sum_response_time.values[0]
    
    
    #y.append(eda_data)
    
    eda_data_wr = get_eda_features(windows[i]['EDA_y'], sample_rate=4)
    #eda_data_wr['subject'] = windows[i]['subject'].head(1).values[0]
    #eda_data_wr["window_trial"] = i
    #eda_data_wr['label'] = windows[i]['label'].head(1).values[0]
    #eda_data_wr['wrist_or_chest'] = 'wrist'
    #y.append(eda_data_wr)
    eda_mean_wr = eda_data_wr.mean_eda.values[0]
    eda_std_wr = eda_data_wr.std_eda.values[0]
    eda_min_wr = eda_data_wr.min_eda.values[0]
    eda_max_wr = eda_data_wr.max_eda.values[0]
    eda_slope_wr = eda_data_wr.slope_eda.values[0]
    eda_range_wr = eda_data_wr.range_eda.values[0]
    eda_mean_scl_wr = eda_data_wr.mean_scl.values[0]
    eda_std_scl_wr = eda_data_wr.std_scl.values[0]
    eda_std_scr_wr = eda_data_wr.std_scr.values[0]
    eda_scl_corr_wr = eda_data_wr.scl_corr.values[0]
    eda_num_scr_seg_wr = eda_data_wr.num_scr_seg.values[0]
    eda_sum_startle_mag_wr = eda_data_wr.sum_startle_mag.values[0]
    eda_sum_response_time_wr = eda_data_wr.sum_response_time.values[0]
    eda_sum_response_areas_wr = eda_data_wr.sum_response_time.values[0]
    
    #BVP Features ~~
    bvp_data = process_bvp(windows[i])
    bvp_mean = bvp_data.bvp_mean.values[0]
    bvp_standard_deviation = bvp_data.bvp_standard_deviation.values[0]
    bvp_tenth_quantile = bvp_data.bvp_tenth_quantile.values[0]
    bvp_nintieth_quantile = bvp_data.bvp_nintieth_quantile.values[0]
    bvp_range = bvp_data.bvp_range.values[0]
    
    
    # Resp Features ~~
    #print("there is an error!:", windows[i]['Resp'])
    resp_data = resp_features(windows[i]['Resp'])
    resp_data['subject'] = windows[i]['subject'].head(1).values[0]
    resp_data['window_trial'] = i
    
    resp_rate = resp_data.resp_rate.values[0]
    mean_inhale_duration = resp_data.mean_inhale_duration.values[0]
    std_inhale_duration = resp_data.std_inhale_duration.values[0]
    mean_exhale_duration = resp_data.mean_exhale_duration.values[0]
    std_exhale_duration = resp_data.std_exhale_duration.values[0]
    ie_ratio = resp_data.ie_ratio.values[0]
    resp_stretch = resp_data.resp_stretch.values[0]
    
    #y.append(resp_data)
    
    #Temp Features ~
    temp_features_wr = process_temp(windows[i], True)
    #temp_features_wr['subject'] = windows[i]['subject'].head(1).values[0]
    #temp_features_wr['window_trial'] = i
    #temp_features_wr['wrist_or_chest'] = 'wrist'
    
    temp_wr_mean = temp_features_wr.temp_mean.values[0]
    temp_wr_standard_deviation = temp_features_wr.temp_standard_deviation.values[0]
    temp_wr_tenth_quantile = temp_features_wr.temp_tenth_quantile.values[0]
    temp_wr_nintieth_quantile = temp_features_wr.temp_nintieth_quantile.values[0]
    temp_wr_range = temp_features_wr.temp_range.values[0]
    
    temp_features = process_temp(windows[i], False)
    #temp_features['subject'] = windows[i]['subject'].head(1).values[0]
    #temp_features['window_trial'] = i
    #temp_features['wrist_or_chest'] = 'chest'
    #y.append(temp_features)
    
    temp_chest_mean = temp_features.temp_mean.values[0]
    temp_chest_standard_deviation = temp_features.temp_standard_deviation.values[0]
    temp_chest_tenth_quantile = temp_features.temp_tenth_quantile.values[0]
    temp_chest_nintieth_quantile = temp_features.temp_nintieth_quantile.values[0]
    temp_chest_range = temp_features.temp_range.values[0]
    
    import math
    acc_x_mean = windows[i].ACC_x.mean()
    acc_y_mean = windows[i].ACC_y.mean()
    acc_z_mean = windows[i].ACC_z.mean()
    temp = math.pow(acc_x_mean, 2) + math.pow(acc_y_mean, 2) + math.pow(acc_z_mean, 2)
    acc_square_root = math.sqrt(temp)
    
    
    
    y.loc[len(y)] = [ECG_Rate_Mean, HRV_RMSSD, HRV_MeanNN, HRV_SDNN, HRV_SDSD,HRV_CVNN, HRV_CVSD, HRV_MedianNN, HRV_MadNN, HRV_MCVNN, HRV_IQRNN, HRV_pNN50, HRV_pNN20, HRV_TINN, HRV_HTI, HRV_ULF,HRV_VLF, HRV_LF, HRV_HF, HRV_VHF, HRV_LFHF, HRV_LFn,HRV_HFn, HRV_LnHF, HRV_SD1, HRV_SD2, HRV_SD1SD2, HRV_S,HRV_CSI, HRV_CVI,HRV_CSI_Modified, HRV_PIP, HRV_IALS,HRV_PSS, HRV_PAS, HRV_GI, HRV_SI, HRV_AI, HRV_PI, HRV_C1d,HRV_C1a, HRV_SD1d, HRV_SD1a, HRV_C2d, HRV_C2a, HRV_SD2d,HRV_SD2a, HRV_Cd, HRV_Ca, HRV_SDNNd, HRV_SDNNa, HRV_ApEn,HRV_SampEn,eda_mean_chest,eda_std_chest,eda_min_chest,eda_max_chest,eda_slope_chest,eda_range_chest,eda_mean_scl_chest, eda_std_scl_chest,eda_std_scr_chest,eda_scl_corr_chest,eda_num_scr_seg_chest,eda_sum_startle_mag_chest,eda_sum_response_time_chest,eda_sum_response_areas_chest,eda_mean_wr,eda_std_wr,eda_min_wr, eda_max_wr,eda_slope_wr,eda_range_wr,eda_mean_scl_wr,eda_std_scl_wr,eda_std_scr_wr,eda_scl_corr_wr,eda_num_scr_seg_wr,eda_sum_startle_mag_wr,eda_sum_response_time_wr,eda_sum_response_areas_wr, resp_rate,mean_inhale_duration,std_inhale_duration,mean_exhale_duration,std_exhale_duration,ie_ratio,resp_stretch,temp_wr_mean,temp_wr_standard_deviation,temp_wr_tenth_quantile,temp_wr_nintieth_quantile, temp_wr_range,temp_chest_mean,temp_chest_standard_deviation,temp_chest_tenth_quantile,temp_chest_nintieth_quantile,temp_chest_range, acc_x_mean, acc_y_mean, acc_z_mean, acc_square_root, bvp_mean, bvp_standard_deviation, bvp_tenth_quantile, bvp_nintieth_quantile, bvp_range, subject, label]
y.to_csv("final.csv")  

final = y

# Removing columns with NA values
na_cols = []
for col in final.columns:
    numnas = sum(final[col].isna())
    print(col, numnas)
    if numnas > 5:
        na_cols.append(col)
na_cols.extend(["Unnamed: 0"])
final = final.drop(columns = na_cols)
print("Dropped", len(na_cols), "columns")

# Create dummies for subject
final["subject"] = final["subject"].astype("category")
final = pd.get_dummies(final, drop_first = True)

# Convert label to 1 for stress, 0 for amusement
final["label"] = (final["label"] == 2.0).astype(int)
final["label"].value_counts()

# Determine all subsets of the data on which we want to model
var_subsets = [
    ("all",[x for x in final.columns if x != "subject" and x != "label"]),
    ("ecg_only",[x for x in final.columns if x.startswith("HRV") or x.startswith("ECG")]),
    ("eda_chest", [x for x in final.columns if x.startswith("eda") and x.endswith("chest")]),
    ("eda_wrist", [x for x in final.columns if x.startswith("eda") and x.endswith("wr")]),
    ("resp_only", ['resp_rate', 'mean_inhale_duration', 'std_inhale_duration', 'mean_exhale_duration', 'std_exhale_duration', 'ie_ratio', 'resp_stretch']),
    ("acc_only", [x for x in final.columns if x.startswith("acc")]),
    ("bvp_only", [x for x in final.columns if x.startswith("bvp")]),
    ("wrist_only", [x for x in final.columns if (x.startswith('eda') and x.endswith('wr')) or x.startswith('acc') or x.startswith('temp') or x.startswith('bvp')]),
    ("chest_only", [x for x in final.columns if (x.startswith('eda') and x.endswith('chest')) or (x.startswith("HRV") or x.startswith("ECG"))] + ['resp_rate', 'mean_inhale_duration', 'std_inhale_duration', 'mean_exhale_duration', 'std_exhale_duration', 'ie_ratio', 'resp_stretch'])
]

# Defining Wrapper for PLS for use in Pipelines
class PLSRegressionWrapper(PLSRegression):

        def transform(self, X):
            return super().transform(X)

        def fit_transform(self, X, Y):
            return self.fit(X,Y).transform(X)


# Model params Pre-CV'd for your convenience -- apologies for the length (wasn't sure if I was allowed to include a JSON)

precomputed_params = {'all': {'logistic': {'C': 100,
   'class_weight': None,
   'dual': False,
   'fit_intercept': True,
   'intercept_scaling': 1,
   'l1_ratio': None,
   'max_iter': 10000,
   'multi_class': 'auto',
   'n_jobs': None,
   'penalty': 'l1',
   'random_state': None,
   'solver': 'liblinear',
   'tol': 0.0001,
   'verbose': 0,
   'warm_start': False},
  'xgboost': {'objective': 'binary:logistic',
   'base_score': 0.5,
   'booster': 'gbtree',
   'colsample_bylevel': 1,
   'colsample_bynode': 1,
   'colsample_bytree': 0.5,
   'gamma': 0.1,
   'gpu_id': -1,
   'importance_type': 'gain',
   'interaction_constraints': '',
   'learning_rate': 0.200000003,
   'max_delta_step': 0,
   'max_depth': 3,
   'min_child_weight': 1,
   'missing': np.nan,
   'monotone_constraints': '()',
   'n_estimators': 100,
   'n_jobs': 0,
   'num_parallel_tree': 1,
   'random_state': 0,
   'reg_alpha': 0,
   'reg_lambda': 1,
   'scale_pos_weight': 1,
   'subsample': 0.5,
   'tree_method': 'exact',
   'validate_parameters': 1,
   'verbosity': None,
   'alpha': 0,
   'eta': 0.2},
  'linear_svm': {'C': 1,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'linear',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'rbf': {'C': 10,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'rbf',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'polynomial': {'C': 10,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'poly',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'pls_rbf': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()), ('rbf_svm', SVC())],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'rbf_svm': SVC(),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'rbf_svm__C': 1.0,
   'rbf_svm__break_ties': False,
   'rbf_svm__cache_size': 200,
   'rbf_svm__class_weight': None,
   'rbf_svm__coef0': 0.0,
   'rbf_svm__decision_function_shape': 'ovr',
   'rbf_svm__degree': 3,
   'rbf_svm__gamma': 'scale',
   'rbf_svm__kernel': 'rbf',
   'rbf_svm__max_iter': -1,
   'rbf_svm__probability': False,
   'rbf_svm__random_state': None,
   'rbf_svm__shrinking': True,
   'rbf_svm__tol': 0.001,
   'rbf_svm__verbose': False},
  'pls_linear': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('linear_svm', SVC(kernel='linear'))],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'linear_svm': SVC(kernel='linear'),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'linear_svm__C': 1.0,
   'linear_svm__break_ties': False,
   'linear_svm__cache_size': 200,
   'linear_svm__class_weight': None,
   'linear_svm__coef0': 0.0,
   'linear_svm__decision_function_shape': 'ovr',
   'linear_svm__degree': 3,
   'linear_svm__gamma': 'scale',
   'linear_svm__kernel': 'linear',
   'linear_svm__max_iter': -1,
   'linear_svm__probability': False,
   'linear_svm__random_state': None,
   'linear_svm__shrinking': True,
   'linear_svm__tol': 0.001,
   'linear_svm__verbose': False},
  'pls_polynomial': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('poly_svm', SVC(kernel='poly'))],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'poly_svm': SVC(kernel='poly'),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'poly_svm__C': 1.0,
   'poly_svm__break_ties': False,
   'poly_svm__cache_size': 200,
   'poly_svm__class_weight': None,
   'poly_svm__coef0': 0.0,
   'poly_svm__decision_function_shape': 'ovr',
   'poly_svm__degree': 3,
   'poly_svm__gamma': 'scale',
   'poly_svm__kernel': 'poly',
   'poly_svm__max_iter': -1,
   'poly_svm__probability': False,
   'poly_svm__random_state': None,
   'poly_svm__shrinking': True,
   'poly_svm__tol': 0.001,
   'poly_svm__verbose': False},
  'pls_lr': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('logistic', LogisticRegression())],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'logistic': LogisticRegression(),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'logistic__C': 1.0,
   'logistic__class_weight': None,
   'logistic__dual': False,
   'logistic__fit_intercept': True,
   'logistic__intercept_scaling': 1,
   'logistic__l1_ratio': None,
   'logistic__max_iter': 100,
   'logistic__multi_class': 'auto',
   'logistic__n_jobs': None,
   'logistic__penalty': 'l2',
   'logistic__random_state': None,
   'logistic__solver': 'lbfgs',
   'logistic__tol': 0.0001,
   'logistic__verbose': 0,
   'logistic__warm_start': False}},
 'ecg_only': {'logistic': {'C': 1,
   'class_weight': None,
   'dual': False,
   'fit_intercept': True,
   'intercept_scaling': 1,
   'l1_ratio': None,
   'max_iter': 10000,
   'multi_class': 'auto',
   'n_jobs': None,
   'penalty': 'l1',
   'random_state': None,
   'solver': 'liblinear',
   'tol': 0.0001,
   'verbose': 0,
   'warm_start': False},
  'xgboost': {'objective': 'binary:logistic',
   'base_score': 0.5,
   'booster': 'gbtree',
   'colsample_bylevel': 1,
   'colsample_bynode': 1,
   'colsample_bytree': 1,
   'gamma': 0,
   'gpu_id': -1,
   'importance_type': 'gain',
   'interaction_constraints': '',
   'learning_rate': 0.00999999978,
   'max_delta_step': 0,
   'max_depth': 3,
   'min_child_weight': 1,
   'missing': np.nan,
   'monotone_constraints': '()',
   'n_estimators': 100,
   'n_jobs': 0,
   'num_parallel_tree': 1,
   'random_state': 0,
   'reg_alpha': 0,
   'reg_lambda': 1,
   'scale_pos_weight': 1,
   'subsample': 1,
   'tree_method': 'exact',
   'validate_parameters': 1,
   'verbosity': None,
   'alpha': 0,
   'eta': 0.01},
  'linear_svm': {'C': 1,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'linear',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'rbf': {'C': 100,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'rbf',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'polynomial': {'C': 10,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'poly',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'pls_rbf': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()), ('rbf_svm', SVC())],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'rbf_svm': SVC(),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'rbf_svm__C': 1.0,
   'rbf_svm__break_ties': False,
   'rbf_svm__cache_size': 200,
   'rbf_svm__class_weight': None,
   'rbf_svm__coef0': 0.0,
   'rbf_svm__decision_function_shape': 'ovr',
   'rbf_svm__degree': 3,
   'rbf_svm__gamma': 'scale',
   'rbf_svm__kernel': 'rbf',
   'rbf_svm__max_iter': -1,
   'rbf_svm__probability': False,
   'rbf_svm__random_state': None,
   'rbf_svm__shrinking': True,
   'rbf_svm__tol': 0.001,
   'rbf_svm__verbose': False},
  'pls_linear': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('linear_svm', SVC(kernel='linear'))],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'linear_svm': SVC(kernel='linear'),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'linear_svm__C': 1.0,
   'linear_svm__break_ties': False,
   'linear_svm__cache_size': 200,
   'linear_svm__class_weight': None,
   'linear_svm__coef0': 0.0,
   'linear_svm__decision_function_shape': 'ovr',
   'linear_svm__degree': 3,
   'linear_svm__gamma': 'scale',
   'linear_svm__kernel': 'linear',
   'linear_svm__max_iter': -1,
   'linear_svm__probability': False,
   'linear_svm__random_state': None,
   'linear_svm__shrinking': True,
   'linear_svm__tol': 0.001,
   'linear_svm__verbose': False},
  'pls_polynomial': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('poly_svm', SVC(kernel='poly'))],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'poly_svm': SVC(kernel='poly'),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'poly_svm__C': 1.0,
   'poly_svm__break_ties': False,
   'poly_svm__cache_size': 200,
   'poly_svm__class_weight': None,
   'poly_svm__coef0': 0.0,
   'poly_svm__decision_function_shape': 'ovr',
   'poly_svm__degree': 3,
   'poly_svm__gamma': 'scale',
   'poly_svm__kernel': 'poly',
   'poly_svm__max_iter': -1,
   'poly_svm__probability': False,
   'poly_svm__random_state': None,
   'poly_svm__shrinking': True,
   'poly_svm__tol': 0.001,
   'poly_svm__verbose': False},
  'pls_lr': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('logistic', LogisticRegression())],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'logistic': LogisticRegression(),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'logistic__C': 1.0,
   'logistic__class_weight': None,
   'logistic__dual': False,
   'logistic__fit_intercept': True,
   'logistic__intercept_scaling': 1,
   'logistic__l1_ratio': None,
   'logistic__max_iter': 100,
   'logistic__multi_class': 'auto',
   'logistic__n_jobs': None,
   'logistic__penalty': 'l2',
   'logistic__random_state': None,
   'logistic__solver': 'lbfgs',
   'logistic__tol': 0.0001,
   'logistic__verbose': 0,
   'logistic__warm_start': False}},
 'eda_chest': {'logistic': {'C': 0.1,
   'class_weight': None,
   'dual': False,
   'fit_intercept': True,
   'intercept_scaling': 1,
   'l1_ratio': None,
   'max_iter': 10000,
   'multi_class': 'auto',
   'n_jobs': None,
   'penalty': 'l1',
   'random_state': None,
   'solver': 'liblinear',
   'tol': 0.0001,
   'verbose': 0,
   'warm_start': False},
  'xgboost': {'objective': 'binary:logistic',
   'base_score': 0.5,
   'booster': 'gbtree',
   'colsample_bylevel': 1,
   'colsample_bynode': 1,
   'colsample_bytree': 0.5,
   'gamma': 0.5,
   'gpu_id': -1,
   'importance_type': 'gain',
   'interaction_constraints': '',
   'learning_rate': 0.200000003,
   'max_delta_step': 0,
   'max_depth': 3,
   'min_child_weight': 1,
   'missing': np.nan,
   'monotone_constraints': '()',
   'n_estimators': 100,
   'n_jobs': 0,
   'num_parallel_tree': 1,
   'random_state': 0,
   'reg_alpha': 0,
   'reg_lambda': 1,
   'scale_pos_weight': 1,
   'subsample': 1,
   'tree_method': 'exact',
   'validate_parameters': 1,
   'verbosity': None,
   'alpha': 0,
   'eta': 0.2},
  'linear_svm': {'C': 1,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'linear',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'rbf': {'C': 1,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'rbf',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'polynomial': {'C': 10,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 5,
   'gamma': 'scale',
   'kernel': 'poly',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'pls_rbf': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()), ('rbf_svm', SVC())],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'rbf_svm': SVC(),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'rbf_svm__C': 1.0,
   'rbf_svm__break_ties': False,
   'rbf_svm__cache_size': 200,
   'rbf_svm__class_weight': None,
   'rbf_svm__coef0': 0.0,
   'rbf_svm__decision_function_shape': 'ovr',
   'rbf_svm__degree': 3,
   'rbf_svm__gamma': 'scale',
   'rbf_svm__kernel': 'rbf',
   'rbf_svm__max_iter': -1,
   'rbf_svm__probability': False,
   'rbf_svm__random_state': None,
   'rbf_svm__shrinking': True,
   'rbf_svm__tol': 0.001,
   'rbf_svm__verbose': False},
  'pls_linear': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('linear_svm', SVC(kernel='linear'))],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'linear_svm': SVC(kernel='linear'),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'linear_svm__C': 1.0,
   'linear_svm__break_ties': False,
   'linear_svm__cache_size': 200,
   'linear_svm__class_weight': None,
   'linear_svm__coef0': 0.0,
   'linear_svm__decision_function_shape': 'ovr',
   'linear_svm__degree': 3,
   'linear_svm__gamma': 'scale',
   'linear_svm__kernel': 'linear',
   'linear_svm__max_iter': -1,
   'linear_svm__probability': False,
   'linear_svm__random_state': None,
   'linear_svm__shrinking': True,
   'linear_svm__tol': 0.001,
   'linear_svm__verbose': False},
  'pls_polynomial': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('poly_svm', SVC(kernel='poly'))],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'poly_svm': SVC(kernel='poly'),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'poly_svm__C': 1.0,
   'poly_svm__break_ties': False,
   'poly_svm__cache_size': 200,
   'poly_svm__class_weight': None,
   'poly_svm__coef0': 0.0,
   'poly_svm__decision_function_shape': 'ovr',
   'poly_svm__degree': 3,
   'poly_svm__gamma': 'scale',
   'poly_svm__kernel': 'poly',
   'poly_svm__max_iter': -1,
   'poly_svm__probability': False,
   'poly_svm__random_state': None,
   'poly_svm__shrinking': True,
   'poly_svm__tol': 0.001,
   'poly_svm__verbose': False},
  'pls_lr': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('logistic', LogisticRegression())],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'logistic': LogisticRegression(),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'logistic__C': 1.0,
   'logistic__class_weight': None,
   'logistic__dual': False,
   'logistic__fit_intercept': True,
   'logistic__intercept_scaling': 1,
   'logistic__l1_ratio': None,
   'logistic__max_iter': 100,
   'logistic__multi_class': 'auto',
   'logistic__n_jobs': None,
   'logistic__penalty': 'l2',
   'logistic__random_state': None,
   'logistic__solver': 'lbfgs',
   'logistic__tol': 0.0001,
   'logistic__verbose': 0,
   'logistic__warm_start': False}},
 'eda_wrist': {'logistic': {'C': 1,
   'class_weight': None,
   'dual': False,
   'fit_intercept': True,
   'intercept_scaling': 1,
   'l1_ratio': None,
   'max_iter': 10000,
   'multi_class': 'auto',
   'n_jobs': None,
   'penalty': 'l1',
   'random_state': None,
   'solver': 'liblinear',
   'tol': 0.0001,
   'verbose': 0,
   'warm_start': False},
  'xgboost': {'objective': 'binary:logistic',
   'base_score': 0.5,
   'booster': 'gbtree',
   'colsample_bylevel': 1,
   'colsample_bynode': 1,
   'colsample_bytree': 0.5,
   'gamma': 0,
   'gpu_id': -1,
   'importance_type': 'gain',
   'interaction_constraints': '',
   'learning_rate': 0.00999999978,
   'max_delta_step': 0,
   'max_depth': 3,
   'min_child_weight': 1,
   'missing': np.nan,
   'monotone_constraints': '()',
   'n_estimators': 100,
   'n_jobs': 0,
   'num_parallel_tree': 1,
   'random_state': 0,
   'reg_alpha': 0,
   'reg_lambda': 1,
   'scale_pos_weight': 1,
   'subsample': 1,
   'tree_method': 'exact',
   'validate_parameters': 1,
   'verbosity': None,
   'alpha': 0,
   'eta': 0.01},
  'linear_svm': {'C': 10,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'linear',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'rbf': {'C': 100,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'rbf',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'polynomial': {'C': 100,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 5,
   'gamma': 'scale',
   'kernel': 'poly',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'pls_rbf': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()), ('rbf_svm', SVC())],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'rbf_svm': SVC(),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'rbf_svm__C': 1.0,
   'rbf_svm__break_ties': False,
   'rbf_svm__cache_size': 200,
   'rbf_svm__class_weight': None,
   'rbf_svm__coef0': 0.0,
   'rbf_svm__decision_function_shape': 'ovr',
   'rbf_svm__degree': 3,
   'rbf_svm__gamma': 'scale',
   'rbf_svm__kernel': 'rbf',
   'rbf_svm__max_iter': -1,
   'rbf_svm__probability': False,
   'rbf_svm__random_state': None,
   'rbf_svm__shrinking': True,
   'rbf_svm__tol': 0.001,
   'rbf_svm__verbose': False},
  'pls_linear': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('linear_svm', SVC(kernel='linear'))],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'linear_svm': SVC(kernel='linear'),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'linear_svm__C': 1.0,
   'linear_svm__break_ties': False,
   'linear_svm__cache_size': 200,
   'linear_svm__class_weight': None,
   'linear_svm__coef0': 0.0,
   'linear_svm__decision_function_shape': 'ovr',
   'linear_svm__degree': 3,
   'linear_svm__gamma': 'scale',
   'linear_svm__kernel': 'linear',
   'linear_svm__max_iter': -1,
   'linear_svm__probability': False,
   'linear_svm__random_state': None,
   'linear_svm__shrinking': True,
   'linear_svm__tol': 0.001,
   'linear_svm__verbose': False},
  'pls_polynomial': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('poly_svm', SVC(kernel='poly'))],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'poly_svm': SVC(kernel='poly'),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'poly_svm__C': 1.0,
   'poly_svm__break_ties': False,
   'poly_svm__cache_size': 200,
   'poly_svm__class_weight': None,
   'poly_svm__coef0': 0.0,
   'poly_svm__decision_function_shape': 'ovr',
   'poly_svm__degree': 3,
   'poly_svm__gamma': 'scale',
   'poly_svm__kernel': 'poly',
   'poly_svm__max_iter': -1,
   'poly_svm__probability': False,
   'poly_svm__random_state': None,
   'poly_svm__shrinking': True,
   'poly_svm__tol': 0.001,
   'poly_svm__verbose': False},
  'pls_lr': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('logistic', LogisticRegression())],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'logistic': LogisticRegression(),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'logistic__C': 1.0,
   'logistic__class_weight': None,
   'logistic__dual': False,
   'logistic__fit_intercept': True,
   'logistic__intercept_scaling': 1,
   'logistic__l1_ratio': None,
   'logistic__max_iter': 100,
   'logistic__multi_class': 'auto',
   'logistic__n_jobs': None,
   'logistic__penalty': 'l2',
   'logistic__random_state': None,
   'logistic__solver': 'lbfgs',
   'logistic__tol': 0.0001,
   'logistic__verbose': 0,
   'logistic__warm_start': False}},
 'resp_only': {'logistic': {'C': 1,
   'class_weight': None,
   'dual': False,
   'fit_intercept': True,
   'intercept_scaling': 1,
   'l1_ratio': None,
   'max_iter': 10000,
   'multi_class': 'auto',
   'n_jobs': None,
   'penalty': 'l1',
   'random_state': None,
   'solver': 'liblinear',
   'tol': 0.0001,
   'verbose': 0,
   'warm_start': False},
  'xgboost': {'objective': 'binary:logistic',
   'base_score': 0.5,
   'booster': 'gbtree',
   'colsample_bylevel': 1,
   'colsample_bynode': 1,
   'colsample_bytree': 0.5,
   'gamma': 0.5,
   'gpu_id': -1,
   'importance_type': 'gain',
   'interaction_constraints': '',
   'learning_rate': 0.200000003,
   'max_delta_step': 0,
   'max_depth': 3,
   'min_child_weight': 1,
   'missing': np.nan,
   'monotone_constraints': '()',
   'n_estimators': 100,
   'n_jobs': 0,
   'num_parallel_tree': 1,
   'random_state': 0,
   'reg_alpha': 1,
   'reg_lambda': 1,
   'scale_pos_weight': 1,
   'subsample': 0.5,
   'tree_method': 'exact',
   'validate_parameters': 1,
   'verbosity': None,
   'alpha': 1,
   'eta': 0.2},
  'linear_svm': {'C': 10,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'linear',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'rbf': {'C': 1,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'rbf',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'polynomial': {'C': 100,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'poly',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'pls_rbf': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()), ('rbf_svm', SVC())],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'rbf_svm': SVC(),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'rbf_svm__C': 1.0,
   'rbf_svm__break_ties': False,
   'rbf_svm__cache_size': 200,
   'rbf_svm__class_weight': None,
   'rbf_svm__coef0': 0.0,
   'rbf_svm__decision_function_shape': 'ovr',
   'rbf_svm__degree': 3,
   'rbf_svm__gamma': 'scale',
   'rbf_svm__kernel': 'rbf',
   'rbf_svm__max_iter': -1,
   'rbf_svm__probability': False,
   'rbf_svm__random_state': None,
   'rbf_svm__shrinking': True,
   'rbf_svm__tol': 0.001,
   'rbf_svm__verbose': False},
  'pls_linear': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('linear_svm', SVC(kernel='linear'))],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'linear_svm': SVC(kernel='linear'),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'linear_svm__C': 1.0,
   'linear_svm__break_ties': False,
   'linear_svm__cache_size': 200,
   'linear_svm__class_weight': None,
   'linear_svm__coef0': 0.0,
   'linear_svm__decision_function_shape': 'ovr',
   'linear_svm__degree': 3,
   'linear_svm__gamma': 'scale',
   'linear_svm__kernel': 'linear',
   'linear_svm__max_iter': -1,
   'linear_svm__probability': False,
   'linear_svm__random_state': None,
   'linear_svm__shrinking': True,
   'linear_svm__tol': 0.001,
   'linear_svm__verbose': False},
  'pls_polynomial': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('poly_svm', SVC(kernel='poly'))],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'poly_svm': SVC(kernel='poly'),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'poly_svm__C': 1.0,
   'poly_svm__break_ties': False,
   'poly_svm__cache_size': 200,
   'poly_svm__class_weight': None,
   'poly_svm__coef0': 0.0,
   'poly_svm__decision_function_shape': 'ovr',
   'poly_svm__degree': 3,
   'poly_svm__gamma': 'scale',
   'poly_svm__kernel': 'poly',
   'poly_svm__max_iter': -1,
   'poly_svm__probability': False,
   'poly_svm__random_state': None,
   'poly_svm__shrinking': True,
   'poly_svm__tol': 0.001,
   'poly_svm__verbose': False},
  'pls_lr': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('logistic', LogisticRegression())],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'logistic': LogisticRegression(),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'logistic__C': 1.0,
   'logistic__class_weight': None,
   'logistic__dual': False,
   'logistic__fit_intercept': True,
   'logistic__intercept_scaling': 1,
   'logistic__l1_ratio': None,
   'logistic__max_iter': 100,
   'logistic__multi_class': 'auto',
   'logistic__n_jobs': None,
   'logistic__penalty': 'l2',
   'logistic__random_state': None,
   'logistic__solver': 'lbfgs',
   'logistic__tol': 0.0001,
   'logistic__verbose': 0,
   'logistic__warm_start': False}},
 'acc_only': {'logistic': {'C': 100,
   'class_weight': None,
   'dual': False,
   'fit_intercept': True,
   'intercept_scaling': 1,
   'l1_ratio': None,
   'max_iter': 10000,
   'multi_class': 'auto',
   'n_jobs': None,
   'penalty': 'l1',
   'random_state': None,
   'solver': 'liblinear',
   'tol': 0.0001,
   'verbose': 0,
   'warm_start': False},
  'xgboost': {'objective': 'binary:logistic',
   'base_score': 0.5,
   'booster': 'gbtree',
   'colsample_bylevel': 1,
   'colsample_bynode': 1,
   'colsample_bytree': 0.5,
   'gamma': 0,
   'gpu_id': -1,
   'importance_type': 'gain',
   'interaction_constraints': '',
   'learning_rate': 0.00999999978,
   'max_delta_step': 0,
   'max_depth': 3,
   'min_child_weight': 1,
   'missing': np.nan,
   'monotone_constraints': '()',
   'n_estimators': 100,
   'n_jobs': 0,
   'num_parallel_tree': 1,
   'random_state': 0,
   'reg_alpha': 10,
   'reg_lambda': 1,
   'scale_pos_weight': 1,
   'subsample': 0.5,
   'tree_method': 'exact',
   'validate_parameters': 1,
   'verbosity': None,
   'alpha': 10,
   'eta': 0.01},
  'linear_svm': {'C': 1,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'linear',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'rbf': {'C': 1,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'rbf',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'polynomial': {'C': 10,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 2,
   'gamma': 'scale',
   'kernel': 'poly',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'pls_rbf': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()), ('rbf_svm', SVC())],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'rbf_svm': SVC(),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'rbf_svm__C': 1.0,
   'rbf_svm__break_ties': False,
   'rbf_svm__cache_size': 200,
   'rbf_svm__class_weight': None,
   'rbf_svm__coef0': 0.0,
   'rbf_svm__decision_function_shape': 'ovr',
   'rbf_svm__degree': 3,
   'rbf_svm__gamma': 'scale',
   'rbf_svm__kernel': 'rbf',
   'rbf_svm__max_iter': -1,
   'rbf_svm__probability': False,
   'rbf_svm__random_state': None,
   'rbf_svm__shrinking': True,
   'rbf_svm__tol': 0.001,
   'rbf_svm__verbose': False},
  'pls_linear': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('linear_svm', SVC(kernel='linear'))],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'linear_svm': SVC(kernel='linear'),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'linear_svm__C': 1.0,
   'linear_svm__break_ties': False,
   'linear_svm__cache_size': 200,
   'linear_svm__class_weight': None,
   'linear_svm__coef0': 0.0,
   'linear_svm__decision_function_shape': 'ovr',
   'linear_svm__degree': 3,
   'linear_svm__gamma': 'scale',
   'linear_svm__kernel': 'linear',
   'linear_svm__max_iter': -1,
   'linear_svm__probability': False,
   'linear_svm__random_state': None,
   'linear_svm__shrinking': True,
   'linear_svm__tol': 0.001,
   'linear_svm__verbose': False},
  'pls_polynomial': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('poly_svm', SVC(kernel='poly'))],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'poly_svm': SVC(kernel='poly'),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'poly_svm__C': 1.0,
   'poly_svm__break_ties': False,
   'poly_svm__cache_size': 200,
   'poly_svm__class_weight': None,
   'poly_svm__coef0': 0.0,
   'poly_svm__decision_function_shape': 'ovr',
   'poly_svm__degree': 3,
   'poly_svm__gamma': 'scale',
   'poly_svm__kernel': 'poly',
   'poly_svm__max_iter': -1,
   'poly_svm__probability': False,
   'poly_svm__random_state': None,
   'poly_svm__shrinking': True,
   'poly_svm__tol': 0.001,
   'poly_svm__verbose': False},
  'pls_lr': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('logistic', LogisticRegression())],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'logistic': LogisticRegression(),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'logistic__C': 1.0,
   'logistic__class_weight': None,
   'logistic__dual': False,
   'logistic__fit_intercept': True,
   'logistic__intercept_scaling': 1,
   'logistic__l1_ratio': None,
   'logistic__max_iter': 100,
   'logistic__multi_class': 'auto',
   'logistic__n_jobs': None,
   'logistic__penalty': 'l2',
   'logistic__random_state': None,
   'logistic__solver': 'lbfgs',
   'logistic__tol': 0.0001,
   'logistic__verbose': 0,
   'logistic__warm_start': False}},
 'bvp_only': {'logistic': {'C': 100,
   'class_weight': None,
   'dual': False,
   'fit_intercept': True,
   'intercept_scaling': 1,
   'l1_ratio': None,
   'max_iter': 10000,
   'multi_class': 'auto',
   'n_jobs': None,
   'penalty': 'l1',
   'random_state': None,
   'solver': 'liblinear',
   'tol': 0.0001,
   'verbose': 0,
   'warm_start': False},
  'xgboost': {'objective': 'binary:logistic',
   'base_score': 0.5,
   'booster': 'gbtree',
   'colsample_bylevel': 1,
   'colsample_bynode': 1,
   'colsample_bytree': 0.5,
   'gamma': 0.5,
   'gpu_id': -1,
   'importance_type': 'gain',
   'interaction_constraints': '',
   'learning_rate': 0.00999999978,
   'max_delta_step': 0,
   'max_depth': 3,
   'min_child_weight': 1,
   'missing': np.nan,
   'monotone_constraints': '()',
   'n_estimators': 100,
   'n_jobs': 0,
   'num_parallel_tree': 1,
   'random_state': 0,
   'reg_alpha': 10,
   'reg_lambda': 1,
   'scale_pos_weight': 1,
   'subsample': 1,
   'tree_method': 'exact',
   'validate_parameters': 1,
   'verbosity': None,
   'alpha': 10,
   'eta': 0.01},
  'linear_svm': {'C': 1,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'linear',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'rbf': {'C': 10,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'rbf',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'polynomial': {'C': 10,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 2,
   'gamma': 'scale',
   'kernel': 'poly',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'pls_rbf': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()), ('rbf_svm', SVC())],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'rbf_svm': SVC(),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'rbf_svm__C': 1.0,
   'rbf_svm__break_ties': False,
   'rbf_svm__cache_size': 200,
   'rbf_svm__class_weight': None,
   'rbf_svm__coef0': 0.0,
   'rbf_svm__decision_function_shape': 'ovr',
   'rbf_svm__degree': 3,
   'rbf_svm__gamma': 'scale',
   'rbf_svm__kernel': 'rbf',
   'rbf_svm__max_iter': -1,
   'rbf_svm__probability': False,
   'rbf_svm__random_state': None,
   'rbf_svm__shrinking': True,
   'rbf_svm__tol': 0.001,
   'rbf_svm__verbose': False},
  'pls_linear': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('linear_svm', SVC(kernel='linear'))],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'linear_svm': SVC(kernel='linear'),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'linear_svm__C': 1.0,
   'linear_svm__break_ties': False,
   'linear_svm__cache_size': 200,
   'linear_svm__class_weight': None,
   'linear_svm__coef0': 0.0,
   'linear_svm__decision_function_shape': 'ovr',
   'linear_svm__degree': 3,
   'linear_svm__gamma': 'scale',
   'linear_svm__kernel': 'linear',
   'linear_svm__max_iter': -1,
   'linear_svm__probability': False,
   'linear_svm__random_state': None,
   'linear_svm__shrinking': True,
   'linear_svm__tol': 0.001,
   'linear_svm__verbose': False},
  'pls_polynomial': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('poly_svm', SVC(kernel='poly'))],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'poly_svm': SVC(kernel='poly'),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'poly_svm__C': 1.0,
   'poly_svm__break_ties': False,
   'poly_svm__cache_size': 200,
   'poly_svm__class_weight': None,
   'poly_svm__coef0': 0.0,
   'poly_svm__decision_function_shape': 'ovr',
   'poly_svm__degree': 3,
   'poly_svm__gamma': 'scale',
   'poly_svm__kernel': 'poly',
   'poly_svm__max_iter': -1,
   'poly_svm__probability': False,
   'poly_svm__random_state': None,
   'poly_svm__shrinking': True,
   'poly_svm__tol': 0.001,
   'poly_svm__verbose': False},
  'pls_lr': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('logistic', LogisticRegression())],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'logistic': LogisticRegression(),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'logistic__C': 1.0,
   'logistic__class_weight': None,
   'logistic__dual': False,
   'logistic__fit_intercept': True,
   'logistic__intercept_scaling': 1,
   'logistic__l1_ratio': None,
   'logistic__max_iter': 100,
   'logistic__multi_class': 'auto',
   'logistic__n_jobs': None,
   'logistic__penalty': 'l2',
   'logistic__random_state': None,
   'logistic__solver': 'lbfgs',
   'logistic__tol': 0.0001,
   'logistic__verbose': 0,
   'logistic__warm_start': False}},
 'wrist_only': {'logistic': {'C': 1,
   'class_weight': None,
   'dual': False,
   'fit_intercept': True,
   'intercept_scaling': 1,
   'l1_ratio': None,
   'max_iter': 10000,
   'multi_class': 'auto',
   'n_jobs': None,
   'penalty': 'l1',
   'random_state': None,
   'solver': 'liblinear',
   'tol': 0.0001,
   'verbose': 0,
   'warm_start': False},
  'xgboost': {'objective': 'binary:logistic',
   'base_score': 0.5,
   'booster': 'gbtree',
   'colsample_bylevel': 1,
   'colsample_bynode': 1,
   'colsample_bytree': 0.5,
   'gamma': 0,
   'gpu_id': -1,
   'importance_type': 'gain',
   'interaction_constraints': '',
   'learning_rate': 0.00999999978,
   'max_delta_step': 0,
   'max_depth': 3,
   'min_child_weight': 1,
   'missing': np.nan,
   'monotone_constraints': '()',
   'n_estimators': 100,
   'n_jobs': 0,
   'num_parallel_tree': 1,
   'random_state': 0,
   'reg_alpha': 10,
   'reg_lambda': 1,
   'scale_pos_weight': 1,
   'subsample': 1,
   'tree_method': 'exact',
   'validate_parameters': 1,
   'verbosity': None,
   'alpha': 10,
   'eta': 0.01},
  'linear_svm': {'C': 1,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'linear',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'rbf': {'C': 1,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'rbf',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'polynomial': {'C': 1,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'poly',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'pls_rbf': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()), ('rbf_svm', SVC())],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'rbf_svm': SVC(),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'rbf_svm__C': 1.0,
   'rbf_svm__break_ties': False,
   'rbf_svm__cache_size': 200,
   'rbf_svm__class_weight': None,
   'rbf_svm__coef0': 0.0,
   'rbf_svm__decision_function_shape': 'ovr',
   'rbf_svm__degree': 3,
   'rbf_svm__gamma': 'scale',
   'rbf_svm__kernel': 'rbf',
   'rbf_svm__max_iter': -1,
   'rbf_svm__probability': False,
   'rbf_svm__random_state': None,
   'rbf_svm__shrinking': True,
   'rbf_svm__tol': 0.001,
   'rbf_svm__verbose': False},
  'pls_linear': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('linear_svm', SVC(kernel='linear'))],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'linear_svm': SVC(kernel='linear'),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'linear_svm__C': 1.0,
   'linear_svm__break_ties': False,
   'linear_svm__cache_size': 200,
   'linear_svm__class_weight': None,
   'linear_svm__coef0': 0.0,
   'linear_svm__decision_function_shape': 'ovr',
   'linear_svm__degree': 3,
   'linear_svm__gamma': 'scale',
   'linear_svm__kernel': 'linear',
   'linear_svm__max_iter': -1,
   'linear_svm__probability': False,
   'linear_svm__random_state': None,
   'linear_svm__shrinking': True,
   'linear_svm__tol': 0.001,
   'linear_svm__verbose': False},
  'pls_polynomial': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('poly_svm', SVC(kernel='poly'))],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'poly_svm': SVC(kernel='poly'),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'poly_svm__C': 1.0,
   'poly_svm__break_ties': False,
   'poly_svm__cache_size': 200,
   'poly_svm__class_weight': None,
   'poly_svm__coef0': 0.0,
   'poly_svm__decision_function_shape': 'ovr',
   'poly_svm__degree': 3,
   'poly_svm__gamma': 'scale',
   'poly_svm__kernel': 'poly',
   'poly_svm__max_iter': -1,
   'poly_svm__probability': False,
   'poly_svm__random_state': None,
   'poly_svm__shrinking': True,
   'poly_svm__tol': 0.001,
   'poly_svm__verbose': False},
  'pls_lr': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('logistic', LogisticRegression())],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'logistic': LogisticRegression(),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'logistic__C': 1.0,
   'logistic__class_weight': None,
   'logistic__dual': False,
   'logistic__fit_intercept': True,
   'logistic__intercept_scaling': 1,
   'logistic__l1_ratio': None,
   'logistic__max_iter': 100,
   'logistic__multi_class': 'auto',
   'logistic__n_jobs': None,
   'logistic__penalty': 'l2',
   'logistic__random_state': None,
   'logistic__solver': 'lbfgs',
   'logistic__tol': 0.0001,
   'logistic__verbose': 0,
   'logistic__warm_start': False}},
 'chest_only': {'logistic': {'C': 1,
   'class_weight': None,
   'dual': False,
   'fit_intercept': True,
   'intercept_scaling': 1,
   'l1_ratio': None,
   'max_iter': 10000,
   'multi_class': 'auto',
   'n_jobs': None,
   'penalty': 'l1',
   'random_state': None,
   'solver': 'liblinear',
   'tol': 0.0001,
   'verbose': 0,
   'warm_start': False},
  'xgboost': {'objective': 'binary:logistic',
   'base_score': 0.5,
   'booster': 'gbtree',
   'colsample_bylevel': 1,
   'colsample_bynode': 1,
   'colsample_bytree': 1,
   'gamma': 0,
   'gpu_id': -1,
   'importance_type': 'gain',
   'interaction_constraints': '',
   'learning_rate': 0.00999999978,
   'max_delta_step': 0,
   'max_depth': 7,
   'min_child_weight': 1,
   'missing': np.nan,
   'monotone_constraints': '()',
   'n_estimators': 100,
   'n_jobs': 0,
   'num_parallel_tree': 1,
   'random_state': 0,
   'reg_alpha': 0,
   'reg_lambda': 1,
   'scale_pos_weight': 1,
   'subsample': 1,
   'tree_method': 'exact',
   'validate_parameters': 1,
   'verbosity': None,
   'alpha': 0,
   'eta': 0.01},
  'linear_svm': {'C': 10,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'linear',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'rbf': {'C': 1,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'rbf',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'polynomial': {'C': 100,
   'break_ties': False,
   'cache_size': 200,
   'class_weight': None,
   'coef0': 0.0,
   'decision_function_shape': 'ovr',
   'degree': 3,
   'gamma': 'scale',
   'kernel': 'poly',
   'max_iter': -1,
   'probability': False,
   'random_state': None,
   'shrinking': True,
   'tol': 0.001,
   'verbose': False},
  'pls_rbf': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()), ('rbf_svm', SVC())],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'rbf_svm': SVC(),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'rbf_svm__C': 1.0,
   'rbf_svm__break_ties': False,
   'rbf_svm__cache_size': 200,
   'rbf_svm__class_weight': None,
   'rbf_svm__coef0': 0.0,
   'rbf_svm__decision_function_shape': 'ovr',
   'rbf_svm__degree': 3,
   'rbf_svm__gamma': 'scale',
   'rbf_svm__kernel': 'rbf',
   'rbf_svm__max_iter': -1,
   'rbf_svm__probability': False,
   'rbf_svm__random_state': None,
   'rbf_svm__shrinking': True,
   'rbf_svm__tol': 0.001,
   'rbf_svm__verbose': False},
  'pls_linear': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('linear_svm', SVC(kernel='linear'))],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'linear_svm': SVC(kernel='linear'),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'linear_svm__C': 1.0,
   'linear_svm__break_ties': False,
   'linear_svm__cache_size': 200,
   'linear_svm__class_weight': None,
   'linear_svm__coef0': 0.0,
   'linear_svm__decision_function_shape': 'ovr',
   'linear_svm__degree': 3,
   'linear_svm__gamma': 'scale',
   'linear_svm__kernel': 'linear',
   'linear_svm__max_iter': -1,
   'linear_svm__probability': False,
   'linear_svm__random_state': None,
   'linear_svm__shrinking': True,
   'linear_svm__tol': 0.001,
   'linear_svm__verbose': False},
  'pls_polynomial': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('poly_svm', SVC(kernel='poly'))],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'poly_svm': SVC(kernel='poly'),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'poly_svm__C': 1.0,
   'poly_svm__break_ties': False,
   'poly_svm__cache_size': 200,
   'poly_svm__class_weight': None,
   'poly_svm__coef0': 0.0,
   'poly_svm__decision_function_shape': 'ovr',
   'poly_svm__degree': 3,
   'poly_svm__gamma': 'scale',
   'poly_svm__kernel': 'poly',
   'poly_svm__max_iter': -1,
   'poly_svm__probability': False,
   'poly_svm__random_state': None,
   'poly_svm__shrinking': True,
   'poly_svm__tol': 0.001,
   'poly_svm__verbose': False},
  'pls_lr': {'memory': None,
   'steps': [('pls', PLSRegressionWrapper()),
    ('logistic', LogisticRegression())],
   'verbose': False,
   'pls': PLSRegressionWrapper(),
   'logistic': LogisticRegression(),
   'pls__copy': True,
   'pls__max_iter': 500,
   'pls__n_components': 2,
   'pls__scale': True,
   'pls__tol': 1e-06,
   'logistic__C': 1.0,
   'logistic__class_weight': None,
   'logistic__dual': False,
   'logistic__fit_intercept': True,
   'logistic__intercept_scaling': 1,
   'logistic__l1_ratio': None,
   'logistic__max_iter': 100,
   'logistic__multi_class': 'auto',
   'logistic__n_jobs': None,
   'logistic__penalty': 'l2',
   'logistic__random_state': None,
   'logistic__solver': 'lbfgs',
   'logistic__tol': 0.0001,
   'logistic__verbose': 0,
   'logistic__warm_start': False}}}




for var_subset_name, var_subset_columns in var_subsets: 
    print(f"Repeating for {var_subset_name}...")
    final_subset = final[var_subset_columns + ["label"]]
    def inf_to_mean(X):
        """
        Takes numpy array X and returns a version replacing inf and na values with their column means
        """
        X = np.nan_to_num(X, nan = np.nan, posinf = np.nan)
        col_mean = np.nanmean(X, axis = 0)
        inds = np.where(np.isnan(X)) 
        X[inds] = np.take(col_mean, inds[1]) 
        return X

    # Format data into numpy arrays to be used in sklearn models
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Y = np.array(final_subset["label"])
    X = final_subset.drop(columns = ["label"])
    X = inf_to_mean(X.to_numpy())
    X = scaler.fit_transform(X)

    X_colnames = final_subset.drop(columns = ["label"]).columns

    best_models = {}
    model_accuracies = {}
    model_f1 = {}

    # Initialize penalized logistic regression
    lr = LogisticRegression()
    lr.set_params(**precomputed_params[var_subset_name]["logistic"])
    lr.fit(X,Y)
    best_lr = lr


    # Variables chosen by the Lasso model and their coefficients
    coef_names_ordered = []
    coef_values_ordered = []
    print("Coefficients:")
    lr_coef_best_idx = np.flip(np.argsort(np.abs(best_lr.coef_).reshape(-1)))
    lr_coef_best = best_lr.coef_.reshape(-1)[lr_coef_best_idx]
    coef_touse = []
    coef_toexclude = []
    for idx, coef in zip(lr_coef_best_idx, lr_coef_best):
        if coef != 0:
            print(f"{X_colnames[idx]}: {coef}")
            coef_touse.append(X_colnames[idx])
            coef_names_ordered.append(X_colnames[idx])
            coef_values_ordered.append(coef)
        else:
            coef_toexclude.append(X_colnames[idx])

    # Export coefficient table
    pd.DataFrame({"Variable Name":coef_names_ordered, "Coefficient":coef_values_ordered}).to_csv(f"lasso_logistic_coefs_{var_subset_name}.csv", index = False)

    lr_cv_preds = cross_val_predict(best_lr, X,Y, cv = 15)
    model_accuracies["logistic"] = accuracy_score(lr_cv_preds, Y.reshape(-1))
    model_f1["logistic"] = f1_score(lr_cv_preds, Y.reshape(-1))

    # Below is code to get the XGB Model from scratch using CV

    # Initialize XGBoost Model
    xgb = XGBClassifier()
    xgb.set_params(**precomputed_params[var_subset_name]["xgboost"])
    xgb.fit(X,Y)
    best_xgb = xgb

    fi = best_xgb.feature_importances_
    fi_best_idx = np.flip(fi.argsort())
    fi_best = np.flip(np.sort(fi))
    fi_vars_ordered = []
    fi_values_ordered = []
    print("MOST IMPORTANT FEATURES:\n")
    for i in range(len(fi_best_idx)):
        print("Feature name: {:>12}     Feature importance: {:>12}".format(X_colnames[fi_best_idx[i]], fi_best[i]))
        fi_vars_ordered.append(X_colnames[fi_best_idx[i]])
        fi_values_ordered.append(fi_best[i])

    pd.DataFrame({"Variable Name":fi_vars_ordered, "Feature Importance":fi_values_ordered}).to_csv(f"feature_importance_chart_{var_subset_name}.csv", index = False)

    xgb_cv_preds = cross_val_predict(best_xgb, X,Y, cv = 15, n_jobs = -1)
    model_accuracies["xgb"] = accuracy_score(xgb_cv_preds, Y.reshape(-1))
    model_f1["xgb"] = f1_score(xgb_cv_preds, Y.reshape(-1))





    best_linear_svm = SVC()
    best_linear_svm.set_params(**precomputed_params[var_subset_name]["linear_svm"])
    best_linear_svm.fit(X,Y)
    best_models["linear_svm"] = best_linear_svm
    linear_svm_cv_preds = cross_val_predict(best_linear_svm, X,Y, cv = 15)
    model_accuracies["linear_svm"] = accuracy_score(linear_svm_cv_preds, Y.reshape(-1))
    model_f1["linear_svm"] = f1_score(linear_svm_cv_preds, Y.reshape(-1))

    best_rbf_svm = SVC()
    best_rbf_svm.set_params(**precomputed_params[var_subset_name]["rbf"])
    best_rbf_svm.fit(X,Y)
    best_models["rbf_svm"] = best_rbf_svm
    rbf_svm_cv_preds = cross_val_predict(best_rbf_svm, X,Y, cv = 15)
    model_accuracies["rbf_svm"] = accuracy_score(rbf_svm_cv_preds, Y.reshape(-1))
    model_f1["rbf_svm"] = f1_score(rbf_svm_cv_preds, Y.reshape(-1))

    best_poly_svm = SVC()
    best_poly_svm.set_params(**precomputed_params[var_subset_name]["polynomial"])
    best_poly_svm.fit(X,Y)
    best_models["poly_svm"] = best_poly_svm
    poly_svm_cv_preds = cross_val_predict(best_poly_svm, X,Y, cv = 15)
    model_accuracies["poly_svm"] = accuracy_score(poly_svm_cv_preds, Y.reshape(-1))
    model_f1["poly_svm"] = f1_score(poly_svm_cv_preds, Y.reshape(-1))

    pls_rbf = Pipeline([("pls", PLSRegressionWrapper()), ("rbf_svm", SVC(kernel = "rbf"))])
    pls_rbf.set_params(**precomputed_params[var_subset_name]["pls_rbf"])
    pls_rbf.fit(X,Y)
    pls_rbf_cv_preds = cross_val_predict(pls_rbf, X, Y, cv = 15, n_jobs = -1)
    model_accuracies["pls_rbf"] = accuracy_score(pls_rbf_cv_preds, Y.reshape(-1))
    model_f1["pls_rbf"] = f1_score(pls_rbf_cv_preds, Y.reshape(-1))

    pls_linear = Pipeline([("pls", PLSRegressionWrapper()), ("rbf_svm", SVC(kernel = "rbf"))])
    pls_linear.set_params(**precomputed_params[var_subset_name]["pls_linear"])
    pls_linear.fit(X,Y)
    pls_linear_cv_preds = cross_val_predict(pls_linear, X, Y, cv = 15, n_jobs = -1)
    model_accuracies["pls_linear"] = accuracy_score(pls_linear_cv_preds, Y.reshape(-1))
    model_f1["pls_linear"] = f1_score(pls_linear_cv_preds, Y.reshape(-1))

    pls_polynomial = Pipeline([("pls", PLSRegressionWrapper()), ("rbf_svm", SVC(kernel = "rbf"))])
    pls_polynomial.set_params(**precomputed_params[var_subset_name]["pls_polynomial"])
    pls_polynomial.fit(X,Y)
    pls_polynomial_cv_preds = cross_val_predict(pls_polynomial, X, Y, cv = 15, n_jobs = -1)
    model_accuracies["pls_poly"] = accuracy_score(pls_polynomial_cv_preds, Y.reshape(-1))
    model_f1["pls_poly"] = f1_score(pls_polynomial_cv_preds, Y.reshape(-1))

    pls_lr = Pipeline([("pls", PLSRegressionWrapper()), ("rbf_svm", SVC(kernel = "rbf"))])
    pls_lr.set_params(**precomputed_params[var_subset_name]["pls_lr"])
    pls_lr.fit(X,Y)
    pls_lr_cv_preds = cross_val_predict(pls_lr, X, Y, cv = 15, n_jobs = -1)
    model_accuracies["pls_lr"] = accuracy_score(pls_lr_cv_preds, Y.reshape(-1))
    model_f1["pls_lr"] = f1_score(pls_lr_cv_preds, Y.reshape(-1))

    model_list = [("logistic", best_lr), ("xgboost", best_xgb)]
    final_estimator = LogisticRegression(max_iter = 10000, penalty = "l1", solver = "liblinear")
    stacking_estimator = StackingClassifier(estimators = model_list, final_estimator = final_estimator, cv = 15)

    y_pred_ensemble = cross_val_predict(stacking_estimator, X, Y, cv = 5, n_jobs = -1)
    model_accuracies["logistic_xgb_stack"] = accuracy_score(y_pred_ensemble, Y.reshape(-1))
    model_f1["logistic_xgb_stack"] = f1_score(y_pred_ensemble, Y.reshape(-1))

    #model_list = [("logistic", best_lr), ("xgboost", best_xgb), ("pls_rbf", pls_rbf), ("pls_linear", pls_linear), ("pls_poly", pls_polynomial), ("pls_logistic", pls_lr)]
    model_list = [("logistic", best_lr), ("xgboost", best_xgb), ("linear_svm", best_linear_svm), ("rbf", best_rbf_svm), ("polynomial", best_poly_svm),
                 ("pls_rbf", pls_rbf), ("pls_linear", pls_linear), ("pls_polynomial", pls_polynomial), ("pls_lr", pls_lr)]
    final_estimator = LogisticRegression(max_iter = 10000, penalty = "l1", solver = "liblinear")
    stacking_estimator = StackingClassifier(estimators = model_list, final_estimator = final_estimator, cv = 15)
    y_pred_full_ensemble = cross_val_predict(stacking_estimator, X, Y, cv = 5, n_jobs = -1)
    model_accuracies["full_stack"] = accuracy_score(y_pred_full_ensemble, Y.reshape(-1))
    model_f1["full_stack"] = f1_score(y_pred_full_ensemble, Y.reshape(-1))

    # Accuracy df
    pd.DataFrame({"Accuracy":model_accuracies, "F1 Score":model_f1}).to_csv(f"model_results_{var_subset_name}.csv", index = False)
