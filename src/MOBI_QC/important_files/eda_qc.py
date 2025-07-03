import pyxdf
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import wave
#import pyaudio
import numpy as np
#import sounddevice as sd
from utils import *
import scipy
from scipy.signal import iirnotch, filtfilt
from glob import glob
import neurokit2 as nk
from scipy.signal import butter, filtfilt

# Checking for nan or missing values in EDA data and return a percentage validity
def eda_signal_integrity_check(eda_df: pd.DataFrame) -> float:
    """
    Checks for missing values in EDA data and calculates the percentage of valid data.
    Args:
        eda_df (pd.DataFrame): DataFrame containing EDA data.
    Returns:
        eda_validity (float): Percentage of valid EDA data.
    """
    count_nan = 0
    for x in eda_df['EDA2']:
        if np.isnan(x) == True:
            count_nan = count_nan + 1

    eda_validity = 100 - (count_nan/len(eda_df['EDA2'])) * 100
    return eda_validity
# Preprocess EDA signal
def eda_preprocess(eda_df: pd.DataFrame, eda_sampling_rate: float) -> tuple[pd.DataFrame, dict]:
    """
    Preprocesses the EDA signal using NeuroKit2.
    Args:
        eda_df (pd.DataFrame): DataFrame containing EDA data.
        eda_sampling_rate (float): Sampling rate of the EDA data.
    Returns:
        eda_signals (pd.DataFrame): Processed EDA signals.
        info (dict): Additional information about the EDA processing.
    """
    eda_signals, info = nk.eda_process(eda_df['EDA2'], sampling_rate=eda_sampling_rate, method='neurokit')
    return eda_signals, info

def scl_stability(scl: pd.Series) -> tuple[float,float,float]:
    """
    Calculates the average, standard deviation, and coefficient of variation of SCL.
    Args:
        scl (pd.Series): Skin Conductance Level (SCL) data.
    Returns:
        average_scl (float): Average SCL.
        scl_sd (float): Standard deviation of SCL.
        scl_cv (float): Coefficient of variation of SCL.
    """
    average_scl = np.mean(scl)
    scl_sd = np.std(scl)
    scl_cv = (scl_sd / average_scl) * 100
    return average_scl, scl_sd, scl_cv

def scl_trend_analysis(eda_signals: pd.DataFrame, eda_df: pd.DataFrame, eda_sampling_rate: float, subject: str) -> plt:
    """
    Analyzes the trend of SCL by calculating slopes and rolling means, and generates a plot.
    Args:
        eda_signals (pd.DataFrame): Processed EDA signals.
        eda_df (pd.DataFrame): Original EDA data.
        eda_sampling_rate (float): Sampling rate of the EDA data.
        subject (str): Subject identifier for saving the plot.
    Returns:
        plt (matplotlib.pyplot): The generated plot.
    """
    scl_df = pd.DataFrame(eda_signals['EDA_Tonic'])
    scl_df['lsl_time_stamp'] = eda_df['lsl_time_stamp']

    # Calculating slope of SCL 
    scl_df['EDA_Tonic_Slope'] = np.gradient(scl_df['EDA_Tonic'], eda_df['lsl_time_stamp'])
    
    # Calculating rolling mean of SCL and Slope of SCL rolling mean
    rolling_mean = pd.Series(eda_signals['EDA_Tonic']).rolling(window=(int)(eda_sampling_rate), center=True).mean()
    slope_rolling_mean = np.gradient(rolling_mean)
    
    plt.figure(figsize = (10,3)) # figsize=(10,5), 20,5
    plt.plot(eda_df['lsl_time_stamp'], slope_rolling_mean, label='SCL Rolling_mean slope', color='orange', linestyle='-')
    plt.plot(eda_df['lsl_time_stamp'], scl_df['EDA_Tonic_Slope'], label='Slope of SCL', color='blue')
    plt.title('SCL Slope and Rolling Mean Slope Over Time', fontsize=16)
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Slope', fontsize=14)
    plt.legend(loc='upper right', fontsize=12)
    plt.savefig(f'report_images/{subject}_eda_slope.png', dpi = 100, bbox_inches='tight')
    # Show the plot
    plt.show()

    return plt

def scr_amplitudes(info: dict) -> tuple[float, float]:
    """
    Calculates the average amplitude of SCRs and their validity percentage.
    Args:
        info (dict): Additional information about the EDA processing.
    Returns:
        average_scr_amplitude (float): Average amplitude of SCRs.
        scr_amplitude_validity (float): Percentage of valid SCR amplitudes.
    """
    scr_amplitudes = [amplitude for amplitude in info['SCR_Amplitude'] if np.isnan(amplitude) != True]
    average_scr_amplitude = np.mean(scr_amplitudes)

    count_invalid_scr = 0
    for amplitude in scr_amplitudes:
        if amplitude > 0.01 and amplitude < 3.0:
            continue
        else:
            count_invalid_scr = count_invalid_scr + 1

    scr_amplitude_validity = 100 - (count_invalid_scr/len(scr_amplitudes)) * 100

    return average_scr_amplitude, scr_amplitude_validity

def eda_snr(eda_signals: pd.DataFrame, eda_df: pd.DataFrame, eda_sampling_rate: float) -> float:
    """
    Calculates the Signal-to-Noise Ratio (SNR) of the EDA signal.
    Args:
        eda_signals (pd.DataFrame): Processed EDA signals.
        eda_df (pd.DataFrame): Original EDA data.
        eda_sampling_rate (float): Sampling rate of the EDA data.
    Returns:
        snr (float): Signal-to-Noise Ratio in decibels (dB).
    """
    duration = len(eda_df['EDA2'].tolist()) / eda_sampling_rate
    t = np.linspace(0, duration , len(eda_df['EDA2']))
    t = t[:5000]

    eda_cleaned = eda_signals['EDA_Clean']
    signal_power = np.var(eda_cleaned)
    noise_signal = eda_signals['EDA_Raw'] - eda_cleaned  
    noise_power = np.var(noise_signal)

    snr = 10 * np.log10(signal_power / noise_power)
    return snr

def eda_report_plot(eda_signals: pd.DataFrame, info: dict, subject: str) -> plt:
    """
    Generates and saves a report plot for the EDA signals.
    Args:
        eda_signals (pd.DataFrame): Processed EDA signals.
        info (dict): Additional information about the EDA processing.
        subject (str): Subject identifier for saving the plot.
    Returns:
        plt (matplotlib.pyplot): The generated plot.
    """
    fig = nk.eda_plot(eda_signals, info)
    fig = plt.gcf()
    axes = fig.get_axes()
    fig.set_size_inches(16, 6) # 20, 10
    raw_signal_line = axes[0].lines[0]
    raw_signal_line.set_color('red')

    handles, labels = axes[0].get_legend_handles_labels()
    for handle, label in zip(handles, labels):
        if label == "EDA_Raw":  
            handle.set_color('red')  

    axes[0].legend(handles, labels)  
    axes[0].legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=10)
    plt.tight_layout()
    plt.savefig(f'report_images/{subject}_eda_report.png')
    plt.show()

    return plt

def eda_qc(xdf_filename: str, stim_df:pd.DataFrame, task='RestingState') -> tuple[dict, plt, plt, pd.DataFrame, bool]:
    """
    Performs quality control on EDA data from an XDF file.
    Args:
        xdf_filename (str): Path to the XDF file.
    Returns:
        vars (dict): Quality control metrics for the EDA data.
        eda_slope_fig (matplotlib.pyplot): SCL trend analysis plot.
        eda_report_fig (matplotlib.pyplot): EDA report plot.
        eda_error (bool): Indicates whether there was an error loading EDA data.
    """
    subject = xdf_filename.split('-')[1].split('/')[0]
    whole_ps_df = import_physio_data(xdf_filename)
    vars = {}
    vars['sampling_rate'], vars['signal_integrity_check'], vars['average_scl'], vars['scl_sd'], vars['scl_cv'], vars['average_scr_amplitude'], vars['sc_validity'], vars['snr'] = np.zeros(8)

    try: 
        ps_df = get_event_data(event=task,
                        df=whole_ps_df,
                        stim_df=stim_df)
        eda_df = ps_df[['EDA2', 'lsl_time_stamp', 'time']]

        eda_sampling_rate = get_sampling_rate(eda_df)
        eda_signals, info = eda_preprocess(eda_df, eda_sampling_rate)
        average_scl, scl_sd, scl_cv = scl_stability(eda_signals['EDA_Tonic'])
        average_scr_amplitude, scr_amplitude_validity = scr_amplitudes(info)
        
        print(f"Effective sampling rate: {eda_sampling_rate:.3f} Hz")
        vars['sampling_rate'] = eda_sampling_rate
        print(f"Signal Integrity Check: {eda_signal_integrity_check(eda_df):.3f} %")
        vars['signal_integrity_check'] = eda_signal_integrity_check(eda_df)
        print(f"Average Skin Conductance Level: {average_scl:.3f} mS")
        vars['average_scl'] = average_scl
        print(f"Skin Conductance Level Standard deviation: {scl_sd:.3f} mS")
        vars['scl_sd'] = scl_sd
        print(f"Skin Conductance Level Coefficient of Variation: {scl_cv:.3f} %")
        vars['scl_cv'] = scl_cv
        print(f"Average Amplitude of Skin Conductance Response: {average_scr_amplitude:.3f} mS")
        vars['average_scr_amplitude'] = average_scr_amplitude
        print(f"Skin Conductance Response Validity: {scr_amplitude_validity:.3f} %")
        vars['sc_validity'] = scr_amplitude_validity
        print(f"Signal to Noise Ratio: {eda_snr(eda_signals, eda_df, eda_sampling_rate):.3f} dB")
        vars['snr'] = eda_snr(eda_signals, eda_df, eda_sampling_rate)

        eda_slope_fig = scl_trend_analysis(eda_signals, eda_df, eda_sampling_rate, subject)
        eda_report_fig = eda_report_plot(eda_signals, info, subject)
        
        eda_error = False
        return vars, eda_slope_fig, eda_report_fig, whole_ps_df, eda_error

    except KeyError: 
        print(f'Error: No EDA data found for participant {subject}')
        vars.update({key: float('nan') for key in vars.keys()})
        eda_error = True
        eda_slope_fig=None
        eda_report_fig=None
        return vars, None, None, whole_ps_df, eda_error

#%%

if __name__ == "__main__":
    pass

# %%