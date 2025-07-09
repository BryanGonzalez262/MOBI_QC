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
from neurokit2.signal import signal_power

def ecg_preprocess(ecg_df: pd.DataFrame, ecg_sampling_rate: float) -> tuple[pd.DataFrame,dict]:
    """
    Preprocesses ECG data using NeuroKit2.
    Args:
        ecg_df (pd.DataFrame): DataFrame containing ECG data.
        ecg_sampling_rate (float): Sampling rate of the ECG data.
    Returns:
        ecg_signals (pd.DataFrame): Processed ECG signals.
        info (dict): Additional information about the ECG processing.
    """
    ecg_signals, info = nk.ecg_process(ecg_df['ECG1'], sampling_rate=ecg_sampling_rate, method='neurokit')
    return ecg_signals, info

def average_heartrate(ecg_signals: pd.DataFrame) -> float:
    """
    Calculates the average heart rate from processed ECG signals.
    Args:
        ecg_signals (pd.DataFrame): Processed ECG signals.
    Returns:
        avg_heartrate (float): Average heart rate in beats per minute.
    """
    avg_heartrate = sum(ecg_signals['ECG_Rate'])/len(ecg_signals['ECG_Rate'])
    return avg_heartrate

def ecg_quality_kurtosis_SQI(ecg_cleaned, method="fisher") -> float:
    """
    Computes the kurtosis of the cleaned ECG signal for quality assessment.
    Args:
        ecg_cleaned (pd.Series): Cleaned ECG signal.
        method (str): Method for kurtosis calculation ("fisher" or "pearson").
    Returns:
        kurtosis (float): Kurtosis value of the ECG signal.
        Return the kurtosis of the signal, with Fisher's or Pearson's method.
    """
    if method == "fisher":
        kurtosis = float(scipy.stats.kurtosis(ecg_cleaned, fisher=True))
    elif method == "pearson":
        kurtosis = float(scipy.stats.kurtosis(ecg_cleaned, fisher=False))
    return kurtosis

def ecg_quality_psd_SQI(
    ecg_cleaned: pd.Series,
    sampling_rate: float,
    window=1024,
    num_spectrum=[5, 15],
    dem_spectrum=[5, 40],
    **kwargs
) -> float:
    """
    Computes the Power Spectrum Distribution (PSD) of the QRS wave.
    Args:
        ecg_cleaned (pd.Series): Cleaned ECG signal.
        sampling_rate (float): Sampling rate of the ECG data.
        window (int): Window size for Welch's method.
        num_spectrum (list): Frequency band for numerator.
        dem_spectrum (list): Frequency band for denominator.
    Returns:
        power_spectrum_distribution (float): PSD ratio for quality assessment.
    """
    psd = signal_power(
        ecg_cleaned,
        sampling_rate=sampling_rate,
        frequency_band=[num_spectrum, dem_spectrum],
        method="welch",
        normalize=False,
        window=window,
        **kwargs
    )

    num_power = psd.iloc[0, 0]
    dem_power = psd.iloc[0, 1]
    power_spectrum_distribution = float(num_power / dem_power) 

    return power_spectrum_distribution

def ecg_quality_baseline_power_SQI(
    ecg_cleaned: pd.Series,
    sampling_rate: float,
    window=1024,
    num_spectrum=[0, 1],
    dem_spectrum=[0, 40],
    **kwargs
) -> float:
    """
    Computes the relative power in the baseline of the ECG signal.
    Args:
        ecg_cleaned (pd.Series): Cleaned ECG signal.
        sampling_rate (float): Sampling rate of the ECG data.
        window (int): Window size for Welch's method.
        num_spectrum (list): Frequency band for numerator.
        dem_spectrum (list): Frequency band for denominator.
    Returns:
        relative_baseline_power (float): Relative baseline power for quality assessment.
    """
    psd = signal_power(
        ecg_cleaned,
        sampling_rate=sampling_rate,
        frequency_band=[num_spectrum, dem_spectrum],
        method="welch",
        normalize=False,
        window=window,
        **kwargs
    )

    num_power = psd.iloc[0, 0]
    dem_power = psd.iloc[0, 1]
    relative_baseline_power = float((1 - num_power) / dem_power)
    return relative_baseline_power

def ecg_snr(ecg_signals: pd.DataFrame, ecg_sampling_rate: float) -> float:
    """
    Calculates the Signal-to-Noise Ratio (SNR) of the ECG signal.
    Args:
        ecg_signals (pd.DataFrame): Processed ECG signals.
    Returns:
        snr (float): Signal-to-Noise Ratio in decibels (dB).
    """
    duration = len(ecg_signals['ECG_Raw'].tolist()) / ecg_sampling_rate
    t = np.linspace(0, duration , len(ecg_signals['ECG_Raw']))
    t = t[:5000]
    ecg_cleaned = ecg_signals['ECG_Clean'] 
    signal_power = np.var(ecg_cleaned)

    noise_signal = ecg_signals['ECG_Raw'] - ecg_cleaned  # residual noise
    noise_power = np.var(noise_signal)
    # Calculate SNR (Signal-to-Noise Ratio in dB)
    snr = float(10 * np.log10(signal_power / noise_power))

    return snr

def ecg_report_plot(ecg_signals:pd.DataFrame, info: dict, subject:str) -> plt:
    """
    Generates and saves a report plot for the ECG signals.
    Args:
        ecg_signals (pd.DataFrame): Processed ECG signals.
        info (dict): Additional information about the ECG processing.
        subject (str): Subject identifier for saving the plot.
    Returns:
        plt (matplotlib.pyplot): The generated plot.
    """
    nk.ecg_plot(ecg_signals, info)
    fig = plt.gcf()
    axes = fig.get_axes()
    fig.set_size_inches(16, 6) # 20,10
    plt.tight_layout()

    # Iterate over each axis and move the legend
    for i, ax in enumerate(axes):
        if i == 0: 
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=15)  # Move legend outside the plot
        elif i == 1:  
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=15)  # Move legend outside the plot
        elif i == 2:  
            ax.legend(loc='upper right', bbox_to_anchor=(1, 1), fontsize=15)  # Move legend to the right
        else:
            ax.legend(loc='center right', bbox_to_anchor=(1, 0.5), fontsize=15)  # Default position for other plots
    for ax in axes:
        ax.grid(True)

    # Save plot
    plt.savefig(f'report_images/{subject}_ecg_report.png')
    plt.show()

    return plt

def ecg_qc(xdf_filename:str, stim_df:pd.DataFrame, task='RestingState') -> tuple[dict, plt, pd.DataFrame, bool]:
    """
    Performs quality control on ECG data from an XDF file.
    Args:
        xdf_filename (str): Path to the XDF file.
    Returns:
        vars (dict): Quality control metrics for the ECG data.
        fig (matplotlib.pyplot): Generated ECG report plot.
        ecg_error (bool): Indicates whether there was an error loading ECG data. 
    """
    subject = xdf_filename.split('sub-')[1].split('/')[0]
    whole_ps_df = import_physio_data(xdf_filename)
    vars = {}
    vars['sampling_rate'], vars['average_heart_rate'], vars['kurtosis_SQI'], vars['power_spectrum_distribution_SQI'], vars['relative_baseline_power_sqi'], vars['SNR'] = np.zeros(6)  

    try:
        ps_df = get_event_data(event=task,
                        df=whole_ps_df,
                        stim_df=stim_df)
        ecg_df = ps_df[['ECG1', 'lsl_time_stamp', 'time']]

        ecg_sampling_rate = get_sampling_rate(ecg_df)  
        ecg_signals, info = ecg_preprocess(ecg_df, ecg_sampling_rate)
        ecg_cleaned = ecg_signals['ECG_Clean']

        print(f"Effective sampling rate: {ecg_sampling_rate}")
        vars['sampling_rate'] = ecg_sampling_rate
        print(f"Average heart rate: {average_heartrate(ecg_signals)}")
        vars['average_heart_rate'] = average_heartrate(ecg_signals)
        print(f"Kurtosis signal quality index: {ecg_quality_kurtosis_SQI(ecg_cleaned)}")
        vars['kurtosis_SQI'] = ecg_quality_kurtosis_SQI(ecg_cleaned)
        print(f"Power spectrum distribution signal quality index: {ecg_quality_psd_SQI(ecg_cleaned, ecg_sampling_rate)}")
        vars['power_spectrum_distribution_SQI'] = ecg_quality_psd_SQI(ecg_cleaned, ecg_sampling_rate)
        print(f"Relative power in baseline signal quality index: {ecg_quality_baseline_power_SQI(ecg_cleaned, ecg_sampling_rate)}")
        vars['relative_baseline_power_sqi'] = ecg_quality_baseline_power_SQI(ecg_cleaned, ecg_sampling_rate)
        print(f"Signal to Noise Ratio: {ecg_snr(ecg_signals, ecg_sampling_rate)}")
        vars['SNR'] = ecg_snr(ecg_signals, ecg_sampling_rate)

        fig = ecg_report_plot(ecg_signals, info, subject)

        ecg_error = False
        return vars, fig, whole_ps_df, ecg_error 

    except KeyError:
        print(f'Error: No ECG data found for participant {subject} in {xdf_filename}.')
        vars.update({key: float('nan') for key in vars.keys()})
        ecg_error = True
        return vars, None, whole_ps_df, ecg_error

#%% 
# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass

# %%
# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass
