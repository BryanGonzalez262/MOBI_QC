import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyxdf
from glob import glob
from scipy.signal import butter, filtfilt
import seaborn as sns
from utils import *

# clean and preprocess
def rsp_preprocess(rsp: pd.Series, sampling_rate: float) -> tuple[np.ndarray, pd.DataFrame, dict]:
    """
    Preprocesses the respiration signal using NeuroKit2, including cleaning and extracting peaks.
    Args:
        rsp (pd.Series): Respiration signal, isolated from ps_df.
        sampling_rate (float): Sampling rate of the respiration data.
    Returns:
        rsp_clean (np.ndarray): Cleaned respiration signal.
        peaks_df (pd.DataFrame): DataFrame containing peaks and troughs.
        peaks_dict (dict): Dictionary containing samples where peaks and troughs are.
    """
    # clean signal
    rsp_clean = nk.rsp_clean(rsp, sampling_rate = sampling_rate, method = 'khodadad')

    # extract peaks
    peaks_df, peaks_dict = nk.rsp_peaks(rsp_clean) # peaks_df: 1 where peaks and troughs are. dict: samples where peaks and troughs are

    return rsp_clean, peaks_df, peaks_dict

# SNR
def rsp_snr(rsp: pd.Series, rsp_clean: np.ndarray) -> float:
    """
    Calculates the Signal-to-Noise Ratio (SNR) of the respiration signal.
    Args:
        rsp (pd.Series): Respiration signal, isolated from ps_df.
        rsp_clean (np.ndarray): Cleaned respiration signal.
    Returns:
        snr (float): Signal-to-Noise Ratio in decibels (dB).
    """
    # signal power
    signal_power = np.var(rsp_clean)

    # noise power (residual noise after subtracting cleaned signal from noisy signal)
    noise_signal = rsp - rsp_clean  # residual noise
    noise_power = np.var(noise_signal)

    snr = 10 * np.log10(signal_power / noise_power)
    return snr

# breath amplitude 
def rsp_breath_amplitude(rsp_clean: np.ndarray, peaks_df: pd.DataFrame, rsp_df: pd.DataFrame, sub_id: str) -> tuple[float, float, str]:
    """
    Calculates and plots the breath amplitude of the respiration signal.
    Args:
        rsp_clean (np.ndarray): Cleaned respiration signal.
        peaks_df (pd.DataFrame): DataFrame indicating which samples contain peaks and troughs.
        rsp_df (pd.DataFrame): DataFrame containing the original respiration signal.
        sub_id (str): Subject ID for saving plots.
    Returns:
        mean (float): Mean breath amplitude.
        std (float): Standard deviation of breath amplitude.
        rang (str): Range of breath amplitude.
    """
    # subtract values of troughs and peaks to get breath amplitude
    cleaned_troughs_values = rsp_clean[peaks_df['RSP_Troughs'].to_numpy() == 1]
    cleaned_peaks_values = rsp_clean[peaks_df['RSP_Peaks'].to_numpy() == 1]
    cleaned_breath_amplitude = cleaned_peaks_values - cleaned_troughs_values

    # stats
    mean = np.mean(cleaned_breath_amplitude)
    std = np.std(cleaned_breath_amplitude)
    rang = f'{np.min(cleaned_breath_amplitude):.4f}' + ' - ' + f'{np.max(cleaned_breath_amplitude):.4f}'

    # plot
    x = rsp_df.time[peaks_df['RSP_Peaks'].to_numpy() == 1]
    y = cleaned_breath_amplitude
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    plt.figure()
    plt.plot(x, y)
    plt.plot(x, p(x), label = 'trendline')
    plt.axhline(np.mean(cleaned_breath_amplitude), color = 'yellowgreen', label = 'mean')
    plt.ylabel('Breath Amplitude (V)')
    plt.xlabel('Time (s)')
    plt.title('Breath Amplitude Across Experiment')
    plt.legend()
    plt.savefig(f'report_images/{sub_id}_rsp_breathamplitude.png')

    return mean, std, rang

# respiration rate
def rsp_rate(rsp_clean: np.ndarray, peaks_dict: dict, sampling_rate: float, sub_id: str) -> tuple[float, float, str]:
    """
    Calculates and plots the respiration rate of the respiration signal.
    Args:
        rsp_clean (np.ndarray): Cleaned respiration signal.
        peaks_dict (dict): Dictionary containing samples where peaks and troughs are.
        sampling_rate (float): Sampling rate of the respiration data.
    Returns:
        mean (float): Mean respiration rate.
        std (float): Standard deviation of respiration rate.
        rang (str): Range of respiration rate.
    """
    rsp_rate = nk.rsp_rate(rsp_clean, peaks_dict, sampling_rate=sampling_rate, method = 'xcorr')
    nk.signal_plot(rsp_rate, sampling_rate=sampling_rate, alpha = 0.6)
    plt.ylabel('Breaths Per Minute')
    plt.title('Respiration Rate Across Experiment')
    plt.savefig(f'report_images/{sub_id}_rsp_respirationrate.png') # {sub_id}_rsp_respirationrate.png

    mean = np.mean(rsp_rate)
    std = np.std(rsp_rate)
    rang = f'{np.min(rsp_rate):.4f}' + ' - ' + f'{np.max(rsp_rate):.4f}'

    return mean, std, rang

# peak to peak interval
def rsp_peak_to_peak(rsp_df: pd.DataFrame, peaks_df: pd.DataFrame, sub_id: str) -> tuple[float, float, str]:
    """
    Calculates and plots the peak-to-peak interval, or the time between each breath, of the respiration signal.
    Args:
        rsp_df (pd.DataFrame): DataFrame containing the original respiration signal.
        peaks_df (pd.DataFrame): DataFrame indicating which samples contain peaks and troughs.
    Returns:
        mean (float): Mean peak-to-peak interval.
        std (float): Standard deviation of peak-to-peak interval.
        rang (str): Range of peak-to-peak interval.
    """
    ptp_df = rsp_df[peaks_df['RSP_Peaks'].to_numpy() == 1]
    ptp_df.reset_index(drop = True, inplace = True)
    ptp_df.loc[:,'time'] = ptp_df.lsl_time_stamp - ptp_df.lsl_time_stamp[0]
    ptp = ptp_df.lsl_time_stamp.diff()

    mean = np.nanmean(ptp)
    std = np.nanstd(ptp)
    rang = f'{np.nanmin(ptp):.4f}' + ' - ' + f'{np.nanmax(ptp):.4f}'

    x = ptp_df['time'][1:]
    y = ptp[1:]
    z = np.polyfit(x, y, 3)
    p = np.poly1d(z)
    plt.figure()
    plt.plot(x, y)
    plt.plot(x, p(x), label = 'trendline')
    plt.axhline(np.nanmean(ptp), color = 'yellowgreen', label = 'mean')
    plt.ylabel('Time Between Breaths (s)')
    plt.xlabel('Time (s)')
    plt.title('Peak to Peak Interval Across Experiment')
    plt.legend()
    plt.savefig(f'report_images/{sub_id}_rsp_peaktopeak.png')

    return mean, std, rang

# baseline drift using lowpass
def rsp_lowpass_filter(rsp: pd.Series, cutoff=0.05, fs=500, order=2) -> np.ndarray:
    """
    Applies a lowpass Butterworth filter to the respiration signal to estimate baseline drift.
    Args:
        rsp (pd.Series): Respiration signal, isolated from ps_df.
        cutoff (float): Cutoff frequency of the filter.
        fs (int): Sampling frequency of the signal.
        order (int): Order of the filter.
    Returns:
        filtered_signal (np.ndarray): Filtered respiration signal.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    return filtfilt(b, a, rsp)

# autocorrelation
def rsp_autocorrelation(rsp: pd.Series, ptp_mean: float, sampling_rate: float, sub_id: str) -> float:
    """
    Calculates the autocorrelation of the respiration signal at a typical breath cycle, calculated 
    using the mean inter-breath interval (peak-to-peak interval) times the sampling rate.
    
    Args:
        rsp (pd.Series): Respiration signal, isolated from ps_df.
        ptp_mean (float): Mean peak-to-peak interval.
        sampling_rate (float): Sampling rate of the respiration data.
    Returns:
        autocorr (float): Autocorrelation at the specified lag.
    """
    lag = int(ptp_mean * sampling_rate)
    autocorr = rsp.autocorr(lag = lag)

    autocorr2 = np.correlate(rsp, rsp, mode='full')
    plt.figure() # 8, 4 before 
    plt.plot(autocorr2)
    plt.title("Autocorrelation at Every Possible Lag")
    plt.ylabel("Degree of Autocorrelation")
    plt.xlabel("Lag")
    plt.savefig(f'report_images/{sub_id}_rsp_autocorrelation.png')

    return autocorr

# final big dict 
def rsp_qc(xdf_filename:str) -> -> tuple[dict, pd.DataFrame]:
    """
    Main function to extract respiration quality control metrics.
    Args:
        xdf_filename (str): Path to the XDF file containing the respiration data.
    Returns:
        vars (dict): Dictionary containing respiration quality control metrics.
        ps_df (pd.DataFrame): DataFrame containing physiological data.
    """
    
    # load data 
    sub_id = xdf_filename.split('-')[1].split('/')[0]
    ps_df = import_physio_data(xdf_filename)

    # get rsp data
    rsp_df = ps_df[['RESPIRATION0', 'lsl_time_stamp']].rename(columns={'RESPIRATION0': 'respiration'})
    rsp_df['time'] = rsp_df['lsl_time_stamp'] - rsp_df['lsl_time_stamp'][0]
    rsp = rsp_df.respiration
    sampling_rate = get_sampling_rate(rsp_df)

    # preprocess
    rsp_clean, peaks_df, peaks_dict = rsp_preprocess(rsp, sampling_rate)

    # variables
    vars = {}
    vars['sampling_rate'] = sampling_rate
    print(f"Effective sampling rate: {sampling_rate:.4f}")

    vars['rsp_snr'] = rsp_snr(rsp, rsp_clean)
    print(f"Signal to Noise Ratio: {vars['rsp_snr']:.4f}")

    vars['breath_amplitude_mean'], vars['breath_amplitude_std'], vars['breath_amplitude_range'] = rsp_breath_amplitude(rsp_clean, peaks_df, rsp_df, sub_id)
    print(f"Breath amplitude mean: {vars['breath_amplitude_mean']:.4f}")
    print(f"Breath amplitude std: {vars['breath_amplitude_std']:.4f}")
    print(f"Breath amplitude range: {vars['breath_amplitude_range']}")

    vars['rsp_rate_mean'], vars['rsp_rate_std'], vars['rsp_rate_range'] = rsp_rate(rsp_clean, peaks_dict, sampling_rate, sub_id)
    print(f"Respiration rate mean: {vars['rsp_rate_mean']:.4f}")
    print(f"Respiration rate std: {vars['rsp_rate_std']:.4f}")
    print(f"Respiration rate range: {vars['rsp_rate_range']}")

    vars['ptp_mean'], vars['ptp_std'], vars['ptp_range'] = rsp_peak_to_peak(rsp_df, peaks_df, sub_id)
    print(f"Peak to peak interval mean: {vars['ptp_mean']:.4f}")
    print(f"Peak to peak interval std: {vars['ptp_std']:.4f}")
    print(f"Peak to peak interval range: {vars['ptp_range']}")

    lowpass = rsp_lowpass_filter(rsp)
    vars['baseline_drift'] = np.std(lowpass)
    print(f"Baseline drift: {vars['baseline_drift']:.4f}")

    vars['autocorrelation'] = rsp_autocorrelation(rsp, vars['ptp_mean'], sampling_rate, sub_id)
    print(f"Autocorrelation at typical breath cycle: {vars['autocorrelation']:.4f}")

    return vars, ps_df

# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass