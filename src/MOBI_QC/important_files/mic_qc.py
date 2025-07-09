import pyxdf
import pandas as pd
import numpy as np
from glob import glob
import librosa
import datetime
import matplotlib.pyplot as plt
from utils import *

def mic_lsl_wav_durations(xdf_filename:str, mic_df: pd.DataFrame) -> float:
    """
    Compare the duration of the .wav file and the lsl timestamps.
    Args:
        xdf_filename (str): path to the xdf file
        mic_df (pd.DataFrame): dataframe containing the microphone data
    Returns:
        dur_diff (float): difference between the duration of the wav file and the lsl timestamps
    """
    # get path of wav file in same folder 
    wav_path = glob(os.path.join(os.path.dirname(xdf_filename), '*.wav'))[0]
    
    # calculate wav and lsl durations
    wav_dur = round(librosa.get_duration(path=wav_path), 3)
    lsl_dur = round(mic_df['lsl_time_stamp'].iloc[-1]- mic_df['lsl_time_stamp'].iloc[0], 3)

    # diff
    dur_diff = abs(wav_dur - lsl_dur)

    return dur_diff

def mic_nans(mic_df: pd.DataFrame) -> tuple[int, float]:
    """
    Calculate the number and percentage of NaN values in the microphone data.
    Args:
        mic_df (pd.DataFrame): dataframe containing the microphone data
    Returns:
        num_NaN (int): number of NaN values in mic_df
        percent_NaN (float): percentage of NaN values in mic_df
    """
    num_NaN = mic_df['int_array'].isna().sum()
    percent_NaN = num_NaN/len(mic_df)

    return num_NaN, percent_NaN

def mic_samples_stats(mic_df: pd.DataFrame) -> tuple[float, float, float, int, int]:
    """
    Calculate stats on the microphone data.
    Args:
        mic_df (pd.DataFrame): dataframe containing the microphone data
    Returns:
        quan25 (float): first quartile of mic samples
        quan75 (float): third quartile of mic samples
        std (float): standard deviation of mic samples
        minn (int): min of mic samples
        maxx (int): max of mic samples
    """
    quan25 = np.quantile(mic_df['int_array'], 0.25)
    quan75 = np.quantile(mic_df['int_array'], 0.75) 
    std = mic_df['int_array'].std()
    minn = min(mic_df.int_array)
    maxx = max(mic_df.int_array)

    return quan25, quan75, std, minn, maxx

def mic_plots(mic_df: pd.DataFrame, stim_df: pd.DataFrame, sub_id: str):
    """
    Create plots for the microphone data.
    Args:
        mic_df (pd.DataFrame): dataframe containing the microphone data
        stim_df (pd.DataFrame): dataframe containing stimulus markers
    """
    #hist
    plt.figure()
    plt.hist(mic_df['int_array'], bins=100)
    plt.xlabel('Microphone Samples')
    plt.ylabel('Count')
    plt.title('Distribution of Microphone Samples')
    plt.savefig(f'report_images/{sub_id}_mic_histogram.png')

    #line plot
    plt.figure(figsize=(10, 3))
    plt.plot(mic_df.lsl_time_stamp, mic_df.int_array)
    for event in stim_df.loc[stim_df.event.str.contains('StoryListening|SocialTask')].iterrows():
        plt.axvline(event[1]['lsl_time_stamp'], color='r')
        plt.text(event[1]['lsl_time_stamp']+4, 0, event[1]['event'], rotation=90, verticalalignment='center', fontweight = 'bold')

    plt.xlabel('LSL Timestamps')
    plt.ylabel('Samples')
    plt.title('Microphone Samples over Time')
    plt.tight_layout()
    plt.savefig(f'report_images/{sub_id}_mic_lineplot.png')

def mic_qc(xdf_filename:str, stim_df:pd.DataFrame, task = 'Experiment') -> tuple[dict, pd.DataFrame, bool]:
    """
    Main function to extract microphone quality control metrics.
    Args:
        xdf_filename (str): Path to the XDF file containing the microphone data.
        stim_df (pd.DataFrame): dataframe containing stimulus markers.
        task (str): arm of the experiment for which user wants quality control performed. Must be one of "Experiment", 
            "RestingState", "StoryListening", "SocialTask".

    Returns:
        vars (dict): Dictionary containing the quality control metrics.
        mic_df (pd.DataFrame): DataFrame containing microphone data.
        mic_error (bool): Whether there was an error loading mic data.
    """
    # load data
    sub_id = xdf_filename.split('sub-')[1].split('/')[0]

    # initialize mic_vars keys as zeros
    vars = {}
    vars['sampling_rate'], vars['lsl_wav_duration_diff'], vars['num_NaN'], vars['percent_NaN'], vars['quan25'], vars['quan75'], vars['std'], vars['min'], vars['max'] = np.zeros(9)

    try:
        # get data
        whole_mic_df = import_mic_data(xdf_filename)
        mic_df = get_event_data(event = task, df = whole_mic_df, stim_df = stim_df)

        # returns zeros in vars and data for entire exp in df if restingstate
        if task == 'RestingState':
            return vars, whole_mic_df, mic_error
        
        # get metrics
        sampling_rate = get_sampling_rate(mic_df)
        vars['sampling_rate'] = sampling_rate
        print(f"Effective sampling rate: {sampling_rate:.4f}")

        vars['lsl_wav_duration_diff'] = mic_lsl_wav_durations(xdf_filename, mic_df)
        print(f"Difference between .wav file and lsl timestamps durations: {vars['lsl_wav_duration_diff']:.4f}")

        vars['num_NaN'], vars['percent_NaN'] = mic_nans(mic_df)
        print(f"number of NaN's: {vars['num_NaN']} \npercent of NaN's: {vars['percent_NaN']:.4%}")
        vars['quan25'], vars['quan75'], vars['std'], vars['min'], vars['max'] = mic_samples_stats(mic_df)
        print('mic samples first quartile: {} \nmic samples third quartile: {}'.format(vars['quan25'], vars['quan75']))
        print('mic samples standard deviation: {:.4f}'.format(vars['std']))
        print(f"mic samples min: {vars['min']} \nmic samples max: {vars['max']}")
        
        # plots
        mic_plots(mic_df, stim_df, sub_id)
        mic_error = False


        return vars, whole_mic_df, mic_error

    except IndexError:
        
        whole_mic_df = pd.DataFrame()
        vars.update({key: float('nan') for key in vars.keys()})
        mic_error = True
        print(f'Error: No mic data found for participant {sub_id} in {xdf_filename}')
        return vars, whole_mic_df, mic_error

# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass