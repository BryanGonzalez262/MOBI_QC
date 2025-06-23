import pyxdf
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import wave
#import pyaudio
import numpy as np
#import sounddevice as sd
from utils import *
from scipy.signal import iirnotch, filtfilt
from glob import glob

def get_seconds_between_triggers(stim_df: pd.DataFrame, trigger1: int, trigger2: int) -> float:
    """
    This function computes the duration between two event triggers in seconds.
    Args:
        stim_df (pd.DataFrame): The dataframe that includes all stimulus triggers, markers and corresponding time stamps 
        trigger1 (int): The end trigger for duration calculation
        trigger2 (int): The start trigger for duration calculation
    Returns:
        duration_between_triggers (float): The duration between given triggers in seconds  
    """   
    duration_between_triggers =  stim_df.loc[stim_df.trigger == trigger1, 'lsl_time_stamp'].values[0] - stim_df.loc[stim_df.trigger == trigger2, 'lsl_time_stamp'].values[0]
    return duration_between_triggers

def missing_markers(events: dict[int, str], stim_df: pd.DataFrame) -> list | None:
    """
    This function checks for missing markers.
    Args:
        events (dict[int, str]): The Dictionary that includes all stimulus markers adn corresponding labels
        stim_df (pd.DataFrame): The dataframe that includes all stimulus triggers, markers and corresponding time stamps    
    Returns:
        missing_markers (list): The list of missing event markers in the given xdf file   
    """    
    missing_markers=[]
    for event in list(events.values()):
        if event in stim_df.event.tolist():
            continue
        else:
            missing_markers = missing_markers + [event]
    if not missing_markers:
        return None
    else:   
        return missing_markers

def total_experiment_duration(stim_df: pd.DataFrame) -> str:
    """
    This function computes the total duration of the experiment.
    Args:
        stim_df (pd.DataFrame): The dataframe that includes all stimulus triggers, markers and corresponding time stamps    
    Returns:
        total_duration (str): The total duration of the experiment in minutes:seconds   
    """    
    minutes_entire_experiment, seconds = divmod(get_seconds_between_triggers(stim_df, 201, 200), 60) 
    total_duration = f"{int(minutes_entire_experiment):02}:{int(seconds):02}"

    return total_duration

def unexpected_durations(stim_df: pd.DataFrame, story_onsets: list, events: dict, audiofiles: list) -> list | None:
    """
    This function checks whether story listening task durations are of expected length.
    Args:
        stim_df (pd.DataFrame): The dataframe that includes all stimulus triggers, markers and corresponding time stamps.
        story_onsets (list): The list of story listening event triggers.
        events (dict): The dictionary that contains all event triggeres and their labels.
        audiofiles (list): The list of paths to all story listenign audiofiles.    
    Returns:
        list_of_task_duration_difference (list): The story listening tasks with durations not within expected length. 
    """    
    durations = pd.DataFrame({
    'trigger':story_onsets,
    'story':[events[x] for x in story_onsets],
    'lsl_duration': [get_seconds_between_triggers(stim_df, x+1, x) for x in story_onsets],
    'audiofile_duration': [wave.open(x).getnframes()/wave.open(x).getframerate() for x in audiofiles] #duration of audio file is number of frames divided by the frame rate.
    })

    durations['difference(sec)'] = durations['audiofile_duration'] - durations['lsl_duration']
    
    task_duration_difference = []

    # Calculating audiofile duration in 48kHz and then comparing with story listening durations from stim_df
    for i in range(len(durations.audiofile_duration)):
        task_duration = (durations.audiofile_duration[i] * 44100) / 48000
        if (durations.lsl_duration[i].round(3) -  task_duration.round(3)) > 0.5:
            task_duration_difference = task_duration_difference + [durations.story[i]]

    if task_duration_difference != []:
        list_task_duration_difference = task_duration_difference
    else:
        list_task_duration_difference = None

    return list_task_duration_difference

def resting_state_social_script_durations(stim_df: pd.DataFrame, trigger: int) -> bool:
    """
    This function checks whether resting state and social script task durations are of expected length.
    Args:
        stim_df (pd.DataFrame): The dataframe that includes all stimulus triggers, markers and corresponding time stamps.
        trigger (int): The event trigger for resting state or social script task    
    Returns:
        True if durations of restign state and social script task are within expected range, otherwise False
    """    
    trial_duration = get_seconds_between_triggers(stim_df, trigger+1, trigger)
    print(trial_duration)
    if trial_duration <= 305.0  and trial_duration >= 298.0:
        return True
    else:
        return False
    
def impedance_check_duration(stim_df: pd.DataFrame) -> str:
    """
    This function computes the duration of impedance check for the EEG system.
    Args:
        stim_df (pd.DataFrame): The dataframe that includes all stimulus triggers, markers and corresponding time stamps.    
    Returns:
        impedance_duration (str): The duration of impedance check in minutes:seconds 
    """    
    impedance_check_mins, impedance_check_seconds = divmod(get_seconds_between_triggers(stim_df, 401, 400), 60)
    impedance_duration = f"{int(impedance_check_mins):02}:{int(impedance_check_seconds):02}"

    return impedance_duration

def ten_seconds_rest(stim_df: pd.DataFrame) -> bool:
    """
    This function checks whether all 10-second rest periods have equal durations.
    Args:
        stim_df (pd.DataFrame): The dataframe that includes all stimulus triggers, markers, and corresponding time stamps.
    Returns:
        equal_rest_durations (bool): True if all 10-second rest durations are equal, otherwise False.
    """    
    # Count the number of occurrences of trigger value 100 using sum
    evs = stim_df.loc[stim_df.event != 'psychopy_time_stamp']
    trigger_count = (evs['trigger'] == 100).sum()

    #rest_onsets = 
    ten_secs_rest_durations = pd.DataFrame({
        'trigger':[x for x in range(trigger_count)],
        'story': ['Onset_10second_rest' for x in range(trigger_count)],
        'lsl_duration': [get_seconds_between_triggers(stim_df, x+1, x) for x in evs['trigger'] if x == 100]})
    #print(ten_secs_rest_durations)

    # Check if rest durations are equal
    equal_rest_durations = all(x == ten_secs_rest_durations['lsl_duration'][0] for x in ten_secs_rest_durations['lsl_duration'])

    return equal_rest_durations

def average_question_response_time(stim_df: pd.DataFrame) -> str:
    """
    This function computes the average response time for all story listening tasks.
    Args:
        stim_df (pd.DataFrame): The dataframe that includes all stimulus triggers, markers, and corresponding time stamps.
    Returns:
        average_response_time (str): The average response time across all story listening tasks in minutes:seconds.
    """
    response_times = []
    trial_response_times = []
    trigger_idx = 0
    for idx, x in enumerate(stim_df['trigger']):
        if idx in stim_df.loc[stim_df.trigger == 300,'lsl_time_stamp'].index:
            response_times = response_times + [stim_df.loc[stim_df.trigger == 301, 'lsl_time_stamp'].values[trigger_idx] - stim_df.loc[stim_df.trigger == 300, 'lsl_time_stamp'].values[trigger_idx]]
            trigger_idx = trigger_idx + 1
        elif x == 100 or x == 400:
            trial_response_times = trial_response_times + [sum(response_times)]
            response_times = []
    average_response_mins, average_response_seconds = divmod(sum(trial_response_times)/len(trial_response_times), 60)     
    average_response_time = f"{int(average_response_mins):02}:{int(average_response_seconds):02}"

    return average_response_time

def behavior_qc(xdf_filename) -> dict[str,int]:
    """
    This function computes the Behavior quality control pipeline for the given xdf file.
    Args:
        xdf_filename (str): The path to the xdf file.
    Returns:
        vars (dict): Contains all quality control measures for behavior data of the give xdf file   
    """    
    events = {
        200: 'Onset_Experiment',
        10: 'Onset_RestingState',
        11: 'Offset_RestingState',
        500: 'Onset_StoryListening',
        501: 'Offset_StoryListening',
        100: 'Onset_10second_rest',
        101: 'Offset_10second_rest', 
        20: 'Onset_CampFriend',
        21: 'Offset_CampFriend',
        30: 'Onset_FrogDissection',
        31: 'Offset_FrogDissection',
        40: 'Onset_DanceContest',
        41: 'Offset_DanceContest',
        50: 'Onset_ZoomClass',
        51: 'Offset_ZoomClass',
        60: 'Onset_Tornado',
        61: 'Offset_Tornado',
        70: 'Onset_BirthdayParty',
        71: 'Offset_BirthdayParty',
        300: 'Onset_subjectInput',
        301: 'Offset_subjectInput',
        302: 'Onset_FavoriteStory',
        303: 'Offset_FavoriteStory',
        304: 'Onset_WorstStory',
        305: 'Offset_WorstStory',
        400: 'Onset_impedanceCheck',
        401: 'Offset_impedanceCheck',
        80: 'Onset_SocialTask',
        81: 'Offset_SocialTask',
        201: 'Offset_Experiment',
    }

    story_onsets = [20, 30, 40, 50, 60, 70]

    audiofiles= [
        "/Users/apurva.gokhe/Documents/CUNY_QC/NEW_AUDIO_44/Camp_Lose_A_Friend.wav",
        "/Users/apurva.gokhe/Documents/CUNY_QC/NEW_AUDIO_44/Frog_Dissection_Disaster.wav",
        "/Users/apurva.gokhe/Documents/CUNY_QC/NEW_AUDIO_44/I_Decided_To_Be_Myself_And_Won_A_Dance_Contest.wav",
        "/Users/apurva.gokhe/Documents/CUNY_QC/NEW_AUDIO_44/I_Fully_Embarrassed_Myself_In_Zoom_Class1.wav",
        "/Users/apurva.gokhe/Documents/CUNY_QC/NEW_AUDIO_44/Left_Home_Alone_in_a_Tornado.wav",
        "/Users/apurva.gokhe/Documents/CUNY_QC/NEW_AUDIO_44/The_Birthday_Party_Prank_44.wav",
    ]


        #xdf_filename = '/Users/apurva.gokhe/Documents/CUNY_QC/data/sub-P5029423/sub-P5029423_ses-S001_task-CUNY_run-001_mobi.xdf'
    stim_df = import_stim_data(xdf_filename)
    
    vars = {}
    print(f"Missing stimulus markers: {missing_markers(events, stim_df)}")
    vars['missing_stimulus_markers'] = missing_markers(events, stim_df)
    print(f"Duration of experiment: {total_experiment_duration(stim_df)}")
    vars['total_duration'] = total_experiment_duration(stim_df)
    print(f"Durations do not match expected length: {unexpected_durations(stim_df, story_onsets, events, audiofiles)}")
    vars['unexpected_durations'] = unexpected_durations(stim_df, story_onsets, events, audiofiles)
    print(f"Is Resting state of expected duration? {resting_state_social_script_durations(stim_df, 10)}") # immediate qc measure, not included in the report
    print(f"Is Social script of expected duration? {resting_state_social_script_durations(stim_df, 80)}") # immediate qc measure, not included in the report
    print(f"Duration of Impedance check: {impedance_check_duration(stim_df)}")
    vars['impedance_check_duration'] = impedance_check_duration(stim_df)
    print(f"Are all 10 seconds rest equal? {ten_seconds_rest(stim_df)}")
    vars['ten_seconds_rest'] = ten_seconds_rest(stim_df)
    print(f"Average response time across all story listening tasks: {average_question_response_time(stim_df)}")
    vars['average_response_time'] = average_question_response_time(stim_df)

    return vars
#%% 

# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass
# %%
