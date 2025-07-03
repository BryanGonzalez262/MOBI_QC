import pandas as pd
import pyxdf
import tarfile
from io import BytesIO
import os

import numpy as np
import sounddevice as sd
from glob import glob
from tqdm import tqdm
import datetime
from stim_correction import trigger_recovery



def get_collection_date(xdf_filename:str):
    """Get the collection date from the xdf file.
    
    Args:
        xdf_filename (str): The xdf file to get the collection date from.
    Returns:(str): the date and time of the first psychopy timestamp.
    """
    stim_df = import_stim_data(xdf_filename)
    stim_df.loc[stim_df.event == "psychopy_time_stamp", "trigger"].to_list()[0]
    return datetime.datetime.fromtimestamp(stim_df.loc[stim_df.event == "psychopy_time_stamp", "trigger"].to_list()[0]).strftime('%Y-%m-%d %H:%M:%S')


def import_webcam_data(xdf_filename:str):    
    cam_data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'name': 'WebcamStream'}], verbose=False)
    frame_nums = [int(i[0]) for i in cam_data[0]['time_series']]
    time_pre = [float(i[1]) for i in cam_data[0]['time_series']]
    time_evnt_ms = [float(i[2]) for i in cam_data[0]['time_series']]
    time_post = [float(i[3]) for i in cam_data[0]['time_series']]

    cam_df = pd.DataFrame({'frame_num': frame_nums, 
                        'time_pre': time_pre, 
                        'cap_time_ms': time_evnt_ms,
                        'time_post': time_post,
                        'lsl_time_stamp': cam_data[0]['time_stamps']})

    cam_df['frame_time_sec'] = (cam_df.cap_time_ms - cam_df.cap_time_ms[0])/1000
    cam_df['lsl_time_sec'] = (cam_df.lsl_time_stamp - cam_df.lsl_time_stamp[0]) *1000
    return cam_df


def import_physio_data(xdf_filename:str):
    data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'name': 'OpenSignals'}], verbose = False)
    column_labels = [data[0]['info']['desc'][0]['channels'][0]['channel'][i]['label'][0] for i in range(len(data[0]['info']['desc'][0]['channels'][0]['channel']))]
    df = pd.DataFrame(data[0]['time_series'], columns=column_labels)
    df['lsl_time_stamp'] = data[0]['time_stamps']
    df['time'] = df.lsl_time_stamp - df.lsl_time_stamp[0]
    return df

def import_mic_data(xdf_filename:str):
    data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'type': 'AudioCapture'}], verbose = False)
    df = pd.DataFrame(data[0]['time_series'], columns=['int_array'])
    df['bytestring'] = df['int_array'].apply(lambda x: np.array(x).tobytes())
    df['duration'] = (data[0]['time_stamps'] - data[0]['time_stamps'][0])/data[0]['info']['effective_srate']
    df['lsl_time_stamp'] = data[0]['time_stamps']
    df['time'] = df.lsl_time_stamp - df.lsl_time_stamp[0]
    return df

def import_video_data(xdf_filename:str):
    data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'type': 'video'}], verbose = False)
    frame_nums = [int(i[0]) for i in data[0]['time_series']]
    time_pre = [float(i[1]) for i in data[0]['time_series']]
    time_evnt_ms = [float(i[2]) for i in data[0]['time_series']]
    time_post = [float(i[3]) for i in data[0]['time_series']]
    df = pd.DataFrame({'frame_num': frame_nums, 
                        'time_pre': time_pre, 
                        'cap_time_ms': time_evnt_ms,
                        'time_post': time_post,
                        'lsl_time_stamp': data[0]['time_stamps']})

    df['frame_time_sec'] = (df.cap_time_ms - df.cap_time_ms[0])/1000
    df['time'] = df.lsl_time_stamp - df.lsl_time_stamp[0]
    return df

def import_et_data(xdf_filename:str):
    data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'type': 'ET'}], verbose = False)
    column_labels = [data[0]['info']['desc'][0]['channels'][0]['channel'][i]['label'][0] for i in range(len(data[0]['info']['desc'][0]['channels'][0]['channel']))]
    df = pd.DataFrame(data[0]['time_series'], columns=column_labels)
    df['lsl_time_stamp'] = data[0]['time_stamps']
    df['time'] = df.lsl_time_stamp - df.lsl_time_stamp[0]
    df['diff'] = df.lsl_time_stamp.diff()
    return df

def import_eeg_data(xdf_filename:str):
    data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'type': 'EEG'}], verbose = False)
    ch_names = [f"E{i+1}" for i in range(data[0]['time_series'].shape[1])]
    df = pd.DataFrame(data[0]['time_series'], columns=ch_names) # index=data[0]['time_stamps']
    df['lsl_time_stamp'] = data[0]['time_stamps']
    #df['time'] = df.lsl_time_stamp - df.lsl_time_stamp[0]
    return df

def import_stim_data(xdf_filename:str):
    '''
    Get the stimuli dataframe from the xdf file.
    
    Args:
        xdf_filename (str): The xdf file to get the stimuli from.
    '''
    data, _ = pyxdf.load_xdf(xdf_filename, select_streams=[{'name':'Stimuli_Markers'}], verbose = False)
    stim_df = pd.DataFrame(data[0]['time_series'])
    stim_df.rename(columns={0: 'trigger'}, inplace=True)

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

    # relabel the event if the trigger is in the events dictionary, else if 
    stim_df['event'] = stim_df['trigger'].apply(lambda x: events[x] if x in events.keys() else 'Bx_input')

    # relabel the event as a psychopy timestamp if the trigger is greater than 5 digits
    stim_df.loc[stim_df.trigger.astype(str).str.len() > 5, 'event'] = 'psychopy_time_stamp'
    stim_df['lsl_time_stamp'] = data[0]['time_stamps']
    stim_df['time'] = (data[0]['time_stamps'] - data[0]['time_stamps'][0])

    dt = datetime.datetime.fromtimestamp(stim_df.loc[stim_df.event == "psychopy_time_stamp", "trigger"].to_list()[0])#.strftime('%Y-%m-%d %H:%M:%S')
    # check if date after 03/25/2025
    
    if (dt > datetime.datetime(2025, 3, 25)) & (dt < datetime.datetime(2025, 5, 23)):
        print('trigger recovery')
        stim_df = trigger_recovery(stim_df, xdf_filename)
    
    return stim_df

def get_event_data(event, df, stim_df):
    """
    Get the data from a given data modality that corresponds to the event in the stimuli dataframe.
    
    Args:
        event (str): The event to get the data for.
        df (pd.DataFrame): The dataframe containing the timeseries along with a column for lsl timestamps.
        stim_df (pd.DataFrame): The stimuli dataframe containing the eventa mapped to lsl timestamps.
    
    Returns:
        pd.DataFrame: The  data corresponding to the event.
        """
    new_df = df.loc[(df['lsl_time_stamp'] >= stim_df.loc[stim_df['event'] == 'Onset_'+event, 'lsl_time_stamp'].values[0]) & 
                  (df['lsl_time_stamp'] <= stim_df.loc[stim_df['event'] == 'Offset_'+event, 'lsl_time_stamp'].values[0])].copy().reset_index(drop = True)
    return new_df

# get durations of certain experiment arm
def get_durations(xdf_path: str, 
    task: str, 
    stim_df: pd.DataFrame
    df_map: dict, 
    error_map: dict
    ) -> dict:
    
    """
    Get the durations of each data stream and compare to their expected duration, given an experiment arm, where the expected duration is calculated from the LSL timestamps of the stimulus markers.
    
    Args:
        xdf_path (str): The path to the xdf file.
        task (str): The part of the experiment to view durations. Can be one of "Experiment", 
            "RestingState", "StoryListening", "SocialTask", or any one of the stories ('BirthdayParty', 
            'ZoomClass', 'Tornado', 'FrogDissection', 'DanceContest', 'CampFriend')
        stim_df (pd.DataFrame): The stimuli dataframe containing the events mapped to lsl timestamps.
        df_map (dict): Contains dataframes for each data modality, loaded through import_modality_data functions in utils.
        error_map (dict): Contains booleans for each data modality indicating error. 
    
    Returns:
        pd.DataFrame: The durations of each stream in seconds and mm:ss and the percent that that duration 
            comprised of the length of that experiment arm.
    """    
    streams = list(df_map.keys())

    # find expected duration (stim lsl_time_stamp length of experiment part)
    exp_start = stim_df.loc[stim_df.event == 'Onset_'+task, 'lsl_time_stamp'].values[0]
    exp_end = stim_df.loc[stim_df.event == 'Offset_'+task, 'lsl_time_stamp'].values[0]
    exp_dur = round(exp_end - exp_start, 4)

    # expected mm:ss
    exp_dt = datetime.timedelta(seconds=exp_dur)
    exp_dt_dur = str(datetime.timedelta(seconds=round(exp_dt.total_seconds())))

    # make + populate durations_df
    durations_df = pd.DataFrame(columns = ['stream', 'duration', 'mm:ss', 'percent'])
    for i, stream in enumerate(streams):
        # don't include mic in resting state
        if task == 'RestingState' and stream == 'mic':
            continue
        if error_map[stream]: 
            subject = xdf_path.split('/')[6].split('-')[1]
            print(f'No {stream} data for participant {subject}')
            continue
        # grab data for stream + experiment part
        event_data = get_event_data(task, df_map[stream], stim_df)

        # print if no data
        if event_data.empty:
            durations_df.loc[i] = [stream, 0, str(datetime.timedelta(seconds=0)), '0.00%']
            print(f'{stream} has no {task} data') 
            continue
        # calculate duration
        start = event_data['lsl_time_stamp'].values[0]
        stop = event_data['lsl_time_stamp'].values[-1]
        dur = round(stop - start, 4)

        # calculate hh:mm:ss
        dt = datetime.timedelta(seconds=dur)
        dt_dur = str(datetime.timedelta(seconds=round(dt.total_seconds())))

        # calculate percent 
        percent = f'{dur/exp_dur:.4%}'

        durations_df.loc[i] = [stream, dur, dt_dur, percent]

    # print which are short
    for i in durations_df.iterrows():
        if i[1]['duration'] == 0:
            continue
        if i[1]['duration'] < (exp_dur - 5): # 5 second margin
            print(f"{i[1]['stream']} is shorter than expected for {task} by {exp_dur - i[1]['duration']:.4f} seconds")
    
    # print + return durations_df
    durations_df.loc[durations_df.index.max() + 1] = ['expected', exp_dur, exp_dt_dur, '100.0000%']
    durations_df.sort_values(by='duration', inplace=True)
    print('\n' + task + ' DataFrame')
    return durations_df

def load_xdf_from_zip(path_to_zip):  
    # Path to the tar.gz file
    tar_gz_file_path = path_to_zip # Path to the tar.gz file

    # Open the tar.gz file
    with tarfile.open(tar_gz_file_path, 'r:gz') as tar:
        file_list = tar.getnames() # List all files in the tar.gz
        file_name = [x for x in file_list if os.path.splitext(x)[1] == '.xdf'][0] # Read a specific file from the tar.gz
        file = tar.extractfile(file_name)
        file_content = file.read()
        data, info = pyxdf.load_xdf(BytesIO(file_content))
        #streams_collected = [stream['info']['name'][0] for stream in data]        
        #print(streams_collected)
    return data, info

def whole_durations(xdf_path):
    """
    Get the durations of each data stream and compare to their expected duration, for the entire experiment, where the expected duration is 
    the max duration of any data stream.
    Args:
        xdf_path (str): The path to the xdf file.

    Returns:
        pd.DataFrame: The durations of each stream in seconds and mm:ss and the percent that that duration comprised 
        of the max duration of all data streams. 
    """
    # import all data modalities
    et_df = import_et_data(xdf_path)
    stim_df = import_stim_data(xdf_path)
    eeg_df = import_eeg_data(xdf_path)
    mic_df = import_mic_data(xdf_path)
    cam_df = import_video_data(xdf_path)
    ps_df = import_physio_data(xdf_path)

    df_map = {
            'et': et_df,
            'ps': ps_df,
            'mic': mic_df,
            'cam': cam_df,
            'eeg': eeg_df
        }

    streams = list(df_map.keys())

    whole_durations_df = pd.DataFrame(columns = ['stream', 'duration', 'mm:ss'])
  
    # populate whole_durations_df
    for i, stream in enumerate(streams):  
        duration = df_map[stream]['lsl_time_stamp'].iloc[-1]- df_map[stream]['lsl_time_stamp'].iloc[0]
        duration = round(duration, 4)
        # convert to mm:ss
        whole_dt = datetime.timedelta(seconds=duration)
        whole_dt_dur = str(datetime.timedelta(seconds=round(whole_dt.total_seconds())))
        whole_durations_df.loc[i] = [stream, duration, whole_dt_dur]
    
    whole_durations_df.sort_values(by = 'duration', inplace = True)

    # percent
    max_dur = whole_durations_df.duration.max()
    whole_durations_df['percent'] = whole_durations_df['duration'].apply(lambda x: f"{x / max_dur:.4%}")

    # print which are short
    for i in whole_durations_df.iterrows():
        if i[1]['duration'] == 0:
            continue
        if i[1]['duration'] < (max_dur - 30): # 30 second margin
            print(f"{i[1]['stream']} is shorter than expected by {max_dur - i[1]['duration']:.4f} seconds")

        
    whole_durations_df.sort_values(by = 'duration', inplace = True)
    return(whole_durations_df)

def get_sampling_rate(df):
    effective_sampling_rate = 1 / (df.lsl_time_stamp.diff().median())
    return effective_sampling_rate

# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass