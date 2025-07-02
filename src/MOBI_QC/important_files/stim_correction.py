#%%
import pyxdf    # 1.17.0'
import pandas as pd    # 2.2.3
import numpy as np    #  2.26
import matplotlib.pyplot as plt    #3.10.1
from glob import glob
import librosa    # 0.11.0
from tqdm import tqdm
import sounddevice as sd    #0.5.2
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))






def trigger_recovery(stim_df, xdf_filename):
    ''' This function recovers the trigger events in the stim_df DataFrame'''
    from utils import get_event_data, import_mic_data

    mic_df = get_event_data(event='StoryListening',
                df=import_mic_data(xdf_filename=xdf_filename), 
                stim_df=stim_df
                )

    rate = 44100
    chunk_size = 4410
    threshold = 75
    audio = mic_df['int_array'].values.astype(np.int16)
    #num_chunks = len(audio) // chunk_size
    n_samples= len(audio)
    is_audio = np.zeros(n_samples, dtype=bool)
    energies = []
    for start in tqdm(range(0, n_samples, chunk_size)):
        end = min(start + chunk_size, n_samples)
        chunk = audio[start:end]
        energy = np.mean(np.abs(chunk))
        if energy > threshold:
            is_audio[start:end] = True
        energies.append(energy)
    mic_df['mic_signal'] = is_audio
    energies = np.array(energies)
    # Make sure 'is_audio' is a boolean Series (or 0/1 integers)
    is_audio = mic_df['mic_signal'].astype(bool)
    # Identify where values change (True <-> False)
    change_points = is_audio.ne(is_audio.shift()).cumsum()
    # Group by these change points and get lengths
    runs = mic_df.groupby(change_points).agg({
        'mic_signal': ['first', 'size'],
        'time': ['first', 'last']
    })
    runs.columns = ['value', 'length', 'start_time', 'end_time']
    runs = runs.reset_index(drop=True)
    # Select runs where value == False and length > 132300
    long_false_blocks = runs[(runs['value'] == False) & (runs['length'] > 132300)]   # silences much be longer than 132300 rows (44100 Hz x 3 seconds = 132300)
    print(long_false_blocks)
    # Initialize column with False
    mic_df['long_silence'] = False

    # Mark rows that fall into those long silence blocks
    for _, row in long_false_blocks.iterrows():
        mic_df.loc[(mic_df['time'] >= row['start_time']) & (mic_df['time'] <= row['end_time']), 'long_silence'] = True

    mic_df['is_audio'] = ~mic_df['long_silence']  # Invert long_silence to get is_audio

    # Example box function
    s = mic_df['is_audio'].astype(int)  # Convert boolean to int (0s and 1s)
    # Treat 1s (or True) as boxes
    is_box = s.astype(bool)

    # Find the start of each new box (True preceded by False)
    box_start = is_box & ~is_box.shift(fill_value=False)

    # Use cumsum to increment on each new box start
    box_ids = box_start.cumsum()

    # Mask: assign box ID where s == 1, else 0
    labeled_boxes = box_ids.where(is_box, 0)
    mic_df['audio_segment'] = labeled_boxes

    psycho = pd.read_csv(glob('/'.join(xdf_filename.split('/')[:-1])+'/*behavior.csv')[0], sep=',', header=0)
    audio_order = [f.split('/')[-1].split('.')[0] for f in psycho.AudioFile.unique() if pd.notna(f)]

    story_count = 0
    # get the duration in seconds of each audio segment
    mic_df['sound_id'] = 'silence'
    for i in range(1, mic_df['audio_segment'].max() + 1):
        segment = mic_df[mic_df['audio_segment'] == i]
        duration = (segment['time'].max() - segment['time'].min())
        if duration < 93:
            mic_df.loc[mic_df['audio_segment'] == i, 'sound_id'] = 'noise' 
        else:
            mic_df.loc[mic_df['audio_segment'] == i, 'sound_id'] = audio_order[story_count]
            story_count += 1

        print(f"Audio Segment {i}: Duration = {duration:.2f} seconds")

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

    for story in audio_order:
        story_df = mic_df[mic_df['sound_id'] == story]
        story_onset = mic_df.loc[mic_df['sound_id'] == story, 'time'].min()
        story_offset = mic_df.loc[mic_df['sound_id'] == story, 'time'].max()

        if story == 'Camp_Lose_A_Friend':
            event_id = 20
        elif story == 'Frog_Dissection_Disaster':
            event_id = 30
        elif story == 'I_Decided_To_Be_Myself_And_Won_A_Dance_Contest':
            event_id = 40
        elif story == 'I_Fully_Embarrassed_Myself_In_Zoom_Class1':
            event_id = 50
        elif story == 'Left_Home_Alone_in_a_Tornado':
            event_id = 60
        elif story == 'The_Birthday_Party_Prank':
            event_id = 70
        # Add the story onset and offset to the stim_df
        row = [event_id, f'{events[event_id]}', story_df.lsl_time_stamp.min(), story_df.time.min()]
        # Append the row to stim_df
        stim_df.loc[len(stim_df)] = row
        row = [event_id + 1, f'{events[event_id+1]}', story_df.lsl_time_stamp.max(), story_df.time.max()]
        # Append the row to stim_df
        stim_df.loc[len(stim_df)] = row
    # Sort stim_df by 'lsl_time_stamp'
    stim_df.sort_values('lsl_time_stamp', inplace=True)

    return stim_df


# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass