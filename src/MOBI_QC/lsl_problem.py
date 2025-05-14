import pyxdf
import pandas as pd
import numpy as np
from glob import glob
import datetime
import matplotlib.pyplot as plt
from pprint import pprint
from utils import *
import math

def lsl_quick_check(ps_df: pd.DataFrame):
    quickcheck = sum([not math.isclose(x, 1/120, abs_tol=1e-2) for x in ps_df.lsl_time_stamp.diff()]) - 1
    return quickcheck

def lsl_problem_plot(ps_df: pd.DataFrame, sub_id: str):
    plt.plot(ps_df['lsl_time_stamp'])
    plt.xlabel('Index')
    plt.ylabel('LSL Time Stamp (s)')
    plt.title('LSL Time Stamps (Physio Data)')
    plt.savefig(f'report_images/{sub_id}_LSL_timestamps.png')

def lsl_loss_percentage(df_dict: dict, sub_id: str) -> pd.DataFrame:
    # df with percent loss (diff greater than median)
    modalities = list(df_dict.keys())
    percent_list = []

    for modality in modalities:
        df = df_dict[modality]

        # median diff between lsl_time_stamp (with 1.05 margin) 
        df['diff'] = df['lsl_time_stamp'].diff()
        median = df['diff'].median() * 1.05
        # number of loss instances  
        loss_instances = (df['diff'] > median).sum()
        if loss_instances != 0:
            # amount of data skipped: values for which diff>median 
            amt_data_lost = df.loc[df['diff'] > median, 'diff'].values[0].sum()
            # total amount of data: last - first lsl_time_stamp
            amt_data_total = df['lsl_time_stamp'].values[-1] - df['lsl_time_stamp'].values[0]

            percent_lost = round(amt_data_lost/amt_data_total * 100, 3)
        else:
            percent_lost = 0
        percent_list.append({'subject': sub_id, 'modality': modality, 'num_losses': loss_instances, 'percent_lost': str(percent_lost)+'%'})
        
    percent_data_loss = pd.DataFrame(percent_list)
    percent_data_loss.sort_values(by='percent_lost', inplace=True, ascending=False)
    nonzero_loss = percent_data_loss[percent_data_loss['num_losses'] != 0]
    return nonzero_loss
    
def lsl_loss_before_social(df_dict: dict, sub_id: str, offset_social_timestamp: float) -> pd.DataFrame:
    modalities = list(df_dict.keys())
    social_percent_list = []

    for modality in modalities:
        df = df_dict[modality]
        social_df = df.loc[df.lsl_time_stamp <= offset_social_timestamp]

        # median diff between lsl_time_stamp (with 1.05 margin) 
        median1 = df['diff'].median() * 1.05

        # number of loss instances  
        loss_instances = (social_df['diff'] > median1).sum()
        percent_lost = 0
        amt_data_lost = 0

        # LSL loss starts and ends before offset_social
        if loss_instances != 0:
            # amount of data skipped: values for which diff>median 
            amt_data_lost = social_df.loc[social_df['diff'] > median1, 'diff'].values[0].sum()

        # offset social is between LSL loss onset + offset
        remaining_lost = offset_social_timestamp - social_df['lsl_time_stamp'].values[-1]
        if (remaining_lost) > 1:
            loss_instances +=1
            amt_data_lost = amt_data_lost + remaining_lost

        amt_data_total = offset_social_timestamp - social_df['lsl_time_stamp'].values[0]
        percent_lost = round(amt_data_lost/amt_data_total * 100, 3)

        social_percent_list.append({'subject': sub_id, 'modality': modality, 'num_losses': loss_instances, 'percent_lost': str(percent_lost)+'%'})
            
    percent_data_loss_social = pd.DataFrame(social_percent_list)
    percent_data_loss_social.sort_values(by='percent_lost', inplace=True, ascending=False)
    nonzero_loss_social = percent_data_loss_social[percent_data_loss_social['num_losses'] != 0]
    return nonzero_loss_social

def lsl_problem(xdf_filename:str):
    # load data 
    sub_id = xdf_filename.split('-')[1].split('/')[0]

    mic_df = import_mic_data(xdf_filename)
    stim_df = import_stim_data(xdf_filename)
    et_df = import_et_data(xdf_filename)
    cam_df = import_video_data(xdf_filename)
    eeg_df = import_eeg_data(xdf_filename)
    ps_df = import_physio_data(xdf_filename)

    df_dict = {
    'ps': ps_df,
    'et': et_df,
    'mic': mic_df,
    'cam': cam_df,
    'eeg': eeg_df}

    offset_social_timestamp = stim_df.loc[stim_df['event'] == 'Offset_SocialTask', 'lsl_time_stamp'].values[0]

    # optional: returns number of loss instances in  ps_df has any loss
    lsl_quick_check(ps_df)

    lsl_problem_plot(ps_df, sub_id)

    vars = {}
    vars['percent_loss'] = lsl_loss_percentage(df_dict, sub_id)
    print(vars['percent_loss'])
    vars['loss_before_social_task'] = lsl_loss_before_social(df_dict, sub_id, offset_social_timestamp)
    print(vars['loss_before_social_task'])

    return vars

