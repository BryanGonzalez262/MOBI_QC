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
    """
    Quick check for gaps in LSL timestamps using physio data.
    Args:
        ps_df (pd.DataFrame): Dataframe containing the physio data.
    Returns:
        quickcheck (int): Number of instances where the difference between consecutive LSL timestamps is not close to inverse of sampling rate.
    """
    sampling_rate = get_sampling_rate(ps_df)
    quickcheck = sum([not math.isclose(x, 1/sampling_rate, abs_tol=1e-2) for x in ps_df.lsl_time_stamp.diff()]) - 1
    return quickcheck

def lsl_problem_plot(plot_df: pd.DataFrame, sub_id: str, modality_to_plot: str):
    """
    Plot the LSL timestamps for the physio data.
    Args:
        plot_df (pd.DataFrame): Dataframe with data to be plotted.
        sub_id (str): Subject ID.
    """
    plt.figure()
    plt.plot(plot_df['lsl_time_stamp'], color = 'g')
    plt.xlabel('Index')
    plt.ylabel('LSL Time Stamp (s)')
    plt.title(f'LSL Time Stamps ({modality_to_plot.upper()} Data)')
    plt.tight_layout()
    plt.savefig(f'report_images/{sub_id}_LSL_timestamps.png')
    
def lsl_loss_percentage(df_map: dict, error_map: dict, sub_id: str) -> pd.DataFrame:
    """
    Calculate the percentage of data loss for each modality based on LSL timestamps.
    Args:
        df_map (dict): Dictionary containing dataframes for each modality.
        error_map (dict): Contains booleans for each data modality indicating error. 
        sub_id (str): Subject ID.
    Returns:
        percent_data_loss (pd.DataFrame): Dataframe containing the percentage of data loss for each modality.
    """
    # df with percent loss (diff greater than median)
    modalities = list(df_map.keys())
    percent_list = []

    for modality in modalities:
        if error_map[modality]: 
            print(f'No {modality} data for participant {sub_id}')
            continue
        df = df_map[modality]

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
            
            percent_lost = amt_data_lost/amt_data_total * 100
        else:
            percent_lost = 0
        percent_list.append({'subject': sub_id, 'stream': modality, 'num_losses': loss_instances, 'percent_lost': round(percent_lost, 4)})
        
    percent_data_loss = pd.DataFrame(percent_list)
    percent_data_loss.sort_values(by='percent_lost', inplace=True, ascending=False)
    nonzero_loss = percent_data_loss[percent_data_loss['num_losses'] != 0]
    return percent_data_loss
    
def lsl_loss_before_social(df_map: dict, error_map: dict, sub_id: str, offset_social_timestamp: float) -> pd.DataFrame:
    """
    Calculate the percentage of data loss before the social task offset for each modality.
    Args:
        df_map (dict): Dictionary containing dataframes for each modality.
        error_map (dict): Contains booleans for each data modality indicating error. 
        sub_id (str): Subject ID.
        offset_social_timestamp (float): Timestamp of  social task offset.
    Returns:
        percent_data_loss_social (pd.DataFrame): Dataframe containing the percentage of data loss before the social task offset for each modality.
    """
    modalities = list(df_map.keys())
    social_percent_list = []

    for modality in modalities:
        if error_map[modality]: 
            print(f'No {modality} data for participant {sub_id}')
            continue
        df = df_map[modality]
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
        percent_lost = amt_data_lost/amt_data_total * 100

        social_percent_list.append({'subject': sub_id, 'stream': modality, 'num_losses': loss_instances, 'percent_lost': round(percent_lost, 4)})
            
    percent_data_loss_social = pd.DataFrame(social_percent_list)
    percent_data_loss_social.sort_values(by='percent_lost', inplace=True, ascending=False)
    nonzero_loss_social = percent_data_loss_social[percent_data_loss_social['num_losses'] != 0]
    return percent_data_loss_social

def lsl_problem_qc(xdf_filename:str, stim_df:pd.DataFrame, df_map:dict, error_map:dict, modality_to_plot='et') -> dict:
    """
    Main function to check for LSL timestamp gaps in the data.
    Args:
        xdf_filename (str): Path to the XDF file.
        stim_df (pd.DataFrame): Contains stimulus markers
        df_map (dict): Dictionary with all data dfs 
        error_map (dict): Contains booleans for each data modality indicating error. 
        modality_to_plot (str): Which modality you want plotted. can be one of 'et', 'ps', 'mic', 'cam', 'eeg'   

        Returns:
        vars (dict): Dictionary containing the percentage of data loss for each modality and the number of loss instances.
    """
    # load data 
    sub_id = xdf_filename.split('sub-')[1].split('/')[0]

    offset_social_timestamp = stim_df.loc[stim_df['event'] == 'Offset_SocialTask', 'lsl_time_stamp'].values[0]

    # optional: returns number of loss instances in ps_df
    # lsl_quick_check(ps_df)
    # ps_df = df_map['ps']
    plot_df = df_map[modality_to_plot]
    lsl_problem_plot(plot_df, sub_id, modality_to_plot)

    vars = {}

    vars['percent_loss'] = lsl_loss_percentage(df_map, error_map, sub_id)
    if vars['percent_loss'].empty:
        vars['percent_loss'] = f"no data loss detected for {sub_id} for entire experiment"
    print(vars['percent_loss'])

    vars['loss_before_social_task'] = lsl_loss_before_social(df_map, error_map, sub_id, offset_social_timestamp)
    if vars['loss_before_social_task'].empty:
        vars['loss_before_social_task'] = f"no data loss detected for {sub_id} before social task"
    print(vars['loss_before_social_task'])

    return vars

# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass