import os
import glob
import pandas as pd
from utils import *
from eeg_qc import *
from et_qc import *
from ecg_qc import *
from eda_qc import *
from rsp_qc import *
from mic_qc import *
from webcam_qc import *
from behavior_qc import *
from lsl_problem import *

def get_subjectID_and_collectionDate(xdf_filename:str):
    
    subject_id = xdf_filename.split('sub-')[1].split('/')[0]
    collection_date = get_collection_date(xdf_filename)
    
    return subject_id, collection_date

#def check_for_report(subject_id):

def generate_qc_dataframe(subject_id:str, collection_date: str, modality_vars):
    
    subject_csv = {'Subject': subject_id, 'Collection Date': collection_date}
    for modality in modality_vars:
        if modality_vars.index(modality) == -1:
            stream_durations_dict = {modality['stream'][i]+' (duration,mm:ss,percent)': [[modality['duration'][i].item(), modality['mm:ss'][i], modality['percent'][i]]] for i in range(modality['stream'].size)} 
            subject_csv.update(stream_durations_dict)
        else:
            subject_csv.update(modality)

    #stream_durations = get_durations(xdf_path=xdf_filename, task='Experiment', stim_df=stim_df, df_map=df_map, error_map=error_map)
    #stream_durations_dict = {stream_durations['stream'][i]+' (duration,mm:ss,percent)': [[stream_durations['duration'][i].item(), stream_durations['mm:ss'][i], stream_durations['percent'][i]]] for i in range(stream_durations['stream'].size)} 
    #subject_csv.update(stream_durations_dict)

    #lsl_problem_vars = lsl_problem_qc(xdf_filename=xdf_filename, df_map=df_map, stim_df=stim_df, modality_to_plot='et')
    #subject_csv.update(lsl_problem_vars)

    subject_csv_df = pd.DataFrame([subject_csv])

    return subject_csv_df

def check_data_exists(filename, subject_id):
    existing_data = pd.read_csv(filename)  
    if subject_id in existing_data['Subject'].values:
        rows = existing_data.loc[existing_data['Subject'] == subject_id]
        return rows
    else:
        rows = pd.DataFrame()
        return rows
    
def save_to_csv(subject_csv_df:pd.DataFrame):
    if os.path.exists('CUNY_QC.csv') == True:
        subject_csv_df.to_csv('CUNY_QC.csv', mode='a', index=False, header=False)
        return 'Saved'
    elif os.path.exists('CUNY_QC.csv') == False:
        subject_csv_df.to_csv('CUNY_QC.csv', mode='a', index=False, header=True)
        return 'Saved'
    else:
        return 'Export Error'

def add_to_csv(subject_id, collection_date, modality_vars):
    subject_csv_df = generate_qc_dataframe(subject_id, collection_date, modality_vars)
    added_to_csv = save_to_csv(subject_csv_df)
    return added_to_csv

def generate_csv(xdf_filename:str, stim_df, modality_vars):

    csvfilename = 'CUNY_QC.csv'
    subject_id, collection_date = get_subjectID_and_collectionDate(xdf_filename)
    if os.path.exists(csvfilename) == True and (check_data_exists(csvfilename, subject_id)).empty != True:
        print ('Participant data already exists')
        return check_data_exists(csvfilename, subject_id)
    else:
        status = add_to_csv(subject_id, collection_date, modality_vars)
        return status


# %%

# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass

# %%
