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

def get_qc_metrics(xdf_filename:str, subject_id:str, stim_df:pd.DataFrame):
    eeg_vars,_,_,eeg_df = compute_eeg_pipeline(xdf_filename, stim_df=stim_df, task='Experiment')
    eeg_vars = {f"eeg_{key}": value for key, value in eeg_vars.items()}

    et_vars,et_df = et_qc(xdf_filename, stim_df, task='Experiment')
    et_vars = {f"et_{key}": value for key, value in et_vars.items()}

    ecg_vars,_,ps_df= ecg_qc(xdf_filename, stim_df, task='Experiment')
    ecg_vars = {f"ecg_{key}": value for key, value in ecg_vars.items()}

    eda_vars,_,_,_ = eda_qc(xdf_filename, stim_df, task='Experiment')
    eda_vars = {f"eda_{key}": value for key, value in eda_vars.items()}

    rsp_vars,_ = rsp_qc(xdf_filename, stim_df, task='Experiment')
    rsp_vars = {f"rsp_{key}": value for key, value in rsp_vars.items()}

    mic_vars,mic_df = mic_qc(xdf_filename, stim_df, task='Experiment')
    mic_vars = {f"mic_{key}": value for key, value in mic_vars.items()}

    webcam_file = glob(f'/Users/apurva.gokhe/Documents/CUNY_QC/data/sub-{subject_id}/*.avi')[0]
    webcam_vars,webcam_df = webcam_qc(xdf_filename=xdf_filename, video_file=webcam_file, stim_df=stim_df, task='Experiment')
    webcam_vars = {f"webcam_{key}": value for key, value in webcam_vars.items()}

    behavior_vars = behavior_qc(xdf_filename)
    behavior_vars = {f"behavior_{key}": value for key, value in behavior_vars.items()}

    df_map = {
        'et': et_df,
        'ps': ps_df,
        'mic': mic_df,
        'cam': webcam_df
        }

    return [eeg_vars, et_vars, ecg_vars, eda_vars, rsp_vars, mic_vars, webcam_vars, behavior_vars], df_map
    #return [eeg_vars, et_vars, ecg_vars, eda_vars, rsp_vars, webcam_vars, behavior_vars]

def generate_qc_dataframe(xdf_filename:str, stim_df:pd.DataFrame, subject_id:str, collection_date: str):
    
    modality_vars, df_map = get_qc_metrics(xdf_filename, subject_id, stim_df)

    subject_csv = {'Subject': subject_id, 'Collection Date': collection_date}
    for modality in modality_vars:
        subject_csv.update(modality)

    stream_durations = get_durations(xdf_path=xdf_filename, task='Experiment', df_map = df_map, stim_df = stim_df)
    stream_durations_dict = {stream_durations['stream'][i]+' (duration,mm:ss,percent)': [[stream_durations['duration'][i].item(), stream_durations['mm:ss'][i], stream_durations['percent'][i]]] for i in range(stream_durations['stream'].size)} 
    subject_csv.update(stream_durations_dict)

    lsl_problem_vars = lsl_problem_qc(xdf_filename=xdf_filename, df_map=df_map, stim_df=stim_df, modality_to_plot='et')
    subject_csv.update(lsl_problem_vars)

    subject_csv_df = pd.DataFrame([subject_csv])

    return subject_csv_df

def check_subject_data_exists(filename, subject_id):
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

def add_to_csv(xdf_filename:str, subject_id, collection_date):
    stim_df = import_stim_data(xdf_filename)
    subject_csv_df = generate_qc_dataframe(xdf_filename, stim_df, subject_id, collection_date)
    added_to_csv = save_to_csv(subject_csv_df)
    return added_to_csv

def generate_csv(xdf_filename:str, stim_df, modality_vars, df_map, error_map):

    csvfilename = 'CUNY_QC.csv'
    subject_id, collection_date = get_subjectID_and_collectionDate(xdf_filename)
    if os.path.exists(csvfilename) == True and (check_subject_data_exists(csvfilename, subject_id)).empty != True:
        print ('Participant data already exists')
        return check_subject_data_exists(csvfilename, subject_id)
    else:
        status = add_to_csv(xdf_filename, subject_id, collection_date, stim_df, modality_vars, df_map, error_map)
        return status


# %%

# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass

# %%
