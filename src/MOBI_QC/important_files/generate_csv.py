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

def unpack_vars(lsl_vars, duration_vars):
    # Unpack lsl vars
    lsl_prob_vars = {}
    stream_vars = {}

    for key in lsl_vars.keys():
        if isinstance(lsl_vars[key], pd.DataFrame):
            lsl_k = lsl_vars[key].to_dict(orient='records')
            for i in range(len(lsl_k)):
                lsl_dict = {f"{lsl_k[i]['stream']}_{k}": v for k, v in lsl_k[i].items() if k != "stream" and k!='subject'}
                lsl_prob_vars.update(lsl_dict)
        else:
            lsl_prob_vars.update({key:lsl_vars[key]})
    
    # Unpack stream durations
    stream = duration_vars['Durations of each modality + comparison to expected duration:'].to_dict(orient='records')
    stream_vars = {}
    for i in range(len(stream)):
        stream_dict = {f"{stream[i]['stream']}_{k}": v for k, v in stream[i].items() if k != "stream"}
        stream_vars.update(stream_dict)
    return lsl_prob_vars, stream_vars

def add_modality_name(modality_vars):
    for key, vars in modality_vars.items():
        vars = {f"{key}_{k}": value for k, value in vars.items()}
        modality_vars[key] = vars
    return modality_vars

def generate_qc_dataframe(subject_id:str, collection_date: str, modality_vars):
    
    subject_csv = {'Subject': subject_id, 'Collection Date': collection_date}
    for modality in modality_vars:
        subject_csv.update(modality)
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
"""
def add_to_csv(subject_id, collection_date, modality_vars):
    modality_vars['lsl_vars'], modality_vars['duration_vars'] = unpack_vars(modality_vars['lsl_vars'], modality_vars['duration_vars'])
    subject_csv_df = generate_qc_dataframe(subject_id, collection_date, modality_vars.values())
    added_to_csv = save_to_csv(subject_csv_df)
    return added_to_csv
"""
def generate_csv(xdf_filename:str, modality_vars:dict):

    csvfilename = 'CUNY_QC.csv'
    subject_id, collection_date = get_subjectID_and_collectionDate(xdf_filename)
    if os.path.exists(csvfilename) == True and (check_data_exists(csvfilename, subject_id)).empty != True:
        print ('Participant data already exists')
        return check_data_exists(csvfilename, subject_id)
    else:
        modality_vars['lsl'], modality_vars['duration'] = unpack_vars(modality_vars['lsl'], modality_vars['duration'])
        modality_vars = add_modality_name(modality_vars)
        subject_csv_df = generate_qc_dataframe(subject_id, collection_date, modality_vars.values())
        status = save_to_csv(subject_csv_df)
        #status = add_to_csv(subject_id, collection_date, modality_vars)
        return status


# %%

# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass

# %%
