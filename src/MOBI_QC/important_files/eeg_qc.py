
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import pyprep
import pyxdf
from utils import *
from scipy.signal import welch
import warnings
import json
warnings.filterwarnings("ignore")
from mne.preprocessing import ICA
from mne_icalabel import label_components



def compute_eeg_pipeline(xdf_filename, stim_df, task='RestingState'):
    """
    This function computes the EEG pipeline for the given xdf file.
    Args:
        xdf_filename (str): The path to the xdf file.
    """    

    def annotate_blinks(
        raw: mne.io.Raw, ch_name: list[str] = ["E25", "E8"]
    ) -> mne.Annotations:
        """Annotate the blinks in the EEG signal.
    
        Args:
            raw (mne.io.Raw): The raw EEG data in mne format.
            ch_name (list[str]): The channels to use for the EOG. Default is
                                ["Fp1", "Fp2"]. I would suggest to use the
                                channels that are the most frontal (just above
                                the eyes). In the case of an EGI system the
                                channels would be "E25" and "E8".
    
        Returns:
            mne.Annotations: The annotations object containing the blink events.
        """
        eog_epochs = mne.preprocessing.create_eog_epochs(raw, ch_name=ch_name)
        blink_annotations = mne.annotations_from_events(
            eog_epochs.events,
            raw.info["sfreq"],
            event_desc={eog_epochs.events[0, 2]: "blink"},
        )
        return blink_annotations

    def annotate_muscle(raw: mne.io.Raw) -> mne.Annotations:
        muscle_annotations, _ = mne.preprocessing.annotate_muscle_zscore(
            raw, 
            threshold=3, # this needs to be calibrated for the entire dataset
            ch_type='eeg', 
            min_length_good=0.1, 
            filter_freq=(95, 120), 
            )
    
        return muscle_annotations
    subject = xdf_filename.split('sub-')[-1].split('_')[0]
    vars = {varname: np.nan for varname in ['percent_good', 'bad_channels_before',
                                            'interpolated_channels', 'bad_channels_after',
                                            'excluded_components']}
    eeg_error = False
    try:
        whole_eeg_df = import_eeg_data(xdf_filename)

        if len(glob('/'.join(xdf_filename.split('/')[:-1]) +'/*.fif')) < 1:
            df = get_event_data(event=task, 
                                df=ddf,
                                stim_df=stim_df)
            
            ch_names = [f"E{i+1}" for i in range(df.shape[1] - 1)]
            info = mne.create_info(ch_names, 
                                sfreq=1/df.lsl_time_stamp.diff().mean(), 
                                ch_types='eeg')
            df.drop(columns=['lsl_time_stamp'], inplace=True)

            raw = mne.io.RawArray(df.T * 1e-6, info=info) # multiplying by 1e-6 converts to volts

            # Create a Cz reference
            value = np.zeros((1, raw.n_times))
            info = mne.create_info(["Cz"], raw.info['sfreq'], ch_types='eeg')
            cz = mne.io.RawArray(value, info)
            raw.add_channels([cz], force_update_info=True)

            # Apply a montage
            montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
            raw.set_montage(montage, on_missing='ignore')


            prep_params = {
                    "ref_chs": "eeg",
                    "reref_chs": "eeg",
                    "line_freqs": np.arange(60, raw.info["sfreq"] / 2, 60),
                }
            # these params set up the robust reference  - i.e. median of all channels and interpolate bad channels
            prep = pyprep.PrepPipeline(raw, montage=montage, channel_wise=True, prep_params=prep_params)
            print("STARTING preprocessing")

            prep_output = prep.fit()
            raw_cleaned = prep_output.raw_eeg
            print("DONE with preprocessing")

            # check if cleaned file already exists
            
            print(f"Bad channels before robust reference: {prep.noisy_channels_original['bad_all']}")
            vars['bad_channels_before'] = prep.noisy_channels_original['bad_all']
            print(f"Interpolated channels: {prep.interpolated_channels}")
            vars['interpolated_channels'] = prep.interpolated_channels
            print(f"Bad channels after interpolation: {prep.still_noisy_channels}")
            vars['bad_channels_after'] = prep.still_noisy_channels
            # safe the vars dictionary to a file
            with open('/'.join(xdf_filename.split('/')[:-1]) + f'/sub-{subject}_ses-S001_task-CUNY_run-001_eeg_clean_vars.json', 'w') as f:
                json.dump(vars, f, indent=4)
        else:
            raw_cleaned = mne.io.read_raw_fif('/'.join(xdf_filename.split('/')[:-1]) + f'/sub-{subject}_ses-S001_task-CUNY_run-001_eeg_clean.fif', preload=True)
            # read the vars dictionary from the file
            with open('/'.join(xdf_filename.split('/')[:-1]) + f'/sub-{subject}_ses-S001_task-CUNY_run-001_eeg_clean_vars.json', 'r') as f:
                vars = json.load(f)
            print(f"Bad channels before robust reference: {vars['bad_channels_before']}")
            print(f"Interpolated channels: {vars['interpolated_channels']}")
            print(f"Bad channels after interpolation: {vars['bad_channels_after']}")
        
        # Annotate blinks and muscle artifacts to compute percent good data
        blink_annotations = annotate_blinks(raw_cleaned, ch_name=["E25", "E8"])
        muscle_annotations = annotate_muscle(raw_cleaned)
        all_annotations = blink_annotations + muscle_annotations + raw_cleaned.annotations
        raw_cleaned.set_annotations(all_annotations)
        # Create a binary array
        binary_mask = np.zeros(len(raw_cleaned.times), dtype=int)
        # Iterate over annotations
        for annot in raw_cleaned.annotations:
            onset_sample = int(annot['onset'] * raw_cleaned.info['sfreq'])
            duration_sample = int(annot['duration'] * raw_cleaned.info['sfreq'])
            binary_mask[onset_sample:onset_sample + duration_sample] = 1
        percent_good = 1 - np.sum(binary_mask) / len(binary_mask)
        print(f'Percent Good Data: {percent_good * 100:.2f}%')
        vars['percent_good'] = percent_good * 100

        # set notch filter
        raw_cleaned.notch_filter(60)
        # set bandpass filter
        raw_cleaned.filter(l_freq=1.0, h_freq=50.0) # only keeping frequencies between 1-50 Hz
        # play around with this number to get components 
        # that seem to represent the actual brain activations well
        num_components = .95 
        ica = ICA(n_components=num_components, method='picard')
        ica.fit(raw_cleaned, reject_by_annotation=True)
        fig = ica.plot_components( title='ICA Components')
        # Save the ICA plot
        if isinstance(fig, list):
            fig[0].savefig(f'report_images/sub-{subject}_eeg_ica_components.png', bbox_inches='tight')
        else:
            fig.savefig(f'report_images/sub-{subject}_eeg_ica_components.png', bbox_inches='tight')

        # MNE ICA LABEL
        ic_labels = label_components(raw_cleaned, ica, method='iclabel')
        # exclude components that are not brain activity
        exclude_components = [i for i, label in enumerate(ic_labels['labels']) if 
                            label not in ['brain', 'other']]
        ica.exclude = exclude_components
        vars['excluded_components'] = exclude_components
        ica.apply(raw_cleaned)
        raw_cleaned.annotations.delete([i for i, desc in enumerate(raw_cleaned.annotations.description) if desc == 'blink' or desc == 'BAD_muscle'])
        save_path = '/'.join(xdf_filename.split('/')[:-1]) + f'/sub-{subject}_ses-S001_task-CUNY_run-001_eeg_clean.fif'
        raw_cleaned.save(save_path, overwrite=True)
        fig = raw_cleaned.plot(show_scrollbars=False,
                                show_scalebars=False,events=None, start=0, 
                                duration=200,n_channels=50, scalings=.35e-4, color='k', title='EEG Data after ICA')
        fig.savefig(f'report_images/{subject}_eeg_cleaned.png', dpi=300, bbox_inches='tight')
        fig = raw_cleaned.plot_psd(fmax=50, average=False, show=True)
        fig.savefig(f'report_images/{subject}_eeg_cleaned_psd.png', dpi=300, bbox_inches='tight')
        # TODO: Possibly add this plot_overlay image to show before/after ICA
        # ica.plot_overlay(raw_cleaned, picks=[0,3,10,11,13,14,18,19])
        return vars, whole_eeg_df, eeg_error

    except IndexError:
        whole_eeg_df = pd.DataFrame()
        vars.update({key: float('nan') for key in vars.keys()})
        eeg_error = True
        print(f'Error: No EEG data found for participant {subject} in {xdf_filename}.')
        return vars, whole_eeg_df, eeg_error


def test_eeg_pipeline(xdf_filename):
    """
    Test the EEG pipeline for the given xdf file.
    Args:
        xdf_filename (str): The path to the xdf file.
    """
    vars, raw_cleaned = [43, [1,2,3]]#compute_eeg_pipeline(xdf_filename)
    print(vars)
    #print(raw_cleaned.info['bads'])
    return vars, raw_cleaned

# allow the functions in this script to be imported into other scripts
if __name__ == "__main__":
    pass
