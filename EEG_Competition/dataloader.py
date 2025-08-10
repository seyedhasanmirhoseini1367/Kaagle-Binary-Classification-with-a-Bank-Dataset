import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import mne  # for loading .set EEG files
from EEG_Competition.Targets_Challenge1 import *


def reaction_time(df):
    rt_target = []
    target_indices = df.index[df['feedback'].str.contains('face', na=False)].tolist()

    for idx in target_indices:
        press_buttons = df[(df.index == idx) & (df['value'].str.contains('buttonPress'))]
        prev_value = df.loc[idx - 1, 'value']
        if prev_value not in ['left_target', 'right_target']:
            continue
        if not press_buttons.empty:
            rt = press_buttons.iloc[0]['onset'] - df.loc[idx - 1, 'onset']
            target = press_buttons.iloc[0]['feedback']
            rt_target.append((np.round(rt, 3), target))
    return rt_target

'''
CCDataset: EEG Contrast Change Detection Dataset Loader
This PyTorch Dataset class loads EEG data and associated targets

Data organization:
Root directories contain multiple subject folders (e.g., sub-XXX), each with an eeg folder.
Each subject has 3 EEG run files (.set) and corresponding event files (.tsv).

What the code does:
Finds all subject EEG folders across given root directories.
For each subject, loads all 3 EEG runs (.set files) and their event TSVs.
Extracts EEG data as tensors (channels × samples).
Parses event files to calculate reaction times and feedback labels (smiley/sad faces) as targets.
Returns data for all runs of a subject as lists of tensors.

Usage:
__len__ returns number of subjects.
__getitem__ returns lists of EEG data tensors and corresponding reaction time & feedback target tensors for all runs of a given subject.

'''

import os
import glob
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import mne  # for loading .set EEG files
from EEG_Competition.Targets_Challenge1 import *


def reaction_time(df):
    """
    Extract reaction times and feedback labels from event DataFrame.
    Looks for button presses following left/right target events and feedback (smiley/sad faces).
    Returns list of tuples: (reaction_time_in_seconds, feedback_label).
    """
    rt_target = []
    target_indices = df.index[df['feedback'].str.contains('face', na=False)].tolist()

    for idx in target_indices:
        # Get buttonPress events at the current index
        press_buttons = df[(df.index == idx) & (df['value'].str.contains('buttonPress'))]
        # Check if the previous event is a left or right target
        prev_value = df.loc[idx - 1, 'value']
        if prev_value not in ['left_target', 'right_target']:
            continue
        if not press_buttons.empty:
            # Calculate reaction time as difference between button press and target onset
            rt = press_buttons.iloc[0]['onset'] - df.loc[idx - 1, 'onset']
            target = press_buttons.iloc[0]['feedback']
            rt_target.append((np.round(rt, 3), target))
    return rt_target


class CCDataset(Dataset):
    def __init__(self, root_dirs, transform=None):
        """
        Initialize dataset by searching root directories for subject EEG folders.
        Stores paths to each subject's eeg directory for loading in __getitem__.
        """
        self.subjects_dirs = []

        for root_dir in root_dirs:
            # Find all subject IDs starting with 'sub-'
            subjects_ids = [id for id in os.listdir(root_dir) if id.startswith('sub-')]
            for subject in subjects_ids:
                subject_dir = os.path.join(root_dir, subject, 'eeg')
                if os.path.isdir(subject_dir):
                    self.subjects_dirs.append(subject_dir)

        self.transform = transform

    def __len__(self):
        # Number of subjects in dataset
        return len(self.subjects_dirs)

    def __getitem__(self, idx):
        """
        For a given subject index, load all EEG runs and corresponding event files.
        Returns:
          - runs_data: list of EEG tensors (channels x samples) per run
          - runs_rt: list of reaction time tensors per run
          - runs_fb: list of feedback label tensors per run
        """
        subject_path = self.subjects_dirs[idx]
        participant_id = os.path.basename(os.path.dirname(subject_path))

        # Find all EEG .set files for the subject's contrastChangeDetection runs
        eeg_paths = glob.glob(
            os.path.join(subject_path, f"{participant_id}_task-contrastChangeDetection_run-*_eeg.set"))

        runs_data = []
        runs_rt = []
        runs_fb = []

        for eeg_path in eeg_paths:
            # Corresponding events TSV file for the EEG run
            tsv_file = eeg_path.replace('_eeg.set', '_events.tsv')
            if not os.path.exists(tsv_file):
                continue  # skip if event file missing

            # Load EEG data using MNE (channels x samples)
            raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
            data = raw.get_data()
            data_tensor = torch.tensor(data, dtype=torch.float32)

            if self.transform:
                data_tensor = self.transform(data_tensor)

            # Load event data and extract reaction times and feedback labels
            tsv_df = pd.read_csv(tsv_file, sep='\t')
            rt_targets = reaction_time(tsv_df)
            rts = [x[0] for x in rt_targets]
            feedbacks = [1 if x[1] == 'smiley_face' else 0 for x in rt_targets]

            runs_data.append(data_tensor)
            runs_rt.append(torch.tensor(rts, dtype=torch.float32))
            runs_fb.append(torch.tensor(feedbacks, dtype=torch.long))

        return runs_data, runs_rt, runs_fb


# Usage example
base_path = 'D:/EEGChallenge'
root_dirs = [os.path.join(base_path, r) for r in os.listdir(base_path)]

dataset = CCDataset(root_dirs=root_dirs)

# Access first subject's data
eeg_batch, rt_batch, fb_batch = dataset[0]

for i in range(len(eeg_batch)):
    print(f'Shape of Run {i}: {eeg_batch[i].shape}')
    print(f'Targets of Run {i}: Reaction times ({len(rt_batch[i])}), Feedbacks ({len(fb_batch[i])})\n')

# DataLoader example (batches subjects, batch_size=1 means one subject per batch)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
