import os
import glob
import pandas as pd
import torch
from torch.utils.data import Dataset
import mne  # for loading .set EEG files

'''
Challenge2Dataset
This PyTorch Dataset loads multi-task EEG data and associated psychopathology factor targets for Challenge 2.
Input: For each subject, loads all available .set EEG recordings from multiple cognitive tasks.
Targets: Loads continuous scores for four psychopathology factors (p_factor, internalizing, externalizing, attention) from the root directory’s participants.tsv.
Output: Returns a list of EEG runs (tensors) per subject, along with a 4D target tensor.
Handles variable number and length of EEG runs per participant.
Designed for subject-level batching in a DataLoader (typically batch_size=1).
'''

class DataloaderChallenge2(Dataset):
    def __init__(self, root_dirs, transform=None):
        """
        root_dirs: list of base directories like hbn_bids_R1, hbn_bids_R2, ...
        Each contains participants folders (sub-xxx) with eeg data and a participants.tsv file with targets.
        """
        self.subjects_dirs = []  # list of tuples (subject_dir, participant_id, root_dir)
        self.transform = transform

        for root_dir in root_dirs:
            # Load targets for this root_dir from participants.tsv
            participants_tsv = os.path.join(root_dir, 'participants.tsv')
            if not os.path.exists(participants_tsv):
                raise FileNotFoundError(f"Missing participants.tsv in {root_dir}")
            self.participants_df = pd.read_csv(participants_tsv, sep='\t')

            # Find all subjects
            subjects_ids = [d for d in os.listdir(root_dir) if d.startswith('sub-')]
            for subject in subjects_ids:
                subject_dir = os.path.join(root_dir, subject, 'eeg')
                if os.path.isdir(subject_dir):
                    self.subjects_dirs.append((subject_dir, subject, root_dir))

    def __len__(self):
        return len(self.subjects_dirs)

    def __getitem__(self, idx):
        subject_dir, participant_id, root_dir = self.subjects_dirs[idx]

        # Load all .set EEG files for this participant
        eeg_paths = glob.glob(os.path.join(subject_dir, '*.set'))
        runs_data = []

        for eeg_path in eeg_paths:
            raw = mne.io.read_raw_eeglab(eeg_path, preload=True)
            data = raw.get_data()  # (channels x samples)
            data_tensor = torch.tensor(data, dtype=torch.float32)
            if self.transform:
                data_tensor = self.transform(data_tensor)
            runs_data.append(data_tensor)

        # Extract targets from participants.tsv
        participants_tsv = os.path.join(root_dir, 'participants.tsv')
        df = pd.read_csv(participants_tsv, sep='\t')
        row = df[df['participant_id'] == participant_id]

        if row.empty:
            raise ValueError(f"No target info found for participant {participant_id}")

        # Extract continuous targets as floats
        p_factor = float(row['p_factor'].values[0])
        internalizing = float(row['internalizing'].values[0])
        externalizing = float(row['externalizing'].values[0])
        attention = float(row['attention'].values[0])

        targets = torch.tensor([p_factor, internalizing, externalizing, attention], dtype=torch.float32)

        return runs_data, targets


# Usage example:
base_path = 'D:/EEGChallenge'
root_dirs = [os.path.join(base_path, d) for d in os.listdir(base_path)]
root_dirs

dataset = DataloaderChallenge2(root_dirs=root_dirs[:1])  # first root dir only

print(f"Number of subjects: {len(dataset)}")

runs_data, targets = dataset[0]

print(f"Number of EEG runs for subject: {len(runs_data)}")
print(f"Shape of first run data: {runs_data[0].shape}")
print(f"Targets: {targets}")
