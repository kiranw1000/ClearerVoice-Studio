import numpy as np
import math, os, csv, ctypes

import torch
import torch.nn as nn
import torch.utils.data as data
import librosa
import soundfile as sf
import pandas as pd

from .utils import DistributedSampler
import multiprocessing as mp

def load_shared_eegs(args, partition='train'):
    mix_lst=open(args.mix_lst_path).read().splitlines()
    mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_lst))#[:200]
    mix_lst = sorted(mix_lst, key=lambda data: float(data.split(',')[-1]), reverse=True)
    trial_list = set([tuple(line.split(",")[1:3]) for line in mix_lst])
    eeg_list = []
    for subject, trial in trial_list:
        eeg_path = f'{args.reference_direc}S{subject}Tra{trial}.npy'
        eeg_data = np.load(eeg_path)
        shared_eeg = mp.Array('f', eeg_data.flatten())  # Flatten for 1D shared array
        eeg_list.append((int(subject), int(trial), shared_eeg, eeg_data.shape))  # Store shape for reshaping
    return eeg_list, mix_lst

def get_dataloader_eeg(args, partition):
    if args.num_workers > 0:
        shared_eegs, mix_lst = load_shared_eegs(args, partition=partition)
        print(f"Loaded {len(shared_eegs)} EEG datasets into shared memory for multiprocessing.")
        datasets = dataset_eeg_mp(args, partition, shared_eegs, mix_lst)
    else:
        datasets = dataset_eeg(args, partition)

    sampler = DistributedSampler(
        datasets,
        num_replicas=args.world_size,
        rank=args.local_rank) if args.distributed else None

    generator = data.DataLoader(datasets,
            batch_size = 1,
            shuffle = (sampler is None),
            num_workers = args.num_workers,
            sampler=sampler,
            collate_fn=custom_collate_fn)
    
    return sampler, generator

def custom_collate_fn(batch):
    a_mix, a_tgt, ref_tgt = batch[0]
    a_mix = torch.tensor(a_mix)
    a_tgt = torch.tensor(a_tgt) 
    ref_tgt = torch.tensor(ref_tgt) 
    return a_mix, a_tgt, ref_tgt

class dataset_eeg(data.Dataset):
    def __init__(self, args, partition):
        self.minibatch =[]
        self.args = args
        self.partition = partition
        self.max_length = args.max_length
        self.audio_sr=args.audio_sr
        self.ref_sr=args.ref_sr
        self.speaker_no=args.speaker_no
        self.batch_size=args.batch_size

        self.mix_lst_path = args.mix_lst_path
        self.audio_direc = args.audio_direc
        self.eeg_direc = args.reference_direc
        
        mix_lst=open(self.mix_lst_path).read().splitlines()
        mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_lst))#[:200]
        mix_lst = sorted(mix_lst, key=lambda data: float(data.split(',')[-1]), reverse=True)
        
        start = 0
        while True:
            end = min(len(mix_lst), start + self.batch_size)
            self.minibatch.append(mix_lst[start:end])
            if end == len(mix_lst):
                break
            start = end

        self.eeg_dict={}
        for subject in range(1,args.subjects+1):
            for trial in range(1,args.trials+1):
                eeg_path = f'{self.eeg_direc}S{subject}Tra{trial}.npy'
                eeg_data = np.load(eeg_path)
                self.eeg_dict[(subject,trial)] = (eeg_data, eeg_data.shape)



    def __getitem__(self, index):
        mix_audios = []
        tgt_audios = []
        tgt_eegs = []

        batch_lst = self.minibatch[index]
        min_length_second = float(batch_lst[-1].split(',')[-1])  # truncate to the shortest utterance in the batch
        min_length_eeg = math.floor(min_length_second * self.ref_sr)
        min_length_audio = math.floor(min_length_second * self.audio_sr)
        min_length_eeg = min(min_length_eeg, self.max_length * self.ref_sr)
        min_length_audio = min(min_length_audio, self.max_length * self.audio_sr)

        for line_cache in batch_lst:
            line = line_cache.split(',')

            # Load target EEG
            subject, trial = line[1], line[2]
            eeg_data = self.get_eeg(int(subject), int(trial))
            eeg_start = int(float(line[4]) * self.ref_sr)
            eeg_end = eeg_start + min_length_eeg
            eeg_data = eeg_data[eeg_start:eeg_end, :]

            # Load target audio
            tgt_audio_path = self.audio_direc + line[3]
            start = float(line[4]) * self.audio_sr
            end = start + min_length_audio
            if not os.path.exists(tgt_audio_path):
                raise FileNotFoundError(f"Target audio file not found: {tgt_audio_path}")
            a_tgt, _ = sf.read(tgt_audio_path, start=int(start), stop=int(end), dtype='float32')
            if a_tgt.size == 0:
                raise ValueError(f"Empty target audio data. Path: {tgt_audio_path}, Start: {start}, End: {end}")

            # Load interfering audio
            int_audio_path = self.audio_direc + line[6]
            start = float(line[7]) * self.audio_sr
            end = start + min_length_audio
            if not os.path.exists(int_audio_path):
                raise FileNotFoundError(f"Interfering audio file not found: {int_audio_path}")
            a_int, _ = sf.read(int_audio_path, start=int(start), stop=int(end), dtype='float32')
            if a_int.size == 0:
                raise ValueError(f"Empty interfering audio data. Path: {int_audio_path}, Start: {start}, End: {end}")

            # Training SNR augmentation
            if float(line[8]) != 0:
                target_power = np.linalg.norm(a_tgt, 2)**2 / a_tgt.size
                intef_power = np.linalg.norm(a_int, 2)**2 / a_int.size
                a_int *= np.sqrt(target_power / intef_power)
                snr_1 = (10**(float(line[8]) / 20))

                max_snr = max(1, snr_1)
                a_tgt /= max_snr
                a_int /= max_snr
                a_int = a_int * snr_1

            a_mix = a_tgt + a_int

            # Audio normalization
            max_val = np.max(np.abs(a_mix))
            if max_val > 1:
                a_mix /= max_val
                a_tgt /= max_val

            mix_audios.append(a_mix)
            tgt_audios.append(a_tgt)
            tgt_eegs.append(eeg_data)

        return np.asarray(mix_audios, dtype=np.float32), np.asarray(tgt_audios, dtype=np.float32), np.asarray(tgt_eegs, dtype=np.float32)

    def get_eeg(self, subject, trial):
        return self.eeg_dict[(subject, trial)][0]

    def __len__(self):
        return len(self.minibatch)


class dataset_eeg_mp(dataset_eeg):
    def __init__(self, args, partition, shared_eegs, mix_lst):
        self.minibatch =[]
        self.args = args
        self.partition = partition
        self.max_length = args.max_length
        self.audio_sr=args.audio_sr
        self.ref_sr=args.ref_sr
        self.speaker_no=args.speaker_no
        self.batch_size=args.batch_size
        self.use_cache = False

        self.mix_lst_path = args.mix_lst_path
        self.audio_direc = args.audio_direc
        self.eeg_direc = args.reference_direc
        
        start = 0
        while True:
            end = min(len(mix_lst), start + self.batch_size)
            self.minibatch.append(mix_lst[start:end])
            if end == len(mix_lst):
                break
            start = end

        self.eeg_dict = {(s, t): (shared, shape) for s, t, shared, shape in shared_eegs}
        
    def get_eeg(self, subject, trial):
        shared_array, shape = self.eeg_dict[(subject, trial)]
        eeg_data = np.ctypeslib.as_array(shared_array.get_obj()).reshape(shape)
        return eeg_data