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
import tqdm

def load_shared_eegs(args, partition='train'):
    mix_lst=open(args.mix_lst_path).read().splitlines()
    mix_lst=list(filter(lambda x: x.split(',')[0]==partition, mix_lst))#[:200]
    mix_lst = sorted(mix_lst, key=lambda data: float(data.split(',')[-1]), reverse=True)
    trial_list = set([tuple(line.split(",")[1:3]) for line in mix_lst])
    eeg_list = []
    for i, (subject, trial) in tqdm.tqdm(enumerate(trial_list), total=len(trial_list), desc=f"Loading EEG data for {partition} partition"):
        eeg_path = f'{args.reference_direc}S{subject}Tra{trial}.npy'
        eeg_data = np.load(eeg_path)
        shared_eeg = mp.Array('f', eeg_data.flatten())  # Flatten for 1D shared array
        eeg_list.append((int(subject), int(trial), shared_eeg, eeg_data.shape))  # Store shape for reshaping
    return eeg_list, mix_lst

def load_shared_eegs_pd(args, partition='train'):
    mix_dtypes = {
        'split': str,
        'subject': int,
        'trial': int,
        'tgt_audio': str,
        'tgt_start': float,
        '': str,
        'int_audio': str,
        'int_start': float,
        'snr': float,
        'length': float
    }
    mix_lst = pd.read_csv(args.mix_lst_path, header=None, dtype=mix_dtypes, engine='c', names=mix_dtypes.keys())
    mix_lst = mix_lst[mix_lst["split"] == partition]
    mix_lst = mix_lst.sort_values(by="length", ascending=False)
    trial_list = set(zip(mix_lst["subject"], mix_lst["trial"]))
    eeg_list = []
    for subject, trial in tqdm.tqdm(trial_list, desc=f"Loading EEG data for {partition} partition"):
        eeg_path = f'{args.reference_direc}S{subject}Tra{trial}.npy'
        eeg_data = np.load(eeg_path)
        shared_eeg = mp.Array('f', eeg_data.flatten())  # Flatten for 1D shared array
        eeg_list.append((int(subject), int(trial), shared_eeg, eeg_data.shape))  # Store shape for reshaping
    return eeg_list, mix_lst

def load_shared_eegs_contrastive(args, partition='train'):
    mix_lst = pd.read_csv(args.mix_lst_path)
    mix_lst = mix_lst[mix_lst["split"] == partition]
    mix_lst = mix_lst.sort_values(by="length", ascending=False)
    if args.pretraining_type == 'subject_contrastive':
        trial_list = set(zip(mix_lst["subject_1"], mix_lst["trial_1"])).union(set(zip(mix_lst["subject_2"], mix_lst["trial_2"])))
    elif args.pretraining_type == 'interference_contrastive':
        trial_list = set(zip(mix_lst["subject_1"], mix_lst["trial_1"])).union(set(zip(mix_lst["subject_1"], mix_lst["trial_2"])))
    else:
        raise ValueError(f"Unknown pretraining_type: {args.pretraining_type}")
    eeg_list = []
    for subject, trial in tqdm.tqdm(trial_list, desc=f"Loading EEG data for {partition} partition"):
        eeg_path = f'{args.reference_direc}S{subject}Tra{trial}.npy'
        eeg_data = np.load(eeg_path)
        shared_eeg = mp.Array('f', eeg_data.flatten())  # Flatten for 1D shared array
        eeg_list.append((int(subject), int(trial), shared_eeg, eeg_data.shape))  # Store shape for reshaping
    return eeg_list, mix_lst

def get_dataloader_eeg(args, partition):
    if args.contrastive:
        print("Using contrastive EEG dataset")
        shared_eegs, mix_lst = load_shared_eegs_contrastive(args, partition=partition)
        print(f"Loaded {len(shared_eegs)} EEG datasets into shared memory for multiprocessing.")
        datasets = dataset_eeg_contrastive(args, partition, shared_eegs, mix_lst)
    else:    
        if args.num_workers > 0:
            shared_eegs, mix_lst = load_shared_eegs_pd(args, partition=partition)
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
        print("fetching normal sample")
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
        
        mix_lst = mix_lst[mix_lst["split"] == partition]
        mix_lst = mix_lst.sort_values(by="length", ascending=False)
        
        start = 0
        while True:
            end = min(len(mix_lst), start + self.batch_size)
            self.minibatch.append(mix_lst[start:end])
            if end == len(mix_lst):
                break
            start = end

        self.eeg_dict = {(s, t): (shared, shape) for s, t, shared, shape in shared_eegs}
        
    def __getitem__(self, index):
        mix_audios = []
        tgt_audios = []
        tgt_eegs = []

        batch_lst = self.minibatch[index]
        min_length_second = float(batch_lst.iloc[-1]["length"])  # truncate to the shortest utterance in the batch
        min_length_eeg = math.floor(min_length_second * self.ref_sr)
        min_length_audio = math.floor(min_length_second * self.audio_sr)
        min_length_eeg = min(min_length_eeg, self.max_length * self.ref_sr)
        min_length_audio = min(min_length_audio, self.max_length * self.audio_sr)

        for _, line_cache in batch_lst.iterrows():

            # Load target EEG
            subject, trial = line_cache["subject"], line_cache["trial"]
            eeg_data = self.get_eeg(int(subject), int(trial))
            eeg_start = int(float(line_cache["tgt_start"]) * self.ref_sr)
            eeg_end = eeg_start + min_length_eeg
            eeg_data = eeg_data[eeg_start:eeg_end, :]

            # Load target audio
            tgt_audio_path = self.audio_direc + line_cache["tgt_audio"]
            start = float(line_cache["tgt_start"]) * self.audio_sr
            end = start + min_length_audio
            if not os.path.exists(tgt_audio_path):
                raise FileNotFoundError(f"Target audio file not found: {tgt_audio_path}")
            a_tgt, _ = sf.read(tgt_audio_path, start=int(start), stop=int(end), dtype='float32')
            if a_tgt.size == 0:
                raise ValueError(f"Empty target audio data. Path: {tgt_audio_path}, Start: {start}, End: {end}")

            # Load interfering audio
            int_audio_path = self.audio_direc + line_cache["int_audio"]
            start = float(line_cache["int_start"]) * self.audio_sr
            end = start + min_length_audio
            if not os.path.exists(int_audio_path):
                raise FileNotFoundError(f"Interfering audio file not found: {int_audio_path}")
            a_int, _ = sf.read(int_audio_path, start=int(start), stop=int(end), dtype='float32')
            if a_int.size == 0:
                raise ValueError(f"Empty interfering audio data. Path: {int_audio_path}, Start: {start}, End: {end}")

            # Training SNR augmentation
            if float(line_cache['snr']) != 0:
                target_power = np.linalg.norm(a_tgt, 2)**2 / a_tgt.size
                intef_power = np.linalg.norm(a_int, 2)**2 / a_int.size
                a_int *= np.sqrt(target_power / intef_power)
                snr_1 = (10**(float(line_cache['snr']) / 20))
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
        shared_array, shape = self.eeg_dict[(subject, trial)]
        eeg_data = np.ctypeslib.as_array(shared_array.get_obj()).reshape(shape)
        return eeg_data
    
class dataset_eeg_contrastive(dataset_eeg_mp, data.Dataset):
    def __init__(self, args, partition, shared_eegs, mix_lst):
        super().__init__(args, partition, shared_eegs, mix_lst)
        self.pretraining_type = args.pretraining_type
        assert self.pretraining_type in ['subject_contrastive', 'interference_contrastive'], "pretraining_type must be 'subject_contrastive' or 'interference_contrastive'"
        if self.pretraining_type == 'subject_contrastive':
            assert mix_lst.columns.tolist() == ["split", "subject_1", "trial_1", "tgt_audio_1", "tgt_start_1", "int_audio", "int_start", "subject_2", "trial_2", "tgt_audio_2", "tgt_start_2", "type", "snr", "length"]
            self.__getitem__ = self.__getitem__subject__
        elif self.pretraining_type == 'interference_contrastive':
            assert mix_lst.columns.tolist() == ["split", "subject_1", "trial_1", "trial_2", "tgt_audio_1", "tgt_start_1", "int_audio_1", "int_start_1", "int_audio_2", "int_start_2", "tgt_audio_2", "tgt_start_2", "type", "snr", "length"]
            self.__getitem__ = self.__getitem__interference__
        else:
            raise ValueError(f"Unknown pretraining_type: {self.pretraining_type}")

    def __getitem__(self, index):
        if self.pretraining_type == 'subject_contrastive':
            return self.__getitem__subject__(index)
        elif self.pretraining_type == 'interference_contrastive':
            return self.__getitem__interference__(index)
        else:
            raise ValueError(f"Unknown pretraining_type: {self.pretraining_type}")

    def __getitem__subject__(self, index):
        base_audios = []
        pair_audios = []
        base_eegs = []
        pair_eegs = []
        pair_types = []
        
        batch_lst = self.minibatch[index]
        min_length_second = float(batch_lst.iloc[-1]["length"])  # truncate to the shortest utterance in the batch
        min_length_eeg = math.floor(min_length_second * self.ref_sr)
        min_length_audio = math.floor(min_length_second * self.audio_sr)
        min_length_eeg = min(min_length_eeg, self.max_length * self.ref_sr)
        min_length_audio = min(min_length_audio, self.max_length * self.audio_sr)
        
        for i, line in batch_lst.iterrows():

            # Load target EEG
            subject1, trial1 = line["subject_1"], line["trial_1"]
            eeg_data1 = self.get_eeg(int(subject1), int(trial1))
            eeg_start1 = int(float(line["tgt_start_1"]) * self.ref_sr)
            eeg_end1 = eeg_start1 + min_length_eeg
            eeg_data1 = eeg_data1[eeg_start1:eeg_end1, :]
            
            # Load paired EEG from a different subject with the same audio mix
            subject2, trial2 = line["subject_2"], line["trial_2"]
            eeg_data2 = self.get_eeg(int(subject2), int(trial2))
            eeg_start2 = int(float(line["int_start"]) * self.ref_sr)
            eeg_end2 = eeg_start2 + min_length_eeg
            eeg_data2 = eeg_data2[eeg_start2:eeg_end2, :]

            # Load target audio
            tgt_audio_1_path = self.audio_direc + line["tgt_audio_1"]
            start = float(line["tgt_start_1"]) * self.audio_sr
            end = start + min_length_audio
            if not os.path.exists(tgt_audio_1_path):
                raise FileNotFoundError(f"Target audio file not found: {tgt_audio_1_path}")
            a_tgt_1, _ = sf.read(tgt_audio_1_path, start=int(start), stop=int(end), dtype='float32')
            if a_tgt_1.size == 0:
                raise ValueError(f"Empty target audio data. Path: {tgt_audio_1_path}, Start: {start}, End: {end}")
            
            # Load paired target audio
            tgt_audio_2_path = self.audio_direc + line["tgt_audio_2"]
            start = float(line["tgt_start_2"]) * self.audio_sr
            end = start + min_length_audio
            if not os.path.exists(tgt_audio_2_path):
                raise FileNotFoundError(f"Target audio file not found: {tgt_audio_2_path}")
            a_tgt_2, _ = sf.read(tgt_audio_2_path, start=int(start), stop=int(end), dtype='float32')
            if a_tgt_2.size == 0:
                raise ValueError(f"Empty target audio data. Path: {tgt_audio_2_path}, Start: {start}, End: {end}")

            # Load interfering audio
            int_audio_path = self.audio_direc + line["int_audio"]
            start = float(line["int_start"]) * self.audio_sr
            end = start + min_length_audio
            if not os.path.exists(int_audio_path):
                raise FileNotFoundError(f"Interfering audio file not found: {int_audio_path}")
            a_int, _ = sf.read(int_audio_path, start=int(start), stop=int(end), dtype='float32')
            if a_int.size == 0:
                raise ValueError(f"Empty interfering audio data. Path: {int_audio_path}, Start: {start}, End: {end}")
            
            a_mix_1 = a_tgt_1 + a_int
            a_mix_2 = a_tgt_2 + a_int

            # Training SNR augmentation
            if float(line["snr"]) != 0:
                target_power = np.linalg.norm(a_tgt_1, 2)**2 / a_tgt_1.size
                intef_power = np.linalg.norm(a_int, 2)**2 / a_int.size
                scaled_a_int *= np.sqrt(target_power / intef_power)
                snr_1 = (10**(float(line["snr"]) / 20))

                max_snr = max(1, snr_1)
                a_tgt_1 /= max_snr
                scaled_a_int /= max_snr
                scaled_a_int_1 = scaled_a_int * snr_1

                target_power = np.linalg.norm(a_tgt_2, 2)**2 / a_tgt_2.size
                scaled_a_int *= np.sqrt(target_power / intef_power)

                max_snr = max(1, snr_1)
                a_tgt_2 /= max_snr
                scaled_a_int /= max_snr
                scaled_a_int_2 = scaled_a_int * snr_1

                a_mix_1 = a_tgt_1 + scaled_a_int_1
                a_mix_2 = a_tgt_2 + scaled_a_int_2

            # Audio normalization
            max_val_1 = np.max(np.abs(a_mix_1))
            if max_val_1 > 1:
                a_mix_1 /= max_val_1

            max_val_2 = np.max(np.abs(a_mix_2))
            if max_val_2 > 1:
                a_mix_2 /= max_val_2
                
            base_audios.append(a_mix_1)
            pair_audios.append(a_mix_2)
            base_eegs.append(eeg_data1)
            pair_eegs.append(eeg_data2)
            pair_types.append(line["type"])
        
        eegs = base_eegs + pair_eegs
        audios = base_audios + pair_audios

        return np.asarray(audios, dtype=np.float32), np.asarray(eegs, dtype=np.float32), np.asarray(pair_types, dtype=np.float32)

    def __getitem__interference__(self, index):
        base_audios = []
        pair_audios = []
        base_eegs = []
        pair_eegs = []
        pair_types = []

        batch_lst = self.minibatch[index]
        min_length_second = float(batch_lst[-1]['length'])  # truncate to the shortest utterance in the batch
        min_length_eeg = math.floor(min_length_second * self.ref_sr)
        min_length_audio = math.floor(min_length_second * self.audio_sr)
        min_length_eeg = min(min_length_eeg, self.max_length * self.ref_sr)
        min_length_audio = min(min_length_audio, self.max_length * self.audio_sr)
        
        for line in batch_lst:

            # Load target EEG
            subject1, trial1 = line["subject_1"], line["trial_1"]
            eeg_data1 = self.get_eeg(int(subject1), int(trial1))
            eeg_start1 = int(float(line["tgt_start_1"]) * self.ref_sr)
            eeg_end1 = eeg_start1 + min_length_eeg
            eeg_data1 = eeg_data1[eeg_start1:eeg_end1, :]

            #Load paired EEG (same target, different interference)
            subject2, trial2 = line["subject_1"], line["trial_2"]
            eeg_data2 = self.get_eeg(int(subject2), int(trial2))
            eeg_start2 = int(float(line["tgt_start_2"]) * self.ref_sr)
            eeg_end2 = eeg_start2 + min_length_eeg
            eeg_data2 = eeg_data2[eeg_start2:eeg_end2, :]

            # Load target audio
            tgt_audio_1_path = self.audio_direc + line["tgt_audio_1"]
            start = float(line["tgt_start_1"]) * self.audio_sr
            end = start + min_length_audio
            if not os.path.exists(tgt_audio_1_path):
                raise FileNotFoundError(f"Target audio file not found: {tgt_audio_1_path}")
            a_tgt_1, _ = sf.read(tgt_audio_1_path, start=int(start), stop=int(end), dtype='float32')
            if a_tgt_1.size == 0:
                raise ValueError(f"Empty target audio data. Path: {tgt_audio_1_path}, Start: {start}, End: {end}")

            # Load paired target audio
            tgt_audio_2_path = self.audio_direc + line["tgt_audio_2"]
            start = float(line["tgt_start_2"]) * self.audio_sr
            end = start + min_length_audio
            if not os.path.exists(tgt_audio_2_path):
                raise FileNotFoundError(f"Target audio file not found: {tgt_audio_2_path}")
            a_tgt_2, _ = sf.read(tgt_audio_2_path, start=int(start), stop=int(end), dtype='float32')
            if a_tgt_2.size == 0:
                raise ValueError(f"Empty target audio data. Path: {tgt_audio_2_path}, Start: {start}, End: {end}")

            # Load interfering audio
            int_audio_1_path = self.audio_direc + line["int_audio_1"]
            start = float(line["int_start_1"]) * self.audio_sr
            end = start + min_length_audio
            if not os.path.exists(int_audio_1_path):
                raise FileNotFoundError(f"Interfering audio file not found: {int_audio_1_path}")
            a_int_1, _ = sf.read(int_audio_1_path, start=int(start), stop=int(end), dtype='float32')
            if a_int_1.size == 0:
                raise ValueError(f"Empty interfering audio data. Path: {int_audio_1_path}, Start: {start}, End: {end}")
            
            # Load paired interference audio
            int_audio_2_path = self.audio_direc + line["int_audio_2"]
            start = float(line["int_start_2"]) * self.audio_sr
            end = start + min_length_audio
            if not os.path.exists(int_audio_2_path):
                raise FileNotFoundError(f"Target audio file not found: {int_audio_2_path}")
            a_int_2, _ = sf.read(int_audio_2_path, start=int(start), stop=int(end), dtype='float32')
            if a_int_2.size == 0:
                raise ValueError(f"Empty target audio data. Path: {int_audio_2_path}, Start: {start}, End: {end}")

            a_mix_1 = a_tgt_1 + a_int_1
            a_mix_2 = a_tgt_2 + a_int_2

            # Training SNR augmentation
            if float(line["snr"]) != 0:
                target_power = np.linalg.norm(a_tgt_1, 2)**2 / a_tgt_1.size
                intef_power = np.linalg.norm(a_int_1, 2)**2 / a_int_1.size
                scaled_a_int_1 *= np.sqrt(target_power / intef_power)
                snr_1 = (10**(float(line["snr"]) / 20))

                max_snr = max(1, snr_1)
                a_tgt_1 /= max_snr
                scaled_a_int_1 /= max_snr
                scaled_a_int_1 = scaled_a_int_1 * snr_1

                intef_power = np.linalg.norm(a_int_2, 2)**2 / a_int_2.size
                scaled_a_int_2 *= np.sqrt(target_power / intef_power)

                max_snr = max(1, snr_1)
                a_tgt_2 /= max_snr
                scaled_a_int_2 /= max_snr
                scaled_a_int_2 = scaled_a_int_2 * snr_1

                a_mix_1 = a_tgt_1 + scaled_a_int_1
                a_mix_2 = a_tgt_2 + scaled_a_int_2

            # Audio normalization
            max_val_1 = np.max(np.abs(a_mix_1))
            if max_val_1 > 1:
                a_mix_1 /= max_val_1

            max_val_2 = np.max(np.abs(a_mix_2))
            if max_val_2 > 1:
                a_mix_2 /= max_val_2
                
            base_audios.append(a_mix_1)
            pair_audios.append(a_mix_2)
            base_eegs.append(eeg_data1)
            pair_eegs.append(eeg_data2)
            pair_types.append(line["type"])
        
        eegs = base_eegs + pair_eegs
        audios = base_audios + pair_audios

        return np.asarray(audios, dtype=np.float32), np.asarray(eegs, dtype=np.float32), np.asarray(pair_types, dtype=np.float32)