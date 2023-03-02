import h5py
import librosa
import yaml
import playsound
import torch
import os
import numpy as np
from pathlib import Path
from dotmap import DotMap
from sentence_transformers import util
from torch.utils.data import Dataset

import sys
sys.path.append(os.path.abspath('../audio-text_retrieval'))
from models_atr.ASE_model import ASE


class AudioDataset(Dataset):
    def __init__(self, valid_idxs=None, name='AudioCaps'):
        """
        Load audio clip's waveform.
        Args:
            name: 'AudioCaps', 'Clotho
        """
        super(AudioDataset, self).__init__()
        self.name = name
        audio_dir_prefix = f'../audio-text_retrieval/data/{name}/waveforms'
        audio_dirs = [f'{audio_dir_prefix}/train/', f'{audio_dir_prefix}/test/', f'{audio_dir_prefix}/val/']
        self.audio_paths = [os.path.join(audio_dir, f) for audio_dir in audio_dirs for f in os.listdir(audio_dir)]
        self.audio_names = [Path(audio_path).stem for audio_path in self.audio_paths]
        if valid_idxs is not None:
            self.audio_paths = [self.audio_paths[idx] for idx in valid_idxs]
            self.audio_names = [self.audio_names[idx] for idx in valid_idxs]

    def __len__(self):
        return len(self.audio_names)

    def __getitem__(self, idx):
        audio_name = self.audio_names[idx]
        audio_path = self.audio_paths[idx]
        audio, _ = librosa.load(self.audio_paths[idx], sr=32000, mono=True)
        audio = AudioDataset.pad_or_truncate(audio, 32000 * 10)
        return audio, audio_name, audio_path, idx, len(audio)

    @staticmethod
    def pad_or_truncate(audio, audio_length):
        """Pad all audio to specific length."""
        length = len(audio)
        if length <= audio_length:
            return np.concatenate((audio, np.zeros(audio_length - length)), axis=0)
        else:
            return audio[:audio_length]


def load_audio_model(device):
    with open('../audio-text_retrieval/settings/settings.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = DotMap(config)

    model = ASE(config, is_module=True).to(device)

    model_path = '../audio-text_retrieval/outputs/exp_name_data_AudioCaps_freeze_True_lr_0.0001_margin_0.2_seed_20/models/best_model.pth'
    print(f'Loading model from {model_path}.')
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model'])

    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_audio_model(device)

    audio_dataset = AudioDataset()
    audio, audio_name, audio_path, *_ = audio_dataset[0]
    audio = torch.from_numpy(audio).unsqueeze(0).to(device)
    text = 'any nl text'
    audio_embed, caption_embed = model(audio, text)
    print(audio_embed)
    print(caption_embed)
    sim = util.cos_sim(audio_embed, caption_embed)
    score = sim.item()
    print(f'{audio_name}: {score}')
    playsound.playsound(audio_path)

"""
    train_loader = get_dataloader('train', config, is_module=True)
    batch_data = next(iter(train_loader))
    
    # for batch_id, batch_data in tqdm(enumerate(train_loader), total=len(train_loader)):
    audios, captions, audio_ids, _ = batch_data
    audios = audios.to(device)
    audio_ids = audio_ids.to(device)
    audio_embeds, caption_embeds = model(audios, captions)
    scores = util.cos_sim(audio_embeds, caption_embeds)  # (batch_size, x batch_size)
"""










