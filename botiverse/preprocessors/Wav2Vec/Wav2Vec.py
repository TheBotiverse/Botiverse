from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import os
import torch
import torchaudio
from tqdm import tqdm
import numpy as np

class Wav2Vec():
    '''
    An interface for transforming audio files into wav2vec vectors.
    '''
    def __init__(self, sample_rate=16000, duration=16000, augment= None):
        '''
        Initialize the Wav2Vec transformer by loading the wav2vec model and setting the sample rate and duration of the audio files.
        :param: sample_rate: The sample rate of the audio files
        :param: duration: The duration of the audio files in milliseconds        
        '''
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")
        self.sample_rate = sample_rate
        self.duration = duration
        self.emb_dim = 768 * 49 * self.duration // 16000        # fact regarding wav2vec2-base-960h
        if augment is None:
            self.augment = Compose([
            TimeStretch(min_rate=0.9, max_rate=1.4, p=0.8),
            PitchShift(min_semitones=-4, max_semitones=1, p=0.7),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
        ])
        else:
            self.augment = augment
            

    def compute_vecs(self, words, n=4):
        '''
        Given a folder dataset with folders each containing audio files, this returns a table of wav2vec vectors (one for each audio file) in the form of a numpy array X and a table of classes in the form of a numpy array y.
        Note that in the process, each audio file is augmented n times and each corresponds to another wav2vec vector.
        '''
        sounds_per_word = len(os.listdir(f"dataset/{words[0]}"))
        self.N = len(words) * sounds_per_word
        X, y = [], []
        for word in tqdm(words):
            for i, file in enumerate(os.listdir(f"dataset/{word}")):
                waveform, sr = torchaudio.load(f"dataset/{word}/{file}")
                # resample if sr != 24K
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
                
                if waveform.shape[0] == 2:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                length = waveform.shape[1]
                
                if length < self.duration:
                    waveform = torch.cat((waveform, torch.zeros(1, self.duration - length)), dim=1)
                elif length > self.duration:
                    waveform = waveform[:, :self.duration]            

                waveform = waveform.squeeze()
                waveform = waveform.detach().numpy()

                for _ in range(n):
                    waveform = self.augment(samples=waveform, sample_rate=self.sample_rate)
                    inputs = self.extractor(waveform, return_tensors="pt", padding=True, sampling_rate=self.sample_rate)
                    features = self.model(inputs.input_values).last_hidden_state
                    features = features.squeeze().detach().numpy()
                    X.append(features)
                    y.append(words.index(word))
        X = np.array(X)
        y = np.array(y)
        return X, y
    
    def get_vec(self, strict_duration=False):
        '''
        Convert the audio file sample.wav into a wav2vec vector.
        '''
        waveform, sr = torchaudio.load(f'sample.wav')
        waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        if waveform.shape[0] == 2:  waveform = torch.mean(waveform, dim=0, keepdim=True)
        length = waveform.shape[1]
        if strict_duration:
            if length < self.duration: waveform = torch.cat((waveform, torch.zeros(1, self.duration - length)), dim=1)
            elif length > self.duration: waveform = waveform[:, :self.duration]
        waveform = waveform.squeeze()
        inputs = self.extractor(waveform, return_tensors="pt", padding=True, sampling_rate=self.sample_rate)
        features = self.model(inputs.input_values).last_hidden_state
        return features