try:
    import torchaudio
    import os
    from tqdm import tqdm
    import torch
    import librosa
    import numpy as np
    import matplotlib.pyplot as plt
    from torch.utils.data import Dataset
    from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
except:
    pass
class Frequency():
    '''
    An interface for transforming audio files into frequency domain representations.
    '''
    def __init__(self, sample_rate=16000, duration=1, augment=None, type='spec', 
                 nmels=70, n_fft=720, hop_length=360, is_log=True, **kwargs):
        
        '''
        Initialize the frequency transformer. 
        
        :param sample_rate: The sample rate of the audio files.
        :type sample_rate: int
        :param duration: The duration of the audio files in seconds.
        :type duration: int
        :param augment: The audio augmentations to apply to the audio files.
        :type augment: audiomentations.Compose
        :param type: The type of frequency domain representation to use. Can be 'spec' for spectrogram or 'mfcc' for Mel-frequency cepstral coefficients.
        :type type: str
        :param nmels: The number of mel bins to use for the Mel-frequency cepstral coefficients.
        :type nmels: int
        :param n_fft: The number of samples to use for each frame of the spectrogram.
        :type n_fft: int
        :param hop_length: The number of samples to shift the window by between frames of the spectrogram.
        :type hop_length: int
        :param is_log: Whether to use a log scale for the spectrogram.
        :type is_log: bool
        :param kwargs: Keyword arguments to be passed to the frequency domain transformer.
        
        '''

        self.sample_rate = sample_rate
        self.duration = duration
        self.emb_dim = 768 * int(49 * self.duration)         # fact regarding wav2vec2-base-960h
        if augment is None:
            self.augment = Compose([
            TimeStretch(min_rate=0.8, max_rate=1.4, p=0.7),
            PitchShift(min_semitones=-4, max_semitones=1, p=0.8),
            Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
            AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
        ])
        else:
            self.augment = augment
        
        if type == 'spectrogram':
            self.transform_func = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=nmels, **kwargs)
        elif type == 'mfcc':
            self.transform_func = torchaudio.transforms.MFCC(sample_rate=self.sample_rate, n_mfcc=nmels, melkwargs={'n_fft':n_fft, 'hop_length':hop_length}, **kwargs)
        else:   raise ValueError(f"Invalid type of transformation {type}. Expected spectrogram or mfcc")
        
        self.is_log = is_log

    def transform_list(self, words, n=4):
        '''
        Given a folder dataset with folders each containing audio files, this returns a table of spectra in the form of a numpy array X and a table of classes in the form of a numpy array y.
        
        :param words: A list of words which are the classes of the speech classifier.
        :type words: list
        :param n: The number of times to augment each audio file.
        :type n: int
        
        :return: A tuple of the form (X, y) where X is a 3D numpy array representing the audio files and y is a 1D numpy array representing the classes of the audio files.
        :rtype: tuple of numpy.ndarray
        '''        
        # may be needed in the future (draft)
        sounds_per_word = len(os.listdir(f"dataset/{words[0]}"))
        self.N = len(words) * sounds_per_word
        
        # main function
        x_data, y_data = [], []
        for word in tqdm(words):
            for i, file in enumerate(os.listdir(f"dataset/{word}")):
                if not file.endswith(".wav"): continue
                waveform, sr = torchaudio.load(f"dataset/{word}/{file}")
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

                if waveform.shape[0] == 2:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                length = waveform.shape[1]
                sample_dur = int(self.duration * self.sample_rate)
                if length < sample_dur:
                    waveform = torch.cat((waveform, torch.zeros(1, sample_dur - length)), dim=1)
                elif length > sample_dur:
                    waveform = waveform[:, :sample_dur]     
                
                waveform = waveform.numpy()
                if n > 0:
                    for _ in range(n):
                        waveform = self.augment(waveform, sample_rate=self.sample_rate)
                        spectrum = self.transform_func(torch.from_numpy(waveform))
                        spectrum = librosa.power_to_db(spectrum[0]) if self.is_log else spectrum[0]
                        # transpose to get (time, freq) instead of (freq, time)
                        spectrum = spectrum.T
                        x_data.append(spectrum)
                        y_data.append(words.index(word))                
                else:
                    spectrum = self.transform_func(torch.from_numpy(waveform))
                    spectrum = librosa.power_to_db(spectrum[0]) if self.is_log else spectrum[0]
                    # transpose to get (time, freq) instead of (freq, time)
                    spectrum = spectrum.T
                    x_data.append(spectrum)
                    y_data.append(words.index(word))
                    
        x_data, y_data = np.array(x_data), np.array(y_data)
        
        return x_data, y_data


    
    def transform(self, path, strict_duration=False):
        '''
        Convert the audio file given in path into a frequency domain representation.
        
        :param path: The path to the audio file.
        :type path: str
        :param strict_duration: Whether to strictly use the duration specified during init or not. If True, then the audio file is padded with zeros if it is shorter than the duration and truncated if it is longer than the duration.
        :type strict_duration: bool
        
        :return: The frequency domain representation of the audio file as a 2D numpy array.
        :rtype: numpy.ndarray
        '''
        waveform, sr = torchaudio.load(path)
        waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        if waveform.shape[0] == 2:  waveform = torch.mean(waveform, dim=0, keepdim=True)
        length = waveform.shape[1]
        if strict_duration:
            sample_dur = int(self.duration * self.sample_rate)
            if length < sample_dur:
                waveform = torch.cat((waveform, torch.zeros(1, sample_dur - length)), dim=1)
            elif length > sample_dur:
                waveform = waveform[:, :sample_dur]
            
        spectrum = self.transform_func(waveform)
        spectrum = librosa.power_to_db(spectrum[0].numpy()) if self.is_log else spectrum[0].numpy()
        spectrum = spectrum.T
        spectrum = spectrum[np.newaxis, ...]
        return spectrum
        


