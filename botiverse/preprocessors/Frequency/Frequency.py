import torchaudio
import os
from tqdm import tqdm
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift

class Frequency():
    '''
    An interface for transforming audio files into frequency domain representations.
    '''
    def __init__(self, sample_rate=16000, duration=1, augment=None, type='spec', 
                 nmels=70, n_fft=720, hop_length=360, is_log=True, **kwargs):

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
        :param: words: A list of words which are the classes of the speech classifier.
        :param: n: The number of times to augment each audio file.
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
        



'''
def plot_freq_domains(words, spectograms, num_pos=SOUNDS_PER_WORD):
    fig, axs = plt.subplots(len(words), num_pos, figsize=(15, 4*len(words)))
    for i, word in enumerate(words):
        for j, (prop, mel) in enumerate(spectograms[word].items()):
            axs[i, j].imshow(mel, origin="lower", aspect="auto")
            axs[i, j].set_title(f"{word.upper()}-{prop}")        
            if j==0:   
                axs[i, j].set_ylabel("freq_bin")
                axs[i, j].set_xlabel("time_window")
        fig.subplots_adjust(hspace=0.5)
    plt.show(block=False)
    

def random_waveform(words, mel_transform, mfcc_transform):
    # select random word from words
    word = np.random.choice(words)
    # select random audio file from word folder
    file = np.random.choice(os.listdir('dataset/' + word))
    # load audio file
    waveform, sample_rate = torchaudio.load('dataset/' + word + '/' + file)
    mel_specgram = mel_transform(waveform)
    mfcc_specgram = mfcc_transform(waveform)
    # plot them side by side
    plt.rcParams['figure.dpi'] = 125
    plt.style.use('dark_background')
    fig, axs = plt.subplots(1, 2, figsize=(15, 4))
    axs[0].imshow(librosa.power_to_db(mel_specgram[0].numpy()), origin="lower", aspect="auto")
    axs[0].set_title("Mel Spectrogram")
    axs[0].set_ylabel("freq_bin")
    axs[0].set_xlabel("time_window")
    axs[1].imshow(librosa.power_to_db(mfcc_specgram[0].numpy()), origin="lower", aspect="auto")
    axs[1].set_title("MFCC")
    axs[1].set_ylabel("freq_bin")
    axs[1].set_xlabel("time_window")
    fig.suptitle(f"\'{word.upper()}\'")
    plt.show(block=False)
    print("Shape of waveform: {}".format(waveform.size()))
    print("Sample rate of waveform: {}".format(sample_rate))



class Spectograms(Dataset):
    def __init__(self, x_data, y_data, is_train=True, transform=None, normalize=False):
        self.is_train = is_train
        self.transform = transform
        self.num_classes = len(np.unique(y_data))
        
        # split the data into train and test
        self.x_train, self.x_val, self.y_train, self.y_val = \
            train_test_split(x_data, y_data, test_size=TEST_SIZE, random_state=1, stratify=y_data)
            
        if normalize:
            self.μ, self.σ = np.mean(self.x_train), np.std(self.x_train)
            self.x_train = (self.x_train - self.μ) / self.σ
            self.x_val = (self.x_val - self.μ) / self.σ
            
    def __len__(self):
        return len(self.x_train) if self.is_train else len(self.x_val)

    def __getitem__(self, index):
        if self.is_train:
            a, y = self.x_train[index], self.y_train[index]
            a, y = self.x_train[index], self.y_train[index]
            # Get the list of images where self.y_train == y and exclude the anchor "a"
            p_ind = np.where(self.y_train == y)[0]
            # exclude the anchor which has ind as index
            p_ind = np.delete(p_ind, np.argwhere(p_ind == index))
            # pop index of the anchor from the list
            p = self.x_train[p_ind[random.randrange(0, p_ind.shape[0])]]

            n_ind = np.where(self.y_train != y)[0]
            # randomly select np.unique(self.y_train).shape[0] number of images
            n_ind = random.sample(list(n_ind), int(NUM_NEGATIVES*self.num_classes))
            list_n = self.x_train[n_ind]
            # find a random negative image
            #n = random.choice(self.x_train[self.y_train != y])

            if self.transform:
                # convert each image to PIL in list_n
                list_n = [self.transform(n) for n in list_n]
                list_n = torch.stack(list_n)
                a, p = a[...,np.newaxis], p[...,np.newaxis]
                a, p = self.transform(a), self.transform(p)

            return a, p, list_n, y

        else:
            a, y = self.x_val[index], self.y_val[index]
            if self.transform:
                a = a[...,np.newaxis]
                a = self.transform(a)
            return a, y
        
 
def show_next_triplet(train_loader, words):
    a, p, list_n, y = next(iter(train_loader))
    a0, p0, list_n0, y0 = a[0].squeeze(), p[0].squeeze(), list_n[0].squeeze(), y[0].squeeze()
    fig, ax = plt.subplots(1, 2+len(list_n0), figsize=(0.5*(2+len(list_n0))*5, 2))
    ax[0].imshow(a0, origin="lower", aspect="auto")
    ax[0].set_title(f'Anchor: {words[y[0]]}')
    ax[1].imshow(p0, origin="lower", aspect="auto")
    ax[1].set_title('Positive')
    for i in range(len(list_n0)):
        ax[i+2].imshow(list_n0[i].squeeze(), origin="lower", aspect="auto")
        ax[i+2].set_title(f'Negative')
    plt.show()
    
'''