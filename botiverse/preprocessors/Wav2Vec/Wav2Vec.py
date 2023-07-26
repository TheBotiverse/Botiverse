try:
    from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
    from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor, Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2PhonemeCTCTokenizer
    import os
    import torch
    import torchaudio
    from tqdm import tqdm
    import numpy as np
    # disable warnings from this file
    from transformers import logging
except:
    pass

class Wav2Vec():
    '''
    An interface for transforming audio files into wav2vec vectors.
    '''
    def __init__(self, sample_rate=16000, duration=1, augment=None):
        '''
        Initialize the Wav2Vec transformer by loading the wav2vec model and setting the sample rate and duration of the audio files.
        
        :param sample_rate: The sample rate of the audio files
        :type sample_rate: int
        :param duration: The duration of the audio files in milliseconds   
        :type duration: int
        :param augment: The audio augmentations to apply to the audio files.
        :type augment: audiomentations.Compose     
        '''
        logging.set_verbosity_error()
        self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-base-960h")

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
            
    def transform_list(self, words, n=4):
        '''
        Given a folder dataset with folders each containing audio files, this returns a table of wav2vec vectors (one for each audio file) in the form of a numpy array X and a table of classes in the form of a numpy array y.
        Note that in the process, each audio file is augmented n times and each corresponds to another wav2vec vector.
        
        :param words: A list of words which are the classes of the speech classifier.
        :type words: list
        :param n: The number of times to augment each audio file.
        :type n: int
        
        :return: A tuple of the form (X, y) where X is a 3D numpy array representing the wav2vec vectors and y is a 1D numpy array representing the classes of the audio files.
        '''
        sounds_per_word = len(os.listdir(f"dataset/{words[0]}"))
        self.N = len(words) * sounds_per_word
        X, y = [], []
        print("Transforming audio files into embeddings...")
        for word in tqdm(words):
            for i, file in enumerate(os.listdir(f"dataset/{word}")):
                if not file.endswith(".wav"): continue
                waveform, sr = torchaudio.load(f"dataset/{word}/{file}")
                # resample if sr != 24K
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
                
                if waveform.shape[0] == 2:
                    waveform = torch.mean(waveform, dim=0, keepdim=True)
                
                length = waveform.shape[1]
                
                sample_dur = int(self.duration * self.sample_rate)
                if length < sample_dur:
                    waveform = torch.cat((waveform, torch.zeros(1, sample_dur - length)), dim=1)
                elif length > sample_dur:
                    waveform = waveform[:, :sample_dur]         

                waveform = waveform.squeeze()
                waveform = waveform.detach().numpy()

                if n >0:
                    for _ in range(n):
                        waveform = self.augment(samples=waveform, sample_rate=self.sample_rate)
                        inputs = self.extractor(waveform, return_tensors="pt", padding=True, sampling_rate=self.sample_rate)
                        features = self.model(inputs.input_values).last_hidden_state
                        features = features.squeeze().detach().numpy()
                        X.append(features)
                        y.append(words.index(word))
                else:
                    inputs = self.extractor(waveform, return_tensors="pt", padding=True, sampling_rate=self.sample_rate)
                    features = self.model(inputs.input_values).last_hidden_state
                    features = features.squeeze().detach().numpy()
                    X.append(features)
                    y.append(words.index(word))
        X = np.array(X)
        y = np.array(y)
        return X, y
    
    def transform(self, path, strict_duration=False):
        '''
        Convert the audio file as in the path into a wav2vec vector.
        
        :param path: The path to the audio file
        :type path: str
        :param strict_duration: If True, the audio file is padded or truncated to the duration specified during init.
        :type strict_duration: bool
        
        :return: The wav2vec vector of the audio file as a 2D numpy array.
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
        waveform = waveform.squeeze()
        inputs = self.extractor(waveform, return_tensors="pt", padding=True, sampling_rate=self.sample_rate)
        features = self.model(inputs.input_values).last_hidden_state.detach().numpy()
        return features
    
    

from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC
from transformers import logging

# Load the pre-trained model and tokenizer

class Wav2Text():
    ''' An interface for converting speech files into text using wav2vec2.'''
    def __init__(self):
        '''Load the pre-trained model and tokenizer'''
        logging.set_verbosity_error()
        self.model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        self.tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
        
    def transcribe(self, path):
        '''
        Given a path to a speech file, return the transcription of the speech file.
        
        :param path: The path to the speech wav file
        :type path: str
        
        :return: The transcription of the speech file
        :rtype: str
        '''
        # load audio
        waveform, sample_rate = torchaudio.load(path)
        
        # resample if sr not 16K
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        waveform = waveform.squeeze()

        # preprocess the audio
        input_values = self.tokenizer(waveform, return_tensors="pt").input_values

        # Perform speech-to-text conversion
        with torch.no_grad():
            logits = self.model(input_values).logits

        # Decode the predicted transcription
        predicted_ids = torch.argmax(logits, dim=2)
        transcription = self.tokenizer.batch_decode(predicted_ids)[0]
        return transcription.lower()

