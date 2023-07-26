try:
    import os
    from botiverse.models import TTS
    from playsound import playsound
    from botiverse.models import LSTMClassifier
    from botiverse.preprocessors import Vocalize, Wav2Vec,  Frequency
    from botiverse.bots.VoiceBot.utils import voice_input
except:
    pass

class SpeechClassifier():
    '''
    An interface for the speech classifier chatbot which classifies speech into one of a set of classes. Suitable when the
    number of classes is small and the words are easily pronounceable.
    '''
    def __init__(self, words, samplerate, duration, repr='wav2vec', machine='lstm', **kwargs):
        '''
        Initialize the dataset and its transformation for the speech classification process.
        
        :param words: A list of words which are the classes of the speech classifier.
        :type words: list
        :param samplerate: The sample rate of the audio files.
        :type samplerate: int
        :param duration: The duration of the audio files in milliseconds.
        :type duration: int
        :param repr: The representation to use for the audio files. Can be 'wav2vec', 'mfcc', 'spectrogram' or a custom representation
        :type repr: str or object
        :param machine: The machine learning model to use for classification. Can be 'lstm' or a custom model.
        :type machine: str or object
        '''
        self.words = words
        self.samplerate = samplerate
        self.duration = duration
        self.machine = machine
        if repr == 'wav2vec':
            self.transformer = Wav2Vec(samplerate, duration)
        elif repr == 'mfcc':
            self.transformer = Frequency(type='mfcc', sample_rate=samplerate, duration=duration, **kwargs)
        elif repr == 'spectrogram':
            self.transformer = Frequency(type='spectrogram', sample_rate=samplerate, duration=duration, **kwargs)
        elif type(repr) != str:
            self.transformer = repr
        else:
            raise ValueError(f"Invalid representation {repr}. Expected wav2vec, mfcc or spectrogram.")


    def generate_read_data(self, n=3, regenerate=False, force_download_noise=False, **kwargs):
        '''
        Generate synthetic audio data for the words specified during init and then corrupt it with noise and audio transformations.
        
        :param n: The number of audio files to generate for each word using audio transformations.
        :type n: int
        :param regenerate: Whether to regenerate the dataset even if it already exists.
        :type regenerate: bool
        :param force_download_noise: Whether to force download the noise dataset even if it already exists.
        :type force_download_noise: bool
        :param kwargs: Keyword arguments to be passed to the transformer (that puts audio in the chosen representation).
        
        :return: A tuple of the form (X, y) where X is a 3D numpy array representing the audio files and y is a 1D numpy array representing the classes of the audio files.
        :rtype: tuple of numpy.ndarray
        '''
        # if there is no dataset folder or if the regenerate flag is set, generate the dataset
        if regenerate or not os.path.exists('dataset'):
            V = Vocalize(self.words)
            Vocalize.corrupt_dataset(self.words, sample_rate=self.samplerate, force_download=force_download_noise)
        X, y = self.transformer.transform_list(self.words, n, **kwargs)
        return X, y

    def fit(self, X, y,  λ=0.001, α=0.01, hidden=128, patience=50, max_epochs=600, **kwargs):
        '''
        Train the speech classifier model.
        
        :param X: A 3D numpy array representing the audio files.
        :type X: numpy.ndarray
        :param y: A 1D numpy array representing the classes of the audio files.
        :type y: numpy.ndarray
        :param λ: The learning rate parameter.
        :type λ: float
        :param α: The regularization parameter.
        :type α: float
        :param hidden: The number of hidden units in the LSTM layer.
        :type hidden: int
        :param patience: The number of bad epochs to wait before early stopping.
        :type patience: int
        :param max_epochs: The maximum number of epochs to train for.
        :type max_epochs: int
        :param kwargs: Keyword arguments to be passed to the model's fit method.
        '''
        if self.machine == 'lstm':
            self.model = LSTMClassifier(X.shape[2], hidden, len(self.words))
            self.model.fit(X, y,  λ, α, max_epochs, patience, **kwargs)
        elif type(self.machine) != str:
            self.model = self.machine
            self.model.fit(X, y, **kwargs)
        else:
            raise ValueError(f"Invalid machine {self.machine}. Expected lstm or a custom model.")
    
    def save(self, path):
        '''
        Save the model to a file.
        
        :param path: The path to the file
        '''
        self.model.save(path+'.bot')
    
    def load(self, path, **kwargs):
        '''
        Load the model from a file.
        
        :param path: The path to the file
        :param kwargs: Keyword arguments to be passed to the model's load method.
        '''
        if self.machine == 'lstm':
            self.model = LSTMClassifier(**kwargs)
            self.model.load(path + '.bot')
        else:
            self.model = self.machine
            self.model.load(path + '.bot')
            
        
    
    def predict(self, path, index=False):
        '''
        Predict the class of the audio file at the given path.
        
        :param path: The path to the audio file to be classified.
        :type path: str
        :param index: Whether to return the index of the class or the class itself.
        :type index: bool
        
        :return: The class of the audio file at the given path.
        :rtype: str or int
        '''
        vec = self.transformer.transform(path, strict_duration=False)
        pred, prob = self.model.predict(vec)
        pred, prob = pred[0], prob[0]
        print("Probability of prediction: ", prob)
        return (self.words[pred], prob) if not index else (pred, prob)
