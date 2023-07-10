import numpy as np
import json
from gtts import gTTS
import tempfile
import os
from botiverse.models import TTS
from playsound import playsound

from botiverse.models import LSTMClassifier
from botiverse.preprocessors import Vocalize, Wav2Vec, Wav2Text, BertEmbedder, Frequency
from botiverse.bots.Vocalizer.utils import voice_input



class Vocalizer():
    '''An interface for the vocalizer chatbot which simulates a call with a customer service bot.'''
    def __init__(self, call_json_path):
        ''' 
        Load the call data from a json file.
        :param call_json_path: The path to the json file containing the call state machine.
        '''
        with open(call_json_path, 'r') as file:
            call_json = file.read()
        self.call_data = json.loads(call_json)
        self.current_node = 'A'
        self.wav2text = Wav2Text()
        self.bert_embeddings = BertEmbedder()

    def generate_speech(self, text, offline=False):
        '''Use Google's TTS or offline FastSpeech 1.0 to play speech from the given text.
        :param text: The text to be converted into speech.
        :param offline: Whether to use offline FastSpeech 1.0 to generate speech.
        '''
        if offline:
            tts = TTS()
            tts.speak(text)
        else:
            tts = gTTS(text=text, lang='en', tld="us", slow=False)
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_audio:
                temp_filename = temp_audio.name
                tts.save(temp_filename)
                playsound(temp_filename)

    def simulate_call(self):
        '''
        Simulate a call with a customer service bot as driven by the call state machine.
        '''
        while True:
            if self.current_node == 'Z':
                # the final state has a different structure, bot only speaks and then the call ends
                bot_message = self.call_data[self.current_node]['Bot']
                self.generate_speech(bot_message)
                break

            # 1 - get the current node's data and from that get the message the bot should speak
            node_data = self.call_data[self.current_node]
            bot_message = node_data['Bot']
            self.generate_speech(bot_message)

            # 2 - get the intent options that the bot expects from the user and classify the user's response
            options = node_data['Options']
            intents = [option['Intent'] for option in options]
            max_dur = node_data['max_duration']
            human_resp = voice_input(record_time=int(max_dur))
            human_resp = self.wav2text.transcribe(human_resp)
            selected_ind, score = self.bert_embeddings.closest_sentence(human_resp, intents, retun_ind=True)
            print(f"you said: {human_resp} and the bot decided that you meant {intents[selected_ind]} with a score of {score}")
            
            # 3 - speak according to the chosen option
            speak_message = options[selected_ind]['Speak']
            self.generate_speech(speak_message)

            # 4 - go to the next state
            self.current_node = options[selected_ind]['Next']


class SpeechClassifier():
    '''
    An interface for the speech classifier chatbot which classifies speech into one of a set of classes. Suitable when the
    number of classes is small and the words are easily pronounceable.
    '''
    def __init__(self, words, samplerate, duration, repr='wav2vec', machine='lstm', **kwargs):
        '''
        Initialize the dataset and its transformation for the speech classification process.
        :param words: A list of words which are the classes of the speech classifier.
        :param samplerate: The sample rate of the audio files.
        :param duration: The duration of the audio files in milliseconds.
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
        Generate audio data for the words specified during init.
        :param n: The number of audio files to generate for each word.
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
        :param λ: The learning rate parameter.
        :param α: The regularization parameter.
        :param patience: The number of bad epochs to wait before early stopping.
        :param max_epochs: The maximum number of epochs to train for.
        '''
        print(X.shape)
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
        :param index: Whether to return the index of the class or the class itself.
        :return: The class of the audio file at the given path.
        '''
        vec = self.transformer.transform(path, strict_duration=False)
        pred, prob = self.model.predict(vec)
        pred, prob = pred[0], prob[0]
        print("Probability of prediction: ", prob)
        return (self.words[pred], prob) if not index else (pred, prob)
