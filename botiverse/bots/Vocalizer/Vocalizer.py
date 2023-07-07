import numpy as np
import json
from gtts import gTTS
import tempfile
import random
from botiverse.models import TTS
from playsound import playsound

from botiverse.models import LSTMClassifier
from botiverse.preprocessors import Vocalize, Wav2Vec, Wav2Text, BertEmbedder
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
            print(f"you said: {human_resp} and the bot decided that you meant {intents[selected_ind]}")
            
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
    def __init__(self, words, samplerate, duration):
        '''
        Initialize the dataset and its transformation for the speech classification process.
        :param words: A list of words which are the classes of the speech classifier.
        :param samplerate: The sample rate of the audio files.
        :param duration: The duration of the audio files in milliseconds.
        '''
        self.words = words
        self.samplerate = samplerate
        self.duration = duration
        V = Vocalize(words)
        Vocalize.corrupt_dataset(words)
        
        self.transformer = Wav2Vec(samplerate, duration)
        self.X, self.y = self.transformer.transform_list(words, n=10)

    def fit(self,  λ=0.001, α=0.01, patience=50, max_epochs=600):
        '''
        Train the speech classifier model.
        :param λ: The learning rate parameter.
        :param α: The regularization parameter.
        :param patience: The number of bad epochs to wait before early stopping.
        :param max_epochs: The maximum number of epochs to train for.
        '''
        self.model = LSTMClassifier(self.X.shape[2], 128, len(self.words))
        self.model.fit(self.X, self.y,  λ, α, max_epochs, patience)
    
    def predict(self, path, index=False):
        '''
        Predict the class of the audio file at the given path.
        :param path: The path to the audio file to be classified.
        :param index: Whether to return the index of the class or the class itself.
        :return: The class of the audio file at the given path.
        '''
        vec = self.transformer.transform(path, strict_duration=False).detach().numpy()
        pred, prob = self.model.predict(vec)
        pred, prob = pred[0], prob[0]
        print("Probability of prediction: ", prob)
        return (self.words[pred], prob) if not index else (pred, prob)
