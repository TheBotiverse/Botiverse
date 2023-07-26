try:
    import numpy as np
    import json
    from gtts import gTTS
    import tempfile
    import os
    from botiverse.models import TTS
    from playsound import playsound
    from botiverse.models import LSTMClassifier
    from botiverse.preprocessors import Vocalize, Wav2Vec, Wav2Text, BertEmbedder, Frequency, BertSentenceEmbedder
    from botiverse.bots.VoiceBot.utils import voice_input
except:
    pass

class VoiceBot():
    '''An interface for the vocalizer chatbot which simulates a call with a customer service bot.'''
    def __init__(self,  call_json_path, repr='BERT-Sentence'):
        ''' 
        Load the call data from a json file that contains the call's state machine.
        
        :param call_json_path: The path to the json file containing the call state machine.
        :type call_json_path: str
        :param repr: The numerical representation to use for the audio files. Can be 'BERT' or 'BERT-Sentence'.
        :type repr: str
        '''
        with open(call_json_path, 'r') as file:
            call_json = file.read()
        self.call_data = json.loads(call_json)
        self.current_node = 'A'
        self.wav2text = Wav2Text()
        if repr == 'BERT':
            self.bert_embeddings = BertEmbedder()
        elif repr == 'BERT-Sentence':
            self.bert_sentence_embeddings = BertSentenceEmbedder()
        else:
            raise Exception(f"Invalid representation {repr}. Expected BERT or BERT-Sentence.")        
    
    def generate_speech(self, text, offline=False):
        '''Use Google's TTS or offline FastSpeech 1.0 to play speech from the given text.
        
        :param text: The text to be converted into speech.
        :type text: str
        :param offline: Whether to use offline FastSpeech 1.0 to generate speech.
        :type offline: bool
        
        :meta private:
        '''
        if offline:
            tts = TTS()
            tts.speak(text)
        else:
            tts = gTTS(text=text, lang='en', tld="us", slow=False)
            with tempfile.NamedTemporaryFile(delete=False) as temp_audio:
                temp_filename = temp_audio.name + ".mp3"
                tts.save(temp_filename)
                # convert to wav
                os.system(f"ffmpeg -i {temp_filename} -acodec pcm_s16le -ac 1 -ar 16000 {temp_filename[:-4]}.wav -loglevel quiet")
                playsound(temp_filename)
                

    def simulate_call(self):
        '''
        Simulate a call with a voice bot as driven by the call state machine.
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
            selected_ind, score = self.bert_sentence_embeddings.closest_sentence(human_resp, intents, retun_ind=True)
            print(f"you said: {human_resp} and the bot decided that you meant {intents[selected_ind]} with a score of {score}")
            
            # 3 - speak according to the chosen option
            speak_message = options[selected_ind]['Speak']
            self.generate_speech(speak_message)

            # 4 - go to the next state
            self.current_node = options[selected_ind]['Next']

