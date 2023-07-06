from gtts import gTTS
import os
from tqdm import tqdm
from pydub import AudioSegment
import librosa
import random
import soundfile as sf
import gdown
import shutil

class Vocalize():
    '''
    An interface for transforming words into audio files via Google's Text-to-Speech API and adding noise to them.
    '''
    def __init__(self, words, sample_rate=16000, duration=16000):
        '''
        Initialize the Vocalize transformer by setting the input words and making the dataset.
        :param: words: A list of words to be transformed into audio files
        :param: sample_rate: The sample rate of the audio files
        :param: duration: The duration of the audio files in milliseconds
        '''
        self.sample_rate = sample_rate
        self.duration = duration
        self.words = words
        self.make_dataset()
        
    def make_dataset(self):
        '''
        Make a dataset of audio files for the given words by using Google's Text-to-Speech API to pronounce the word
        in australian, british, american, indian, and south african accents.
        '''
        # make a folder for each word in folder 'dataset
        for word in self.words:
            if not os.path.exists('dataset/' + word):
                os.makedirs('dataset/' + word)

        # Make audio for each word
        for word in tqdm(self.words):
            tlds = ["com.au", "co.uk", "us", "co.in", "co.za"]
            for i, tld in enumerate(tlds):
                tts = gTTS(text=word, lang="en", tld=tld, slow=False)                       # Sample rate of 24K
                tts.save(f"dataset/{word}/{i}.mp3")
                sound = AudioSegment.from_mp3(f"dataset/{word}/{i}.mp3")
                sound.export(f"dataset/{word}/{i}.wav", format="wav")
                os.remove(f"dataset/{word}/{i}.mp3")

    @staticmethod
    def corrupt_dataset(words=None, duration=16000, sample_rate=16000):
        '''
        Given a folder 'dataset' with folders each containing audio files, this randomly adds noise to each audio file and saves it
        by applying specific noise introduction logic. If noise is not found locally, it is downloaded from Google Drive.
        :param: words: A list of words to be transformed into audio files (i.e., the folder names)
        :param: duration: The duration of the audio files in milliseconds
        :param: sample_rate: The sample rate of the audio files
        '''
        # if words is None then assume they are the folder names in dataset
        if words is None: words = os.listdir('dataset')
        
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        if not os.path.exists(f"{curr_dir}/noises"):
            print("Noises not found. Downloading the noise sounds to be used for augmentation...")
            # if not, download the WaveGlow folder
            f_id = '13sOukAKPjoW1K0Ic-8t49P_nkO1dj1oc' 
            gdown.download(f'https://drive.google.com/uc?export=download&confirm=pbef&id={f_id}', curr_dir + '/noises.zip', quiet=False)
            # extract the folder
            shutil.unpack_archive(curr_dir + '/noises.zip', curr_dir)
            print("Done.")       
            
        for word in tqdm(words):
            num_sounds = len(os.listdir(f"dataset/{word}"))
            for i in range(num_sounds):
                noise_added = False
                waveform, sample_rate = librosa.load(f"dataset/{word}/{i}.wav", sr=None)

                #waveform = librosa.effects.pitch_shift(y=waveform, sr=sample_rate, n_steps=-2)

                # with probability 50% add room noise, 35% cafe noise, and 15% traffic noise
                noise_prob = random.random()
                if noise_prob < 0.5:
                    noise_waveform, sr = librosa.load(os.path.join(curr_dir, "./noises/room.wav"), sr=None)
                    noise_type = "room"
                    noise_added = True
                elif noise_prob < 0.75 and not noise_added:
                    noise_waveform, sr = librosa.load(os.path.join(curr_dir, "./noises/cafe.wav"), sr=None)
                    noise_type = "cafe"
                    noise_added = True
                else:
                    noise_waveform, sr = librosa.load(os.path.join(curr_dir, "./noises/traffic.wav"), sr=None)
                    noise_type = "traffic"

                # Resample noise waveform to match the sample rate of the target waveform
                noise_waveform = librosa.resample(y=noise_waveform, orig_sr=sr, target_sr=sample_rate)

                # Trim the waveforms to match the desired duration
                target_duration = int(duration / 1000)  # convert milliseconds to seconds
                max_offset = len(noise_waveform) - len(waveform)
                offset = random.randint(0, max_offset)
                noise_waveform = noise_waveform[offset:offset+len(waveform)]

                # Add noise to the waveform
                waveform = waveform + noise_waveform

                # Save the modified waveform with noise
                output_folder = f"dataset/{word}"
                output_path = f"{output_folder}/{i}-{noise_type}.wav"
                sf.write(output_path, waveform, sample_rate, 'PCM_24')

