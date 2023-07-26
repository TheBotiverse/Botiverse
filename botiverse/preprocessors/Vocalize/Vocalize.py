try:
    from gtts import gTTS
    import os
    from tqdm import tqdm
    import librosa
    import random
    import soundfile as sf
    import gdown
    import shutil
except:
    pass

class Vocalize():
    '''
    An interface for transforming words into audio files via Google's Text-to-Speech API and adding noise to them.
    '''
    def __init__(self, words):
        '''
        Initialize the Vocalize transformer by setting the input words and making the dataset.
        
        :param words: A list of words to be transformed into audio files
        :type words: list
        '''
        self.words = words
        self.make_dataset()
        
    def make_dataset(self):
        '''
        Make a dataset of audio files for the given words by using Google's Text-to-Speech API to pronounce the word
        in australian, british, american, indian, and south african accents.
        '''
        # if there is a folder called dataset, delete it
        if os.path.exists('dataset'): shutil.rmtree('dataset')
        # make a folder for each word in folder 'dataset
        for word in self.words:
            if not os.path.exists('dataset/' + word):
                os.makedirs('dataset/' + word)

        # Make audio for each word
        print("Making audio files for each word...")
        for word in tqdm(self.words):
            tlds = ["com.au", "co.uk", "us", "co.in", "co.za"]
            for i, tld in enumerate(tlds):
                tts = gTTS(text=word, lang="en", tld=tld, slow=False)                       # Sample rate of 24K
                tts.save(f"dataset/{word}/{i}.mp3")
                # convert to wav
                os.system(f"ffmpeg -i dataset/{word}/{i}.mp3 -acodec pcm_s16le -ac 1 -ar 16000 dataset/{word}/{i}.wav -loglevel quiet")                
                # remove the mp3 file
                os.remove(f"dataset/{word}/{i}.mp3")
    @staticmethod
    def corrupt_dataset(words=None, sample_rate=16000, traffic=False, force_download=False):
        '''
        Given a folder 'dataset' with folders each containing audio files, this randomly adds noise to each audio file and saves it
        by applying specific noise introduction logic. If noise is not found locally, it is downloaded from Google Drive.
        
        :param words: A list of words to be transformed into audio files (i.e., the folder names)
        :type words: list
        :param sample_rate: The sample rate of the audio files
        :type sample_rate: int
        :param traffic: Whether to add traffic noise to the audio files
        :type traffic: bool
        :param force_download: Whether to force download the noise dataset even if it already exists.
        :type force_download: bool
        '''
        # if words is None then assume they are the folder names in dataset
        if words is None: words = os.listdir('dataset')
        
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        # does not exist or is empty
        if not os.path.exists(f"{curr_dir}/noises") or force_download:
            print("Noises not found. Downloading the noise sounds to be used for augmentation...")
            # if not, download the WaveGlow folder
            f_id = '13sOukAKPjoW1K0Ic-8t49P_nkO1dj1oc' 
            gdown.download(f'https://drive.google.com/uc?export=download&confirm=pbef&id={f_id}', curr_dir + '/noises.zip', quiet=False)
            # extract the folder
            shutil.unpack_archive(curr_dir + '/noises.zip', curr_dir)
            print("Done.")   
            # remove the zip file
            os.remove(curr_dir + '/noises.zip') 
        
        print("Corrupting the dataset...")
        for word in tqdm(words):
            # for each file in the folder
            for file in os.listdir(f"dataset/{word}"):
                noise_added = False
                waveform, sample_rate = librosa.load(f"dataset/{word}/{file}", sr=sample_rate)

                #waveform = librosa.effects.pitch_shift(y=waveform, sr=sample_rate, n_steps=-2)

                # with probability 100% add room noise, 40% cafe noise, and 20% traffic noise
                noise_prob = random.random()
                noise_waveform, sr = librosa.load(os.path.join(curr_dir, "./noises/room.wav"), sr=None)
                noise_type = "room"
                
                if noise_prob < 0.4:
                    noise_waveform, sr = librosa.load(os.path.join(curr_dir, "./noises/cafe.wav"), sr=None)
                    noise_type = "cafe"
                    noise_added = True
                elif noise_prob < 0.6 and not noise_added and traffic:
                    noise_waveform, sr = librosa.load(os.path.join(curr_dir, "./noises/traffic.wav"), sr=None)
                    noise_type = "traffic"

                # Resample noise waveform to match the sample rate of the target waveform
                noise_waveform = librosa.resample(y=noise_waveform, orig_sr=sr, target_sr=sample_rate)

                # Trim the waveforms to match the desired duration
                max_offset = len(noise_waveform) - len(waveform)
                offset = random.randint(0, max_offset)
                noise_waveform = noise_waveform[offset:offset+len(waveform)]

                # Add noise to the waveform
                waveform = waveform + noise_waveform

                # Save the modified waveform with noise
                output_folder = f"dataset/{word}"
                # same name but append the noise type
                output_path = f"{output_folder}/{file.split('.')[0]}_{noise_type}.wav"
                sf.write(output_path, waveform, sample_rate, 'PCM_24')
                
                # remove the original file 
                os.remove(f"dataset/{word}/{file}") 

