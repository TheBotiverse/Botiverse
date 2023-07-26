try:
    import pyaudio
except:
    pass
import wave
from array import array
from tqdm import tqdm

def voice_input(record_time=3, voice_threshold=900, save_path='sample.wav'):
    '''
    Upon call, record audio for record_time seconds and save it to save_path while only inputting audio that is above the voice_threshold.
    
    :param record_time: The number of seconds to record for.
    :type record_time: int
    :param voice_threshold: The minimum volume of audio to record.
    :type voice_threshold: int
    :param save_path: The path to save the audio file to.
    :type save_path: str
    
    :return: The path to the audio file.
    :rtype: str
    '''
    # """"
    #instantiate the pyaudio
    audio = pyaudio.PyAudio()

    #recording prerequisites
    chunk, channels, sample_rate = 1024, 1, 16000
    stream = audio.open(format=pyaudio.paInt16, channels=channels, rate=sample_rate, input=True, frames_per_buffer=1024)

    #starting recording
    frames=[]
    pbar = tqdm(range(0,int(sample_rate/chunk*record_time)), desc=f"Recording")
    for i in pbar:
        data = stream.read(chunk)
        data_chunk = array('h', data)
        chunk_volume = max(data_chunk)
        if voice_threshold is not None:
            state = 'Talking' if chunk_volume > voice_threshold else 'Silence'
            if state == 'Talking': frames.append(data)
        else:
            frames.append(data)
            state = 'Recording'
        pbar.set_postfix({"Volume": chunk_volume, "State": state})

    stream.stop_stream(); stream.close(); audio.terminate()
    
    # save audio recorded from stream
    wavfile = wave.open(save_path,'wb')
    wavfile.setnchannels(channels)
    wavfile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    wavfile.setframerate(sample_rate)
    wavfile.writeframes(b''.join(frames))
    wavfile.close()

    return save_path
    # """"
    #pass