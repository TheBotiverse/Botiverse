# Speech Classification Guide

Botiverse initially implemented a speech classifier that was destined to be used with the voice bot. The speech classifier is simply a model that takes a speech signal and outputs a class. 

We will import the speech classifier from `botiverse.bots`, its there because its a very simple bot of sorts that can understand some speech. We also import `voice_input` so we can later test it.


```python
from botiverse.bots import SpeechClassifier
from botiverse.bots.VoiceBot.utils import voice_input
```

In this guide, we will use the speech classifier to tackle the problem of generalizing from a few samples of synthetically generated speech data to real speech data.

### Synthetically Generate Voice Data

The speech classifier can be trained on existing data, it takes the words or phrases to classify between and if no existing dataset folder is found while calling `generate_read_data`, it will synthetically generate data for each word while performing random useful audio transformations (can be customly passed) and corrupting the dataset with noise.


```python
S = SpeechClassifier(['Yes', 'No'], samplerate=16000, duration=0.9, machine='lstm', repr='wav2vec')
X, y = S.generate_read_data(force_download_noise=False, n=5)
```

While making an instance of the speech classifer we use `lstm` as the core model and `wav2vec` as a representation. The speech classifier also supports `spec` and `mfcc` representations but they make tasks like these (no data available) much harder.

#### Train the Model

We train the model and pass all the necessary core model parameters (they are optional, and here the go to the `lstm`). The `patience` parameter decides when to stop training. If `patience` epochs have passed with no improvement, then take the model we had prior to those 100 epochs. Check the documentation for other parameters.


```python
S.fit(X, y, hidden=128, patience=100, α=0.01)
#S.save('speechclassifier')
```

We can optionally save the model as well and later load it as expected.

#### Predict

In this, we record audio for 3 seconds and with a `voice_threshold` of 900 to later classify the record into either 'yes' or 'no'. The `voice_threshold` is helpful in not only recording when there is voice, that's why the output record will often be less than 3 seconds.


```python
audio_path = voice_input(record_time=3, voice_threshold=900)
S.predict(audio_path)
```
