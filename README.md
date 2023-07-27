<div align="center">
<img src="https://github.com/TheBotiverse/Botiverse/assets/49572294/4cb418bb-de6f-4f13-9df4-878d4b50d66d" height=120/>
</div>
<p>
<b>Botiverse</b> is Python package that bridges the gap between developers <i>regardless of their machine learning expertise</i> and 
building chatbots. It offers a <b>diverse set</b> of <b>modern chatbot architectures</b> that are <b>ready to be trained</b> in a <b>high-level fashion</b> while 
offering optional <b>fine-grained control</b> for advanced use-cases.
</p>

We strongly recommend referring to the [documentation](https://botiverse.readthedocs.io/en/latest/) which also comes with a humble [user guide](https://botiverse.readthedocs.io/en/latest/user_guide.html).

## 🚀 Installation
For standard use, consider
```shell
pip install botiverse
```
<p>
This installs botiverse excluding the dependencies needed for the voice bot and its preprocessors. To include those as well, 
consider installing
</p>

```shell
pip install botiverse[voice]
```
and make sure to also have FFMPEG on your machine, as needed by the unavoidable dependency `PyAudio`.

## 🏮 Basic Demo
Import the chatbot you need from `botiverse.bots`. All bots have a similar interface consisting of a read, train and infer method.

```python
from botiverse.bots import BasicBot

# make a chatbot instance
bot = BasicBot(machine='nn', repr='tf-idf')
# read the data
bot.read_data('dataset.json')
# train the chatbot
bot.train()
# infer
bot.infer("Hello there!")
```

## 💥 Supported Chatbots
Botiverse offers 7 main chatbot architectures that cover a wide variety of use cases:

<img width="1426" alt="image" src="https://github.com/TheBotiverse/Botiverse/assets/49572294/d3626974-2d7f-4e13-a3a0-17dd029f786e">
<br><br>
<table>
  <tr>
    <th>Chatbot</th>
    <th>Description</th>
    <th>Example Use Case</th>
  </tr>
  <tr>
    <td>Basic Bot</td>
    <td>A light-weight intent-based chatbot based on classical or deep classiciation models</td>
    <td>Answer frequently asked questions on a website while remaining insensitive to wording</td>
  </tr>
    <tr>
    <td>Whiz Bot</td>
    <td>A multi-lingual intent-based chatbot based on deep sequential models</td>
    <td>Similar to basic bot but suitable for cases where there is more data or better performance or multilinguality is needed in return of more computation</td>
  </tr>
    <tr>
    <td>Task Bot</td>
    <td>A task-oriented chatbot based on encoder transformer models</td>
    <td>A chatbot that can collect all the needed information to perform a task such as booking a flight or a hotel</td>
  </tr>
    </tr>
  <tr>
    <td>Basic Task Bot</td>
    <td>A basic light-weight version of the task bot purely based on Regex and grammars</td>
    <td>When insufficient data exists for the deep version and developers are willing to design a general grammar for the task</td>
  </tr>
  <tr>
    <td>Converse Bot</td>
    <td>A conversational chatbot based on language modeling with transformers</td>
    <td>A chatbot that converses similar to human agents; e.g., like a narrow version of ChatGPT as customer service</td>
  </tr>
    <tr>
    <td>Voice Bot</td>
    <td>A voice bot that simulates a call state machine based on deep speech and embedding models</td>
    <td>A voice bot that collects important information from callers before transferring them to a real agent</td>
  </tr>
  <tr>
    <td>Theorizer</td>
    <td>Based on deep classification and language models</td>
    <td>Converts textual data into question-answer pairs suitable for later training</td>
  </tr>
</table>

## 💥 Supported Preprocessors and Models

<img width="1426" alt="image" src="https://github.com/TheBotiverse/Botiverse/assets/49572294/15520524-07cc-4bb1-a230-198cba398da7">


- All chatbot architectures that Botiverse support (i.e., in `botiverse.bots`) are composed of a representer that puts the input text or audio in the right representation and a model that is responsible for the model's output.
- All representers (top row) and models (bottom row) with a non-white frame were implemented from scratch for some definition of that.
- Beyond being a chatbot package, most representers and models can be also used independently and share the same API. For instance, you can import your favorite model or representer from `botiverse.models` or `botiverse.preprocessors` respectively and use it for any task.
- Some chatbot architectures also allows using a customly defined representer or model as long as it satisfies the relevant interface.

Now let's learn more about each chatbot available inn `botiverse.bots.`

## <img width="30" alt="image" src="https://github.com/TheBotiverse/Botiverse/assets/49572294/501fc8ff-9849-4e6d-9ed5-eaf10c04eefb"> Basic Bot
### 🏃‍♂️ Quick Example
```python
bot = BasicBot(machine='nn', repr='tf-idf')
bot.read_data('dataset.json')
bot.train()
bot.infer("Hello there!")
```
### 🗂️ User Guide & Docs
Please check this for the [documentation](https://botiverse.readthedocs.io/en/latest/botiverse.bots.BasicBot.html) which also includes the [user guide](https://botiverse.readthedocs.io/en/latest/BasicBot.html).

### 🏮 Demo
The following is the result (in its best form) from training the `Basic Bot` on a small synthetic `dataset.json` as found in the examples to answer FAQs for a university website



You can simulate a similar demo offline using the notebook in the `Examples` folder or online on [Google collab](google.com).
> Google colab won't have a server to run the chat gui, the options are to use a humble version by setting `server=False` or to provide an [ngrok](https://ngrok.com/) authentication token in the `auth_token` argument.

You can simulate a similar demo offline using the notebook in the `Examples` folder or online on [Google collab](google.com).

## <img width="30" alt="image" src="https://github.com/TheBotiverse/Botiverse/assets/49572294/63bd3fde-dc1b-470f-a43e-6de61a33ca56"> Whiz Bot
### 🏃‍♂️ Quick Example
```python
bot = WhizBot(repr='BERT')
bot.read_data('./dataset_ar.json')
bot.train(epochs=10, batch_size=32)
bot.infer("ما هي الدورات المتاحة؟")
```
### 🗂️ User Guide & Docs
Please check this for the [documentation](https://botiverse.readthedocs.io/en/latest/botiverse.bots.WhizBot.html) which also includes the [user guide](https://botiverse.readthedocs.io/en/latest/WhizBot.html).

### 🏮 Demo
The following is the result (in its best form) from training the `Whiz Bot` on a small synthetic `dataset.json` as found in the examples to answer FAQs for a university website in Arabic



You can simulate a similar demo offline using the notebook in the `Examples` folder or online on [Google collab](google.com).
> Note that the performance of both the basic bot and whiz bot largely scales with the quality and size of the dataset; the one we use here is a small synthetic version generated by LLMs and could be greatly improved if given time.


## <img width="30" alt="image" src="https://github.com/TheBotiverse/Botiverse/assets/49572294/853ef1f1-e8a2-4244-88c6-dcb09209ad91"> Basic Task Bot
### 🏃‍♂️ Quick Example
```python
chatbot = BasicTaskBot(domains_slots, templates, domains_pattern, slots_pattern)
bot.read_data('dataset.json')
bot.train()
bot.infer("I want to book a flight")
```
### 🗂️ User Guide & Docs
Please check this for the [documentation](https://botiverse.readthedocs.io/en/latest/botiverse.bots.BasicTaskBot.html) which also includes the [user guide](https://botiverse.readthedocs.io/en/latest/BasicTaskBot.html).

### 🏮 Demo
The following is the result from building a simple `Basic Task Bot` to perform simple flight booking tasks

You can simulate a similar demo offline using the notebook in the `Examples` folder or online on [Google collab](google.com).


## <img width="30" alt="image" src="https://github.com/TheBotiverse/Botiverse/assets/49572294/853ef1f1-e8a2-4244-88c6-dcb09209ad91"> Task Bot
### 🏃‍♂️ Quick Example
```python
bot = TaskBot(domains, slot_list, start, templates)
bot.read_data(train_path, dev_path, test_path)
bot.train()
bot.infer("I want to eat in a restaurant")
```
### 🗂️ User Guide & Docs
Please check this for the [documentation](https://botiverse.readthedocs.io/en/latest/botiverse.bots.TaskBot.html) which also includes the [user guide](https://botiverse.readthedocs.io/en/latest/TaskBot.html).

### 🏮 Demo
The following is the result from training the `Task Bot` on the [sim-R](https://github.com/google-research-datasets/simulated-dialogue) dataset which includes many possible tasks.

You can simulate a similar demo offline using the notebook in the `Examples` folder or online on [Google collab](google.com).


## <img width="30" alt="image" src="https://github.com/TheBotiverse/Botiverse/assets/49572294/8de3844a-6864-406d-9e71-bb575be7d698"> Converse Bot
### 🏃‍♂️ Quick Example
```python
bot = ConverseBot()
bot.read_data("conversations.json")
bot.train(epochs=1, batch_size=1)
bot.save_model("conversebot.pt")
bot.infer("What is Wikipedia?")
```
### 🗂️ User Guide & Docs
Please check this for the [documentation](https://botiverse.readthedocs.io/en/latest/botiverse.bots.ConverseBot.html) which also includes the [user guide](https://botiverse.readthedocs.io/en/latest/ConverseBot.html).

### 🏮 Demo
The following is the result from training the `Converse Bot` on Amazon customer service conversations dataset after further pretraining it on a large corpus of conversation

You can simulate a similar demo offline using the notebook in the `Examples` folder or online on [Google collab](google.com).


## <img width="30" alt="image" src="https://github.com/TheBotiverse/Botiverse/assets/49572294/b66968fb-d035-4fe7-8851-0b72bc6b7789"> Voice Bot
### 🏃‍♂️ Quick Example
```python
bot = VoiceBot('call.json')
bot.simulate_call()
```
### 🗂️ User Guide & Docs
Please check this for the [documentation](https://botiverse.readthedocs.io/en/latest/botiverse.bots.VoiceBot.html) which also includes the [user guide](https://botiverse.readthedocs.io/en/latest/VoiceBot.html). An independent submodule of the `voice bot` is a speech classifier which may learn from zero-shot data (synthetic generation). If interested in that then check this for the [documentation](https://botiverse.readthedocs.io/en/latest/botiverse.bots.SpeechClassifier.html) which also includes the [user guide](https://botiverse.readthedocs.io/en/latest/SpeechClassifier.html).

### 🏮 Demo
The following is the result from building a `Voice Bot` on a hand-crafted call state machine as found in the examples. The voice bot requires no training data.

You can only simulate a similar demo offline using the notebook in the `Examples` folder. This applies to both the voice bot and the speech classifier.

## <img width="25" alt="image" src="https://github.com/TheBotiverse/Botiverse/assets/49572294/ba7a4ab4-74ed-46f4-a071-4e29ba1a3a72"> Theorizer
### 🏃‍♂️ Quick Example
```python
context = "Some very long text" 
QAs = generate(context)
print(json.dumps(QAs,indent=4))
```
### 🗂️ User Guide & Docs
Please check this for the [documentation](https://botiverse.readthedocs.io/en/latest/botiverse.bots.Theorizer.html) which also includes the [user guide](https://botiverse.readthedocs.io/en/latest/Theorizer.html).

### 🏮 Demo
No demo is available yet for the Theorizer; you may check the example in the `Examples` folder. 
