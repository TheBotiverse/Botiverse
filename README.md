<div align="center">
<img src="https://github.com/TheBotiverse/Botiverse/assets/49572294/4cb418bb-de6f-4f13-9df4-878d4b50d66d" height=120/>
</div>
<p align='justify'>
<b>Botiverse</b> is Python package that bridges the gap between developers <i>regardless of their machine learning expertise</i> and 
building chatbots. It offers a <b>diverse set</b> of <b>modern chatbot architectures</b> that are <b>ready to be trained</b> in a <b>high-level fashion</b> while 
offering optional <b>fine-grained control</b> for advanced use-cases.
</p>

We strongly recommend referring to the [documentation](https://botiverse.readthedocs.io/en/latest/) which also comes with a humble [user guide](https://botiverse.readthedocs.io/en/latest/user_guide.html).

## 🚀 Installation
For standard use, consider Python 3.9+ and
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
<b>Botiverse</b> offers 7 main chatbot architectures that cover a wide variety of use cases:

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


- All chatbot architectures that Botiverse support (i.e., in `botiverse.bots`) are composed of a representer that puts the input text or audio in the right representation and a model that is responsible for the chatbot's output.
- All representers (top row) and models (bottom row) with a non-white frame were implemented from scratch for some definition of that.
- Beyond being a chatbot package, most representers and models can be also used independently and share the same API. For instance, you can import your favorite model or representer from `botiverse.models` or `botiverse.preprocessors` respectively and use it for any ordinary machine learning task.
- It follows that some chatbot architectures also allow using a customly defined representer or model as long as it satisfies the relevant unified interface (as in the docs).

Now let's learn more about each chatbot available in `botiverse.bots`

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

![Basic](https://github.com/TheBotiverse/Botiverse/assets/49572294/976edf97-66be-468b-9c1b-3533edd7c3d1)

You can simulate a similar demo offline using the notebook in the [Examples](https://github.com/TheBotiverse/Botiverse/tree/main/Examples) folder or online on [Google collab](https://colab.research.google.com/drive/1MW4PmQ8BOBkfXO-X-IpzHr6-n8CsXKt7#scrollTo=fifdbsJduJF-).

> Google colab won't have a server to run the chat gui, the options are to use a humble version by setting `server=False` or to provide an [ngrok](https://ngrok.com/) authentication token in the `auth_token` argument.

> You will have to manually drop the dataset from the examples folder into the data section in colab.

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

![Whiz](https://github.com/TheBotiverse/Botiverse/assets/49572294/d85c2825-5061-4d5e-b8b3-67823e93f789)


You can simulate a similar demo offline using the notebook in the [Examples](https://github.com/TheBotiverse/Botiverse/tree/main/Examples) folder or online on [Google collab](https://drive.google.com/file/d/14sq63p-HLAmWkZXX42aJSI3uCSMfoAkY/view?usp=sharing).
> Note that the performance of both the basic bot and whiz bot largely scales with the quality and size of the dataset; the one we use here is a small synthetic version generated by LLMs and could be greatly improved if given time.


## <img width="30" alt="image" src="https://github.com/TheBotiverse/Botiverse/assets/49572294/853ef1f1-e8a2-4244-88c6-dcb09209ad91"> Basic Task Bot
### 🏃‍♂️ Quick Example
```python
tbot = BasicTaskBot(domains_slots, templates, domains_pattern, slots_pattern)
bot.infer("I want to book a flight")
```
### 🗂️ User Guide & Docs
Please check this for the [documentation](https://botiverse.readthedocs.io/en/latest/botiverse.bots.BasicTaskBot.html) which also includes the [user guide](https://botiverse.readthedocs.io/en/latest/BasicTaskBot.html).

### 🏮 Demo
The following is the result from building a simple `Basic Task Bot` to perform simple flight booking tasks

![BasicTask](https://github.com/TheBotiverse/Botiverse/assets/49572294/ea319c36-5574-4434-99cc-67f18ad9593f)


You can simulate a similar demo offline using the notebook in the [Examples](https://github.com/TheBotiverse/Botiverse/tree/main/Examples) folder or online on [Google collab](https://drive.google.com/file/d/1V0__NnSFjg4DmN_fp_-gI7WmMz2G7_cb/view?usp=sharing).


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

![TaskBot](https://github.com/TheBotiverse/Botiverse/assets/49572294/1bc188a6-c15d-4fad-b72c-421340b7e00c)


You can simulate a similar demo offline using the notebook in the [Examples](https://github.com/TheBotiverse/Botiverse/tree/main/Examples) folder or online on [Google collab](https://drive.google.com/file/d/1IgpKFZGX5UKABLfB0fzUJ624ZJoLFEX6/view?usp=sharing).

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
The following is the result from the `Converse Bot` before training on Amazon customer service conversations dataset and after it was pretrained on an assistance corpus. You can check for post-training results by checking the examples (training takes time).

![Converse](https://github.com/TheBotiverse/Botiverse/assets/49572294/f343a884-86c4-42ab-9339-1c5a34393bd5)

You can simulate a similar demo offline using the notebook in the [Examples](https://github.com/TheBotiverse/Botiverse/tree/main/Examples) folder or online on [Google collab](https://drive.google.com/file/d/1YCznzhRzv_TDmj1F595THe-6KwQEWf2Y/view?usp=sharing).


## <img width="30" alt="image" src="https://github.com/TheBotiverse/Botiverse/assets/49572294/b66968fb-d035-4fe7-8851-0b72bc6b7789"> Voice Bot
### 🏃‍♂️ Quick Example
```python
bot = VoiceBot('call.json')
bot.simulate_call()
```
### 🗂️ User Guide & Docs
Please check this for the [documentation](https://botiverse.readthedocs.io/en/latest/botiverse.bots.VoiceBot.html) which also includes the [user guide](https://botiverse.readthedocs.io/en/latest/VoiceBot.html). An independent submodule of the `voice bot` is a speech classifier which may learn from zero-shot data (synthetic generation). If interested in that then check this for the [documentation](https://botiverse.readthedocs.io/en/latest/botiverse.bots.SpeechClassifier.html) which also includes the [user guide](https://botiverse.readthedocs.io/en/latest/SpeechClassifier.html).

### 🏮 Demo
The following is the result from building a `Voice Bot` on a hand-crafted call state machine as found in the [Examples](https://github.com/TheBotiverse/Botiverse/tree/main/Examples). The voice bot requires no training data.


https://github.com/TheBotiverse/Botiverse/assets/49572294/cd58965e-3659-4495-baa1-d87da1c01215

You can only simulate a similar demo offline using the notebook in the [Examples](https://github.com/TheBotiverse/Botiverse/tree/main/Examples) folder. This applies to both the voice bot and the speech classifier.

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
No demo is available yet for the Theorizer; you may check the example in the [Examples](https://github.com/TheBotiverse/Botiverse/tree/main/Examples) folder. 

## 🌆 Models and Preprocessors
Most could be indepdendently used in any task; please consult the relevant section of the [documentation](https://botiverse.readthedocs.io/en/latest/) and the `Examples` folder.


## 👥 Collaborators
<table>
<tr>
    <td align="center">
        <a href="https://github.com/EssamWisam">
            <img src="https://avatars.githubusercontent.com/u/49572294?v=4" width="100;" alt="EssamWisam"/>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/YousefAtefB">
            <img src="https://avatars.githubusercontent.com/u/72487484?v=4" width="100;" alt="YousefAtefB"/>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/Muhammad-saad-2000">
            <img src="https://avatars.githubusercontent.com/u/61880555?v=4" width="100;" alt="Muhammad-saad-2000"/>
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/Kariiem">
            <img src="https://avatars.githubusercontent.com/u/48629566?v=4" width="100;" alt="Kariiem"/>
        </a>
    </td>
</tr>
  <tr>
    <td align="center">
        <a href="https://github.com/EssamWisam">
            Essam
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/YousefAtefB">
            Yousef Atef
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/Muhammad-saad-2000">
          Muhammad Saad
        </a>
    </td>
    <td align="center">
        <a href="https://github.com/Kariiem">
            Kariiem Taha
        </a>
    </td>
</tr>
    <tr>
    <td align="center">
        Basic Bot and Voice Bot & Relevant Models
    </td>
    <td align="center">
        Basic and Deep Task Bot & Relevant Models
    </td>
    <td align="center">
        Whiz and Converse Bot & Relevant Models
    </td>
    <td align="center">
       Theorizer & Relevant Models
    </td>
</tr>
</table>

Sincere thanks to [Abdelrahman Jamal](https://github.com/Hero2323) for helping test the package on Windows.
