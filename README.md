<div align="center">
<img src="https://github.com/TheBotiverse/Botiverse/assets/49572294/4cb418bb-de6f-4f13-9df4-878d4b50d66d" height=120/>
</div>
<p>
<b>Botiverse</b> is Python package that bridges the gap between developers <i>regardless of their machine learning expertise</i> and 
building chatbots. It offers a <b>diverse set</b> of <b>modern chatbot architectures</b> that are <b>ready to be trained</b> in a <b>high-level fashion</b> while 
offering optional <b>fine-grained control</b> for advanced use-cases.
</p>

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
    <td>Similar to basic bot but suitable for cases where there is more data or better performance is needed in return of more computation</td>
  </tr>
    <tr>
    <td>Task Bot</td>
    <td>A task-oriented chatbot based on encoder transformer models</td>
    <td>A chatbot that can collect all the needed information to perform a task such as booking a flight or a hotel</td>
  </tr>
    </tr>
  <tr>
    <td>Task Bot</td>
    <td>A basic light-weight version of the task bot purely based on Regex and grammars</td>
    <td>When insufficient data exists for the deep version and developers are willing to design a general grammar for the task</td>
  </tr>
  <tr>
    <td>Converse Bot</td>
    <td>A conversational chatbot based on language modeling with transformers</td>
    <td>A chatbot that converses similar to human agents; e.g., a narrow version of ChatGPT as customer service</td>
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

<img width="1426" alt="image" src="https://github.com/TheBotiverse/Botiverse/assets/49572294/c75c9534-d4d4-4d70-a400-2f96b62f38c1">

