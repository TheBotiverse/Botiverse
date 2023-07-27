# Whiz Bot Guide

We start by importing the basic bot from `botiverse.bots` and gui for testing.


```python
from botiverse.bots import WhizBot
from botiverse import chat_gui
```

The Whiz Bot is much like the basic bot except that it is capable of using multinigual embeddings and sequential models which means better performance and multi-linguality at the cose of more training time. In this we train on an Arabic dataset similar to the one we used with the basic bot.

#### Dataset Sample

```json
  {
    "tag": "برامج",
    "patterns": [
      "ما هي البرامج التي تقدمها الجامعة؟",
      "ما هي المقررات المتاحة؟", 
      "أخبرني عن البرامج الأكاديمية",
      "هل يمكنك تقديم معلومات عن التخصصات؟" 
    ],
    "responses": [
      "...تقدم جامعتنا مجموعة واسعة من البرامج في",
      "...نقدم برامج أكاديمية متنوعة تشمل مجالات دراسية مختلفة"
    ]
  }
```

### Initiate Chatbot

We start by initiating the whiz bot. Although it supports two different models (`linear` and `GRU`); each of those has its own representation `BERT` and `BytePairOneHotEncoding` respectively (for the latter, `repr` is passed as `GRU`)

bot = WhizBot(repr='BERT')

### Read the Data

We read the data similar to how we did with the basic bot


```python
bot.read_data('./dataset_ar.json')
```

### Train the chatbot

We train the chatbot where we can also supply the number of epochs and batch size.


```python
bot.train(epochs=10, batch_size=32)
```

    Train Acc: 0.93: 100%|██████████| 240/240 [00:01<00:00, 220.21it/s]


### Infer

Finally, we can infer given real data as usual


```python
bot.infer("ما هي الدورات المتاحة؟")
```




    "Hello! Welcome to our university's website."



### Deploy the Chatbot

And deploy the model if needed.


```python
chat_gui("Whiz Bot", bot.infer)
```


```python
# convert to markdown
import nbformat
from nbconvert import MarkdownExporter
# get the notebook filename 
with open('WhizBot.ipynb', 'r') as file:
    notebook_content = nbformat.read(file, as_version=4)

# Initialize the Markdown exporter
md_exporter = MarkdownExporter()

# Convert the notebook to Markdown
markdown_output, _ = md_exporter.from_notebook_node(notebook_content)

# Save the Markdown content to a file
with open('WhizBot.md', 'w', encoding='utf-8') as file:
    file.write(markdown_output)
```
