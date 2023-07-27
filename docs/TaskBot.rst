Task Bot Guide
==============

We start by importing the task bot from ``botiverse.bots`` and as well
import ``chat_gui`` for testing purposes

.. code:: ipython3

    from botiverse import chat_gui
    from botiverse.bots import TaskBot

The task bot generalizes the basic version that we talked about earlier
to become completely insensitive to how the user phrases their response
and to be able to collect all the needed information to perform a task.
It also has extra ``inform`` and ``refer`` and other features that make
it capable of being a better simulation of a real human agent.

The architecture is based on the
:literal:`TRIPPY`\` model which is based on`\ BERT\`\ ``and the dataset we will use is``\ sim-R\`
which overall tallies with implementing the paper by Heck et
al. `TripPy: A Triple Copy Strategy for Value Independent Neural Dialog
State Tracking <https://arxiv.org/abs/2005.02877>`__ which is the
state-of-the-art in task-oriented chatbots.

Dataset Sample
~~~~~~~~~~~~~~

``json {     "turn_idx": 4,     "system_utterance": "4 people are going tonight , correct ?",     "user_utterance": "yes , 4 people are going tonight .",     "turn_slots": {},     "system_act": {         "restaurant-num_people": [             "4"         ],         "restaurant-date": [             "tonight"         ]     } },``

Setup State Info
~~~~~~~~~~~~~~~~

The info we need to set up is very much like the basic task bot except
that there are no patterns involved.

1. Set the Domain (i.e., task) and its Slots
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    # The domains e.g. restaurant, hotel, etc.
    domains = ["restaurant"]
    
    # The names of the slots where each slot is prefixed with the domain name e.g. restaurant-name, hotel-name, etc.
    slot_list = [
        "restaurant-category",
        "restaurant-rating",
        "restaurant-num_people",
        "restaurant-location",
        "restaurant-restaurant_name",
        "restaurant-time",
        "restaurant-date",
        "restaurant-price_range",
        "restaurant-meal"
    ]

2. Set Initial Utterance and Chatbot Utterances per Slot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code:: ipython3

    # The utterances for the system to start the conversation
    start = [
        {
            'utterance': 'Hi I am Tody, I can help you reserve a restaurant?',
            'slots': [],
            'system_act': {}
        }
    ]
    
    # The templates for generating responses
    templates = [
        {
            'utterance': 'what type of food do you want and in what area?',
            'slots': ['restaurant-location', 'restaurant-category'],
            'system_act': {}
        },
        {
            'utterance': 'what is your preferred price range and rating?',
            'slots': ['restaurant-price_range', 'restaurant-rating'],
            'system_act': {}
        },
        {
            'utterance': 'how many people will be in your party?',
            'slots': ['restaurant-num_people'],
            'system_act': {}
        },
        {
            'utterance': 'what time and date would you like to reserve a table for?',
            'slots': ['restaurant-time', 'restaurant-date'],
            'system_act': {}
        },
        {
            'utterance': 'May I suggest kfc restaurant?',
            'slots': ['restaurant-restaurant_name'],
            'system_act': {'restaurant-restaurant_name': ['kfc']}
        },
        {
            'utterance': 'ok done, here is your reservation number: 123456',
            'slots': [],
            'system_act': {}
        }
    ]

Initiate Chatbot
~~~~~~~~~~~~~~~~

Similar to the basic task bot, we start by making an instance of the
chatbot while providing the domains (tasks), the slots for each, the
utterance the bot should start with and the templates the bot should
utter when a slot is missing (those with higher priority; mentioned
first; are uttered by the bot first).

.. code:: ipython3

    chatbot = TaskBot(domains, slot_list, start, templates, verbose=True)

Load the Dataset
~~~~~~~~~~~~~~~~

We load the train, validation and test data. Here we use the sim-R
dataset for task-oriented dialogue systems. In general, you should at
least provide the training data.

.. code:: ipython3

    train_path = 'sim-R_dataset/train_dials.json'
    dev_path = 'sim-R_dataset/dev_dials.json'
    test_path = 'sim-R_dataset/test_dials.json'
    
    chatbot.read_data(train_path, dev_path, test_path)

Train the Chatbot
~~~~~~~~~~~~~~~~~

We train the chatbot on the read data

.. code:: ipython3

    chatbot.train()

Load Pre-trained Model
~~~~~~~~~~~~~~~~~~~~~~

Alternatively, because sim-R is a popular dataset that can suit a wide
variety of use-cases, we have pretrained a version of the task bot on it
and you can load it directly.

.. code:: ipython3

    chatbot.load_dst_model(pretrained='sim-R')

Deploy the Chatbot
~~~~~~~~~~~~~~~~~~

After training or loading the pretrained model, it can be deployed. Here
we use a simple chat GUI for that purpose.

.. code:: ipython3

    chat_gui('Task Bot', chatbot.infer)
