Converse Bot Guide
==================

We start by importing the converse bot from ``botiverse.bots`` and as
well import ``chat_gui`` for testing purposes

.. code:: ipython3

    from botiverse.bots import ConverseBot
    from botiverse import chat_gui

The converse bot is based on the state-of-the-art model `Text-to-text
Transfer Transformer (T5) <https://arxiv.org/abs/1910.10683>`__ by
Raffel et al. It was finetuned on a large corpus of conversations to be
suitable as a chatbot here at Botiverse and is capable of being further
finetuned using company data to simulate a chatbot the interacts similar
to human agents.

Hence, the starting point is a dataset of conversation by human agents
similar to the following:

Dataset Sample
^^^^^^^^^^^^^^

.. code:: json

   [
     [
       "how long does it take for an order cancellation to process?",
       "Great question! Generally, if an order hasn't begun processing, the ..."
     ],
     [
       "how can I report an undelivered package with no tracking updates or news from the shipping company?",
       "Greetings! Who is the carrier and what is the current status of your package listed here: [URL] ? ^TL"
     ],
     [
       "i have been waiting for a parcel which has been dispatched and said to arrive on Friday ...?",
       "Uh no! That is not what we expect! We would like to take a closer look into this, in ..."
     ],
   ]

The json format is very easy to deal with and is just a list of
questions and their answers by human agents as found in real
conversations. The typical firm with sufficient experience can easily
have a lot of this data.

Read the Data
~~~~~~~~~~~~~

We start by reading the data

.. code:: ipython3

    bot = ConverseBot()
    bot.read_data("conversations.json")

Train and save the chatbot
~~~~~~~~~~~~~~~~~~~~~~~~~~

And then training the model and saving it. The ``train`` method supports
arguments for the number of epochs and the batch size and provides a way
to measure the chatbots accuracy on the training data as it trains.

.. code:: ipython3

    bot.train(epochs=1, batch_size=1)
    bot.save_model("conversebot.pt")


.. parsed-literal::

    Train Acc: 0.93: 100%|██████████| 240/240 [00:01<00:00, 220.21it/s]


Infer
~~~~~

Once trained we can use the ``infer`` method as usual. It takes a string
for the user input and returns a string for the chatbot’s response.

.. code:: ipython3

    bot.infer("What is Wikipedia?")




.. parsed-literal::

    "Hello! Welcome to our university's website."



Deploy the Chatbot
~~~~~~~~~~~~~~~~~~

Finally, we can deploy the chatbot as needed.

.. code:: ipython3

    chat_gui('Converse Bot', bot.infer, server=True)
