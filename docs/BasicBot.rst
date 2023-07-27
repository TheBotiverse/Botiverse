Basic Bot Guide
===============

We start by importing the basic bot from ``botiverse.bots`` and gui for
testing.

.. code:: ipython3

    from botiverse.bots import BasicBot
    from botiverse import chat_gui

The following is a sample from a dataset of the type which the basic bot
can deal with, you can find the whole dataset in the User Guide folder.

Dataset Sample
^^^^^^^^^^^^^^

.. code:: json

       {
         "tag": "admissions",
         "patterns": [
           "How can I apply to the university?",
           "What are the admission requirements?",
           "Tell me about the application process"
         ],
         "responses": [
           "To apply to our university....",
           "To begin your application process, please refer to the 'Admissions'..."
         ]
       }

Read the Data
~~~~~~~~~~~~~

We start by making an instance of the basic chatbot. ``machine`` is the
underlying machine learning model to use. The basic bot supports ``nn``
and ``svm`` for now or any custom model that satisfies a similar fit and
predict interface (as found in the documentation). ``repr`` is the
representation to use for the data. The basic bot supports ``tf-idf``,
``glove``, ``tf-idf-glove``, ``bow`` for now or any custom
representation that satisfies a similar\ ``transform`` and
``transform_list`` interface

.. code:: ipython3

    bot = BasicBot(machine='nn', repr='glove')

After making an instance of the bot with a neural network model and with
the GloVe as the word representation, we read let the chatbot read the
data.

.. code:: ipython3

    bot.read_data('dataset.json')

Train and save the chatbot
~~~~~~~~~~~~~~~~~~~~~~~~~~

The chatbot can be trained at this point on the data has read.
Generally, the train method can take all the hyperparameters that the
``fit`` method of the underlying model can take. In our case for ``SVM``
or ``NN`` it just uses good defaults.

.. code:: ipython3

    bot.train()
    bot.save('BasicBot')


.. parsed-literal::

    Train Acc: 1.0: 100%|██████████| 240/240 [00:01<00:00, 183.84it/s] 


Infer
~~~~~

Once the chatbot has been trained, the infer method can be called to get
a response to a given input. In general, ``infer`` also take a
``confidence`` parameter to decide when the chatbot should say “I don’t
know”

.. code:: ipython3

    bot.infer("Hello there! ")




.. parsed-literal::

    "The university campus provides a wide range of facilities to support academic, social, and recreational needs. From well-equipped classrooms and libraries to sports facilities and student clubs, there's something for everyone. Check out our website's 'Campus Life' section to discover more about the available amenities."



Deploy the Chatbot
~~~~~~~~~~~~~~~~~~

To demonstrate working in a real backend, we consider first loading the
bot from its file. The dataset is necessary as it includes the responses
and is too small anyway.

.. code:: ipython3

    newbot = BasicBot(machine='nn', repr='glove')
    newbot.load('BasicBot', 'dataset.json')

We can run the GUI by running the following. If server is not set to
true then a much more humble version of the GUI will run locally using
Python’s ``input()`` and ``print()`` functions.

.. code:: ipython3

    chat_gui('Basic Bot', bot.infer, server=True)

Using a Custom Model and Transform
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In this we demonstrate how a custom model or representation can be used
with the basic bot. We start by importing a neural network and GloVe
that satisfy the expected interfaces.

.. code:: ipython3

    from botiverse.models import NeuralNet
    from botiverse.preprocessors import GloVe

We make instances of both to later pass to the chatbot

.. code:: ipython3

    model = NeuralNet(structure=[50, 12, 8], activation='sigmoid')
    transform = GloVe()

After passing them to the basic bot, we follow up with the normal
pipeline.

.. code:: ipython3

    bot = BasicBot(machine=model, repr=transform)
    bot.read_data('dataset.json')

Notice how the ``train`` method is flexible to take all the parameters
needed by the model’s fit.

.. code:: ipython3

    bot.train(batch_size=1, epochs=30, λ = 0.02, eval_train=True, val_split=0.0)


.. parsed-literal::

    Train Acc: 0.71: 100%|██████████| 30/30 [00:00<00:00, 220.55it/s]


.. code:: ipython3

    newbot.infer("Thanks")




.. parsed-literal::

    'Glad I could help! If you have any more questions, feel free to ask.'

