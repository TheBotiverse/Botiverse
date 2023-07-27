Basic Task Bot Guide
====================

We start by importing the basic task bot and the chat gui for later
demonstration

.. code:: ipython3

    from botiverse import chat_gui
    from botiverse.bots import BasicTaskBot

The basic task bot works by simply interpreting user responses to
collect information. To do any task such as booka flight, all the bot
needs is information. Each piece of information is called a slot and the
whole task the bot performs is called a domain.

We start by defining - The tasks the bot should be able to perform and
the info needed for each task (the bot’s objective is to correctly
collet such information) - Some utterances for the bot to use when one
of the pieces of information (slots) is missing - Patterns that when
match user input indicate that a task should be performed (start
collecting information) - Patterns that when match user input indicate
that information pertaining to a slot has been provided

The patterns form a grammar for the chatbot as the correspond to all the
possible ways a user can interact with the bot.

Define the Grammar
~~~~~~~~~~~~~~~~~~

Let’s showcase the basic task bot using an extremely simple example.

1. Decide the Domains (i.e., tasks) and Slots in each
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we consider one task only which is booking a flight and which for
the bot is equivalent to collecting information about the source,
destination and day of the flight.

.. code:: ipython3

    domains_slots = {
        "book-flight": ["source", "destination", "day"]
        }

2. Decide Chatbot Utterances per Slot
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose a slot is empty, what should the chatbot say to request
information from the user. We provide templates for each slot.

.. code:: ipython3

    
    templates = {
                    "book-flight":
                    {
                        "source": ["Where do you want to fly from?",
                                   "From where will you take the flight?"],
                        "destination": ["What is your destination?",
                                        "Where do you want to go?"],
                        "day": ["What day do you want to leave?"]
                    }
                }

3. Write Patterns that Initiate a Task
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now the serious part is to write patterns that match user queries that
correspond to initiating a task. We use a very simple example here.

.. code:: ipython3

    domains_pattern = {"book-flight": r"(i|I) want to (book|reserve) a? flights?"}

4. Write Patterns to Interpret Answers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Finally, we provide patterns to collect the needed information.

.. code:: ipython3

    slots_pattern = {
                        "book-flight":
                        {
                            "source": r"from(?: city)? (cairo|giza)",
                            "destination": r"to(?: city)? (cairo|giza)",
                            "day": r"(saturday|sunday|monday|tuesday|wednesday|thursday|friday)"
                        }
                    }

Build the chatbot
~~~~~~~~~~~~~~~~~

That’s it, we can now make an instance of our chatbot and guess what, no
training is needed at all!

.. code:: ipython3

    chatbot = BasicTaskBot(domains_slots, templates, domains_pattern, slots_pattern, verbose=True)

Deploy the Chatbot
~~~~~~~~~~~~~~~~~~

We can try out our chatbot using the GUI as follows

.. code:: ipython3

    chat_gui('Task Bot', chatbot.infer)


.. parsed-literal::

     * Serving Flask app 'botiverse.gui.gui'
     * Debug mode: off


.. parsed-literal::

    [31m[1mWARNING: This is a development server. Do not use it in a production deployment. Use a production WSGI server instead.[0m
     * Running on http://127.0.0.1:5000
    [33mPress CTRL+C to quit[0m




.. parsed-literal::

    <botiverse.gui.gui.chat_gui at 0x106a484f0>



.. code:: ipython3

    !jupyter nbconvert --to markdown BasicTaskBot.ipynb
    !jupyter nbconvert --to rst BasicTaskBot.ipynb

w