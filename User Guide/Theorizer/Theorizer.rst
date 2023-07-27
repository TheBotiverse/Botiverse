Theorizer Guide
===============

.. code:: ipython3

    from botiverse.Theorizer.generate import generate
    import json

Let’s try the model (perhaps, untrained) on some data

.. code:: ipython3

    context = "Bob is eating a delicious cake in Vancouver." 
    
    qa_dict = generate(context)
    print(json.dumps(qa_dict,indent=4))

.. code:: ipython3

    !jupyter nbconvert --to markdown Theorizer.ipynb
    !jupyter nbconvert --to rst Theorizer.ipynb
