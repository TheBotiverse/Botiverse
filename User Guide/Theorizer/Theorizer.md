# Theorizer Guide


```python
from botiverse.Theorizer.generate import generate
import json
```

Let's try the model (perhaps, untrained) on some data


```python
context = "Bob is eating a delicious cake in Vancouver." 

qa_dict = generate(context)
print(json.dumps(qa_dict,indent=4))
```
