# Voice Bot Guide


```python
from botiverse.bots import VoiceBot
```

The voice bot is one of the easiest modules to deal with from `botiverse.bots`, all it takes is to design a call finite state machine similar to `call.json` in the guide folder. The following is a sample from it.

### Dataset Sample

  ```json
{
    "A":{
    "Bot": "Thanks for calling Botiverse Airlines. Would you like us to proceed in English or Spanish?",
    "max_duration": "3",
    "Options": [
        {
            "Intent": "English",
            "Speak": "Okay. We will continue in English",
            "Next": "B"
        },
        {
            "Intent": "Spanish",
            "Speak": "Sorry. But I do not speak spanish yet.",
            "Next": "A"
        }
    ]
},
}
``` 

The voice bot always assumes that the first state in the `call.json` is `A` and that the last slot is `Z`.

### Load the Call Finite State Machine

After the call finite state machine is designed, we simply pass it while initiating the voice bot


```python
bot = VoiceBot('call.json')
```

### Simulate a Call

Then we follow up with `bot.simulate_call()` and the call starts. Whenever the bot says something `max_duration` time will be waited (and a progress bar shown) to record a response.


```python
bot.simulate_call()
```
