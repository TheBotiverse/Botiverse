from flask import Flask, render_template, request, jsonify
import random



class chat_gui():
    chat_func = None
    bot_type = None
    # one of ['Whiz Bot', 'Basic Bot', 'Task Bot', 'Converse Bot', 'Voice Bot', 'Theorizer']
    def __init__(self, bot_type, chat_func):
        chat_gui.bot_type = bot_type
        chat_gui.chat_func = chat_func
        app.run(port=5000, debug=False)

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('chat.html', bot=chat_gui.bot_type)


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    return get_Chat_response(input)


def get_Chat_response(text):
    return chat_gui.chat_func(text)


