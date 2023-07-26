from flask import Flask, render_template, request, jsonify
from pyngrok import ngrok


class chat_gui():
    chat_func = None
    bot_type = None
    # one of ['Whiz Bot', 'Basic Bot', 'Task Bot', 'Converse Bot', 'Voice Bot', 'Theorizer']
    def __init__(self, bot_type, chat_func, server=True, auth_token=None):
        if server:
            chat_gui.bot_type = bot_type
            chat_gui.chat_func = chat_func
            if auth_token:
                ngrok.set_auth_token(auth_token)
                if auth_token:  ngrok.set_auth_token(auth_token)
                ngrok_tunnel = ngrok.connect(5000)
                print('Public URL:', ngrok_tunnel.public_url)
            app.run()
        else:
            while True:
                input_text = input("You: ")
                if input_text == "":    break
                output=chat_func(input_text)
                print(f"{bot_type}: {output}")

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


