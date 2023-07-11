import tkinter as tk
from tkinter import scrolledtext, messagebox

class chat_gui:
    def __init__(self, chatbot_function, speak_function=None):
        self.chatbot_function = chatbot_function
        self.speak_function = speak_function

        # Create the main window
        self.root = tk.Tk()
        self.root.title("Botiverse Chat")
        self.root.geometry("600x500")
        self.root.configure(bg='#222222')
        # disable vertical resizing
        self.root.resizable(True, False)
        
        # Create the chat display area
        self.chat_display = scrolledtext.ScrolledText(self.root, width=40, height=25, state='disabled', bg='#1d1d1d', fg='#1d1d1d')
        self.chat_display.configure(state='disabled', borderwidth=0)
        self.chat_display.configure(highlightthickness=0, relief='flat')
                
        # reduce chat display font
        # self.chat_display.configure(font=("Avenir", 12))

        # Create the input area
        self.input_entry = tk.Entry(self.root, width=30, bg='#505050', fg='#ffffff')
        self.input_entry.bind('<Return>', self.process_input)
        # remove border form input
        self.input_entry.configure(highlightthickness=0, relief='flat')
        # increase height of input area
        # self.input_entry.configure(font=("Avenir", 12))
        
        # Create the send button
        self.send_button = tk.Button(self.root, text="Send", command=self.process_input, width=10)
        # make button grey
        self.send_button.configure(bg='#505050', fg='#ffffff', activebackground='#1d1d1d', activeforeground='#ffffff', background='#1d1d1d', foreground='#ffffff')
        # remove border from button
        #self.send_button.configure(highlightthickness=0, relief='flat')
        # set font
        # self.send_button.configure(font=("Avenir", 15))
        # remove send button 
        self.send_button.pack_forget()

        # Position the widgets in the window
        self.chat_display.pack(fill="both", padx=10, pady=10)
        self.input_entry.pack(fill="x", padx=10, pady=5)
        self.send_button.pack(fill="x", pady=5)

    def process_input(self, event=None):
        # Get the user input
        user_input = self.input_entry.get()
        self.input_entry.delete(0, 'end')

        # Display user input in the chat area
        self.display_message(user_input, user=True)

        # Pass the user input to the chatbot function and get the response
        bot_response = self.chatbot_function(user_input)
        
        # Display bot response in the chat area
        self.display_message(bot_response, user=False)


    def display_message(self, message, user=True):
        # Enable the chat display area to insert text
        self.chat_display.configure(state='normal')

        # Insert the message into the chat display area with a chat bubble
        if user:
            self.chat_display.insert('end', message + '\n', 'user')
        else:
            self.chat_display.insert('end', message + '\n', 'bot')
            if self.speak_function is not None:
                self.speak_function(message)

        # Disable the chat display area to prevent editing
        self.chat_display.configure(state='disabled')

        # Scroll to the bottom of the chat display area
        self.chat_display.see('end')


                
    def run(self):
        # Configure the chat bubble styles
        self.chat_display.tag_configure('user', foreground='white', justify='left', font=('Avenir', 15, 'bold'),
                                        background='#505050', borderwidth=2, relief='solid', 
                                        wrap='word', spacing3=10, spacing1=10, lmargin1=10, spacing2=5)
        
        self.chat_display.tag_configure('bot', foreground='white', justify='right', font=('Avenir', 15, 'bold'),
                                        background='#303030', borderwidth=2, relief='solid', 
                                        wrap='word', spacing3=10, spacing1=10, rmargin=10, spacing2=5)

        

        # Start the main GUI loop
        self.root.mainloop()


# Example usage:
'''
def chatbot_function(input):
    # Your chatbot logic goes here
    # Process the input and generate a response
    response = "Hello! You said: " + input
    return response

chat_gui = ChatGUI(chatbot_function)
chat_gui.run()
'''
