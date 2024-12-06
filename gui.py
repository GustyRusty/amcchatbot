import random
import json
import pickle
import numpy as np
import nltk
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
import tkinter as tk
from tkinter import scrolledtext, messagebox

lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json").read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

def clean_up_sentences(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bagw(sentence):
    sentence_words = clean_up_sentences(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bagw(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result = ""
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def send_message(event=None):
    user_input = input_box.get()
    if user_input.strip() == "":
        messagebox.showwarning("Warning", "Please enter a message.")
        return

    chat_area.config(state='normal')
    chat_area.insert(tk.END, "You: " + user_input + "\n", 'user')
    input_box.delete(0, tk.END)

    ints = predict_class(user_input)
    res = get_response(ints, intents)
    chat_area.insert(tk.END, "Eric: " + res + "\n", 'bot')
    chat_area.config(state='disabled')
    chat_area.see(tk.END)

root = tk.Tk()
root.title("AMC 8 Preparation Chatbot")
root.geometry("500x600")
root.configure(bg="#f0f8ff")

chat_area = scrolledtext.ScrolledText(root, state='disabled', wrap=tk.WORD, bg="#e6f7ff", fg="#000000", font=("Helvetica", 14))
chat_area.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)

input_box = tk.Entry(root, width=80, bg="#ffffff", fg="#000000", font=("Helvetica", 14))
input_box.pack(pady=10, padx=10)

input_box.bind("<Return>", send_message)

instruction_label = tk.Label(root, text="Press Enter to send your message", bg="#f0f8ff", fg="#00008B", font=("Helvetica", 10, 'italic'))
instruction_label.pack(pady=5)

chat_area.tag_config('user', foreground='blue')  
chat_area.tag_config('bot', foreground='green')   

chat_area.config(state='normal')
chat_area.insert(tk.END, "Eric: Hi, I am your AMC 8 Preparation Chatbot Eric. How may I help you today?\n", 'bot')
chat_area.config(state='disabled')

root.mainloop()