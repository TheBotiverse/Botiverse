import os
# from botiverse.Theorizer import generate
import gdown
import zipfile

curr_dir = os.path.dirname(os.path.abspath(__file__))

model_dir = os.path.join(curr_dir,"model")
squad_dir = os.path.join(curr_dir,"squad")

if not os.path.exists(os.path.join(squad_dir,"sample_probs.pkl")):
    print("Sample probs not found. Downloading Theorizer sample probs...")
    url = "https://drive.google.com/uc?id=1UjZaqM9jf9nzeK1R7WdSSLNEVKdxJOHO"
    gdown.download(url,os.path.join(squad_dir,"sample_probs.pkl"), quiet=False)
    print("Done.")

if not os.path.exists(os.path.join(model_dir,"pretrained-model")):
    print("Weights not found. Downloading Theorizer weights...")
    model_path = os.path.join(model_dir,"pretrained-model")
    url = "https://drive.google.com/drive/folders/1rUvMP1HdE_H4TAMG8HxHT6z5Y0ZOnBsg"
    gdown.download_folder(url,output=model_path,quiet=False)
    print("Done.")

from spacy.cli import download
download('en_core_web_sm')

import nltk
nltk.download('punkt')

