'''
Common constants
'''
import os

THEORIZER_PATH= os.path.abspath(os.path.dirname(__file__))
BOTIVERSE_PATH=None
SQUAD_TRAIN_PATH=None

QUESTION_TYPES = [
    "Who", "Where", "When", "Why", "Which", "What", "How", "Yes-No", "Other"]
INFO_QUESTION_TYPES = [
    "Who", "Where", "When", "Why", "Which", "What", "How"]
YES_NO_QUESTION_TYPES = [
    "Am", "Is", "Was", "Were", "Are",
    "Does", "Do", "Did",
    "Have", "Had", "Has",
    "Could", "Can",
    "Shall", "Should",
    "Will", "Would",
    "May", "Might"]
Q_TYPE2ID_DICT = {
    "What": 0, "Who": 1, "How": 2,
    "Where": 3, "When": 4, "Why": 5,
    "Which": 6, "Boolean": 7, "Other": 8}

STOP_WORDS = []