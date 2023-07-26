from dataclasses import dataclass
from typing import List, Tuple, TypeVar, Callable, Optional, Union, Type
import copy
import spacy
import benepar
import numpy as np
import math
import os

current_file_dir = os.path.dirname(os.path.abspath(__file__))
NLP = spacy.load('en_core_web_sm')
benepar.download('benepar_en3')
PARSER = benepar.Parser("benepar_en3")

FUNCTION_WORDS = set(
    [
        word.rstrip().lower()
        for word in open(
            os.path.join(current_file_dir,"function_words.txt"), "r", encoding="utf-8"
        ).readlines()
    ]
)
NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE = set([
    "the",
    "of",
    "for",
    "to",
    "is",
    "are",
    "and",
    "was",
    "were",
    ",",
    "?",
    ";",
    "!",
    ".",
]
)

QUESTION_TYPES = [
    "who",
    "where",
    "when",
    "why",
    "which",
    "what",
    "how",
    "boolean",
    "other",
]
QUESTION_WORDS = ["who", "where", "when", "why", "which", "what", "how"]
BOOL_QUESTION_WORDS = [
    "am",
    "is",
    "was",
    "were",
    "are",
    "does",
    "do",
    "did",
    "have",
    "had",
    "has",
    "could",
    "can",
    "shall",
    "should",
    "will",
    "would",
    "may",
    "might",
]
Q_TYPE2ID_DICT = {
    "what": 0,
    "who": 1,
    "how": 2,
    "where": 3,
    "when": 4,
    "why": 5,
    "which": 6,
    "boolean": 7,
    "other": 8,
}



class SpacyDoc:
    """
    A stub class for the spacy.tokens.doc.Doc object, used for type hinting.
    """
    pass
class SpacyToken:
    """
    A stub class for the spacy.tokens object, used for type hinting.
    """
    pass


@dataclass
class InfoConfig:
    """
    Configuration for information extraction.
    """

    # style options
    num_sample_style: int = 2
    # answer options
    num_sample_answer: int = 5
    ans_len_bin_width: int = 3
    ans_len_min_val: int = 0
    ans_len_max_val: int = 30
    ans_limit: int = 30
    # clue options
    num_sample_clue: int = 2
    is_clue_topN: int = 20
    clue_dep_dist_bin_width: int = 2
    clue_dep_dist_min_val: int = 0
    clue_dep_dist_max_val: int = 20
    # question options
    ques_limit: int = 50
    # sampling options
    sent_limit: int = 100
    max_sample_times: int = 20


def find_token_spans_in_text(text: str, tokens: str) -> List[Tuple[int, int]]:
    """
    Get the character-level spans of the specified tokens in the input text.

    :param text: str, input text
    :param tokens: list of str, token texts to find in the input text
    :return: list of tuples, representing the character-level spans of each token in the input text
             Each tuple contains the start and end indices of the token in the text (end index exclusive)

    :raises Exception: If any of the specified tokens cannot be found in the input text
    """
    
    current = 0
    token_idx = 0
    spans = []
    text_len = len(text)
    token_len = len(tokens[token_idx])
    while current < text_len:
        if text[current:current + token_len] == tokens[token_idx]:
            spans.append((current, current + token_len))
            current += token_len
            token_idx += 1
            if token_idx >= len(tokens):
                break
            token_len = len(tokens[token_idx])
        else:
            current += 1

    if token_idx != len(tokens):
        print("Token {} cannot be found".format(tokens[token_idx]))
        raise Exception()

    return spans


def match_spans(pattern: str, input_text: str) -> List[Tuple[int, int]]:
    """
    Find all occurrences of the given pattern in the input text and return the character-level spans.

    :param pattern: str, the pattern to match in the input text
    :param input_text: str, the input text where the pattern will be searched
    :return: list of tuples, each tuple represents the character-level span of the pattern in the input text
             Each tuple contains the start and end indices of the pattern in the text (end index exclusive)
    """
    pattern_len = len(pattern)
    text_len = len(input_text)
    spans = []

    for idx in range(text_len - pattern_len + 1):
        if input_text[idx:idx + pattern_len] == pattern:
            spans.append((idx, idx + pattern_len))

    return spans


def normalize(text: str) -> str:
    """
    Normalize the given text by replacing `` with " and '' with ".
    """
    return text.replace("''", '" ').replace("``", '" ')

def token_to_char_indices(sentence):
    """
    Generate character index ranges for each token in a given sentence using spaCy.

    Args:
        sentence (str): The input sentence to be tokenized.

    Returns:
        token_to_char_range (dict): A dictionary where keys are token indices and values are tuples representing the start
                                    and end indices (inclusive) of the token in the sentence.
        char_to_token (dict): A dictionary where keys are character indices and values are the corresponding token indices
                              in the sentence.
    """
    doc = NLP(sentence)
    token_to_char_range = {}
    char_to_token = {}
    for token in doc:
        start_idx, end_idx = token.idx, token.idx + len(token.text) - 1
        token_to_char_range[token.i] = (start_idx, end_idx)
        
        for char_idx in range(start_idx, end_idx + 1):
            char_to_token[char_idx] = token.i

    return token_to_char_range, char_to_token


def weighted_sample(choices, probs):
    """
    Sample from `choices` with probability according to `probs`.

    Args:
        choices (list): A list of elements to sample from.
        probs (list): A list of probabilities corresponding to each element in `choices`. 
                      The probabilities don't need to be normalized.

    Returns:
        any: A randomly sampled element from `choices` based on the provided probabilities.
    """
    probs = np.array(probs) / sum(probs)
    cumulative_probs = np.cumsum(probs)
    r = np.random.random()

    # Find the index of the first cumulative probability greater than or equal to r
    index = np.searchsorted(cumulative_probs, r)
    return choices[index]

def value_to_bin(input_val: int, min_val: int, max_val: int, bin_width: int):
    """
    Determine the bin index for the given input value, based on a binned range between min_val and max_val with a specified bin width.

    Args:
        input_val (float): The input value to be binned.
        min_val (float): The minimum value of the range.
        max_val (float): The maximum value of the range.
        bin_width (float): The width of each bin in the range.

    Returns:
        int: The bin index of the input value.
            - If the input value is within the range, the function returns the corresponding bin index.
            - If the input value is greater than the maximum value, the function returns the index of the last bin + 1.
            - If the input value is less than the minimum value, the function returns -1.
    """
    """
    Disclaimer this function is adapted from the original implementation
    """
    if min_val <= input_val <= max_val:
        return math.ceil((input_val - min_val) / bin_width)
    elif input_val > max_val:
        return math.ceil((max_val - min_val) / bin_width) + 1
    else:
        return -1

def str_find(text, tklist):
    """
    Searches for a sequence of tokens in a given text string, allowing for spaces between characters.

    The function takes a text string and a list of tokens as input, and returns the start and end character
    indices of the sequence of tokens in the text. If the sequence is not found within the text, the function
    returns (-1, -1). Spaces between characters in the input text are allowed and do not affect the search.

    Args:
        text (str): The input text string in which to search for the sequence of tokens.
        tklist (list): A list of tokens representing the sequence to search for in the text. Each token is a string.

    Returns:
        tuple: A tuple containing two integer values:
               - The start character index of the sequence in the text, or -1 if the sequence is not found.
               - The end character index of the sequence in the text, or -1 if the sequence is not found.

    Example:
        >>> text = "This is an example text."
        >>> tklist = ["ex", "am", "ple"]
        >>> str_find(text, tklist)
        (11, 18)
    """
    search_str = "".join(tklist)
    tk1 = tklist[0]
    pos = text.find(tk1)

    while pos >= 0 and pos < len(text):
        i, j = pos, 0
        while i < len(text) and j < len(search_str):
            if text[i] == " ":
                i += 1
                continue
            if text[i] == search_str[j]:
                i += 1
                j += 1
            else:
                break

        if j == len(search_str):
            return pos, i - 1

        newpos = text[pos + 1:].find(tk1)
        pos = pos + 1 + newpos if newpos >= 0 else -1

    return -1, -1

def test_find_token_spans_in_text():
    text = "Welcome hello world"
    tokens = ["hello", "world"]
    spans = find_token_spans_in_text(text, tokens)
    assert spans == [(8, 13), (14, 19)]


def test_match_spans():
    pattern = "hello"
    input_text = "Welcome hello world"
    spans = match_spans(pattern, input_text)
    assert spans == [(8, 13)]


if __name__ == "__main__":
    print(match_spans("hello", "Welcome hello world"))
