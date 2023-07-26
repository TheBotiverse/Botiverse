from typing import List, Tuple, Dict, Any
from collections import Counter
from dataclasses import dataclass
import os

import numpy as np
from .utils import *
from .info_extractor import *
import pickle as pkl

current_file_dir = os.path.dirname(os.path.abspath(__file__))

@dataclass
class AnswerSample:
    answer_text: str
    char_st: int
    char_ed: int
    st: int
    ed: int
    answer_bio_ids: List[str]
    answer_pos_tag: str
    answer_ner_tag: str


@dataclass
class ClueSample:
    clue_text: str
    clue_binary_ids: np.ndarray


def select_answers(chunklist, sentence, sample_probs, config=InfoConfig()) -> List[AnswerSample]:
    """
    Select multiple answer chunks from a given list of chunks based on their probability.

    Args:
        chunklist (list): A list of chunks, where each chunk is a tuple containing NER tag, POS tag,
                          token leaves, start index, and end index.
        sentence (str): The input sentence from which the chunks are extracted.
        sample_probs (dict): A dictionary containing the probabilities of different answer conditions.
        config (InfoConfig, optional): A configuration object containing parameters for the sampling process.

    Returns:
        list: A list of sampled answers, where each answer is a tuple containing answer text, character start index,
              character end index, token start index, token end index, answer BIO tags, POS tag, and NER tag.
    """
    token2idx, idx2token = token_to_char_indices(sentence)
    # I can write haskell, can you?
    # chunk === chunk_ner_tag, chunk_pos_tag, leaves_without_position, st, ed
    chunk: Tuple[str, str, List[str], int, int]
    a_probs = [
        sample_probs["a"]
        ["_".join(["-".join([chunk[0], chunk[1]]),
                   str(value_to_bin(
                       abs(chunk[3] - chunk[4] + 1),
                       config.ans_len_min_val,
                       config.ans_len_max_val,
                       config.ans_len_bin_width
                   ))])]
        if chunk[2][0].lower() not in NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE
        else 1
        for chunk in chunklist
    ]

    sampled_chunk_ids = set(
        weighted_sample(list(range(len(chunklist))), a_probs)
        for _ in range(config.max_sample_times)
    )
    sentence_spacydoc = NLP(sentence)

    def process_chunk(chunk):
        chunk_ner_tag, chunk_pos_tag, leaves, st, ed = chunk
        context = sentence
        char_st, char_ed = str_find(context, leaves)
        if char_st < 0:
            return None
        answer_text = context[char_st: char_ed + 1]
        st, ed = idx2token[char_st], idx2token[char_ed]
        answer_bio_ids = ["O"] * len(sentence_spacydoc)
        answer_bio_ids[st: ed + 1] = ["I"] * (ed - st + 1)
        answer_bio_ids[st] = "B"
        char_st, char_ed = token2idx[st][0], token2idx[ed][1]
        return AnswerSample(
            answer_text=answer_text,
            char_st=char_st,
            char_ed=char_ed,
            st=st,
            ed=ed,
            answer_bio_ids=answer_bio_ids,
            answer_pos_tag=chunk_pos_tag,
            answer_ner_tag=chunk_ner_tag,
        )

    sampled_answers = [
        answer for answer in (process_chunk(chunklist[chunk_id]) for chunk_id in sampled_chunk_ids) if answer
    ]
    return sampled_answers[:config.num_sample_answer]


def select_questions(ans: AnswerSample, sample_probs, config=InfoConfig()):
    """
    Select question styles based on the answer's POS and NER tags, given sample probabilities.

    Args:
        ans (AnswerSample): A tuple containing information about the answer, including its text, indices, BIO tags,
                            POS tag, and NER tag.
        sample_probs (dict): A dictionary containing the probabilities of different question styles.
        config (InfoConfig, optional): A configuration object containing the maximum number of sampling attempts and
                                       the desired number of question styles to sample.

    Returns:
        list: A list of sampled question styles.
    """
    a_tag = "-".join([ans.answer_pos_tag, ans.answer_ner_tag])

    # Get style probabilities
    s_probs = [
        sample_probs["s|c,a"].get("_".join([s, a_tag]), 1) for s in QUESTION_TYPES
    ]

    # Sample question styles
    sampled_styles = []
    for _ in range(config.max_sample_times):
        sampled_s = weighted_sample(QUESTION_TYPES, s_probs)
        if sampled_s not in sampled_styles:
            sampled_styles.append(sampled_s)
            if len(sampled_styles) >= config.num_sample_style:
                break

    return sampled_styles


def select_clues(chunklist, doc: SpacyDoc, ans: AnswerSample, sample_probs, config=InfoConfig()):
    """
    Select clues from a list of chunks based on the dependency distance and the probability of the chunk given the answer.

    Args:
        chunklist (list): A list of chunks, each containing NER tag, POS tag, text, start index, and end index.
        doc (spacy.tokens.Doc): A SpaCy document containing the tokens of the sentence.
        ans (AnswerSample): A tuple containing information about the answer, including its text, indices, BIO tags,
                            POS tag, and NER tag.
        config (InfoConfig, optional): A configuration object containing the maximum number of sampling attempts and
                                       the desired number of clues to sample.

    Returns:
        list: A list of sampled clues, with each clue containing its text and binary ids.
    """
    st, idx2related = ans.st, get_dependency_paths(doc)[1]
    context_tokens = [token.text for token in doc]

    # Calculate chunk probabilities
    c_probs = []
    for chunk in chunklist:
        c_tag = "-".join([chunk[1], chunk[0]])
        dep_dist = min(abs(chunk[3] - st), min(len(path)
                       for tk_id, path in idx2related[st] if tk_id == chunk[3]))
        dep_dist_bin = value_to_bin(dep_dist, config.clue_dep_dist_min_val,
                                    config.clue_dep_dist_max_val, config.clue_dep_dist_bin_width)

        if chunk[2][0].lower() not in NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE:
            c_probs.append(sample_probs["c|a"].get(
                "_".join([c_tag, str(dep_dist_bin)]), 1))
        else:
            c_probs.append(1)

    # Sample clues
    chunk_ids = list(range(len(chunklist)))
    sampled_clue_chunk_ids = []
    for _ in range(config.max_sample_times):
        sampled_chunk_id = weighted_sample(chunk_ids, c_probs)
        if sampled_chunk_id not in sampled_clue_chunk_ids:
            sampled_clue_chunk_ids.append(sampled_chunk_id)
            if len(sampled_clue_chunk_ids) >= config.num_sample_clue:
                break

    # Extract clue details
    sampled_clues = []
    for chunk_id in sampled_clue_chunk_ids:
        chunk = chunklist[chunk_id]
        clue_start, clue_end = chunk[3], chunk[4]
        clue_text = " ".join(context_tokens[clue_start: clue_end + 1])
        clue_binary_ids = [0] * len(doc)
        clue_binary_ids[clue_start: clue_end + 1] = [1] * \
            (clue_end - clue_start + 1)
        sampled_clues.append(ClueSample(clue_text=clue_text,
                             clue_binary_ids=clue_binary_ids))

    return sampled_clues


def select(sentence, sample_probs, config=InfoConfig()):
    sampled_infos = []
    chunklist = chunks(sentence)
    doc = NLP(sentence)
    for ans in select_answers(chunklist, sentence, sample_probs):
        info = {
            "answer": {
                "answer_text": ans.answer_text,
                "char_start": ans.char_st,
                "char_end": ans.char_ed,
                "answer_bio_ids": ans.answer_bio_ids,
                "answer_chunk_tag": ans.answer_pos_tag,
            },
            "styles": None,
            "clues": None,
        }
        # sample styleselect
        styles = select_questions(ans, sample_probs)
        info["styles"] = list(styles)

        # sample clue
        selected_clues = select_clues(chunklist, doc, ans, sample_probs)
        info["clues"] = selected_clues

        sampled_infos.append(info)

    result = {
        "context": sentence,
        "selected_infos": sampled_infos,
        "ans_sent_doc": doc.text,
    }
    return result


def read_sample_probs(sample_probs_path):
    with open(sample_probs_path, "rb") as f:
        sample_probs = pkl.load(f)
    return sample_probs

def select_with_default_sampel_probs(sentence):
    sample_probs_path = os.path.join(current_file_dir,"sample_probs.pkl")
    sample_probs = read_sample_probs(sample_probs_path)
    selection = select(
        "Bob is eating a delicious cake in Vancouver.", sample_probs)
    return selection

def test():
    sample_probs_path = "botiverse/Theorizer/squad/sample_probs.pkl"
    sample_probs = read_sample_probs(sample_probs_path)
    selection = select(
        "Bob is eating a delicious cake in Vancouver.", sample_probs)
    print(selection)


if __name__ == "__main__":
    test()
