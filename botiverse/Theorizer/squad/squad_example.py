from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Tuple
from tqdm import tqdm
from .utils import *
from .info_extractor import SquadAugmentedExample, InfoConfig, extract_clue_and_question_info, extract_clue, extract_question_type_and_id, NLP
from collections import Counter
import os
import multiprocess as mp
import math
import numpy as np
import pickle as pkl

@dataclass
class SquadExample:
    """
    A single example for the Squad-Zhou dataset, as loaded from disk.
    """
    context_text: str
    question_text: str
    answer_text: str
    answer_start: int


@dataclass
class SquadProcessedExample:
    """
    A single example for the processed SQuAD Zhou dataset, used for training.
    """
    context_text: str
    question_text: str
    question_type: str
    answer_text: str
    answer_start: int
    clue_text: str
    clue_start: int
    para_id: int


def read_squad_examples(input_file: str) -> List[SquadExample]:
    """
    Read a SQuAD Zhou text file into a list of SquadExample.
    """
    raw_examples = []
    with open(input_file, encoding="utf-8") as fh:
        lines = fh.readlines()
        for line in tqdm(lines):
            fields = line.strip().split("\t")
            input_sent, answer_text, question_text = fields[6], fields[8], fields[9]
            answer_start_token = int(fields[1].split()[0])

            # Calculate token spans and answer start in the tokenized sentence
            token_spans = find_token_spans_in_text(
                fields[0], fields[0].split())
            answer_start_in_tokenized_sent = token_spans[answer_start_token][0]

            # Find all answer spans in the input sentence
            answer_spans = match_spans(answer_text, input_sent)
            # skip if no answer spans are found, the answer does does not exist in the input sentence
            if len(answer_spans) == 0:
                continue
            # Find the closest answer span to the answer start in the tokenized sentence
            answer_start = min(
                answer_spans,
                key=lambda span: abs(span[0] - answer_start_in_tokenized_sent),
            )[0]

            example = SquadExample(
                question_text=normalize(question_text),
                context_text=normalize(input_sent),
                answer_text=normalize(answer_text),
                answer_start=answer_start,

            )
            raw_examples.append(example)

    return raw_examples


def create_squad_example_with_info(raw_ex: List[SquadExample]) -> List[SquadAugmentedExample]:
    """
    Augment the raw examples with question-type and clue info.
    """

    num_process = 1
    start_index = 0
    end_index = len(raw_ex)
    batch_size = len(raw_ex) // num_process


    def task(j):
        start = start_index + j * batch_size
        end = min(start_index + (j + 1) * batch_size, end_index)
        examples = []
        e: SquadExample
        for e in tqdm(raw_ex[start:end], desc=f"Process {j}", position=j, leave=False):
            new_e = extract_clue_and_question_info(
                sentence=e.context_text, question=e.question_text, answer=e.answer_text, answer_start=e.answer_start)
            examples.append(new_e)
        return examples
    
    # examples_list = []
    # with mp.Pool(num_process) as pool:
    #     examples_list = pool.map(task, range(num_process))
    
    examples_with_info = task(0)
    # for e in examples_list:
        # examples_with_info += e

    return examples_with_info


def calculate_probability_distribution(augmented_examples: List[SquadAugmentedExample]) -> Dict[str, Counter]:
    """
    Calculates the probability distribution of answer, clue, and sentence based on the given list of augmented examples.

    The probability distribution is defined as:
    P(a, c, s) = p(a) * p(c|a) * p(s|c, a)
               = p(a|a_tag, a_length) * p(c|c_tag, dep_dist) * p(s|a_tag)

    Args:
        augmented_examples (List[SquadAugmentedExample]): A list of SquadAugmentedExample objects.

    Returns:
        Dict[str, Counter]: A dictionary containing the probability distribution of answer, clue, and sentence.
    """

    """
    Disclaimer this function is adapted from the original implementation
    """
    sla_tag = []
    clc_tag_dep_dist = []
    ala_tag_a_length = []

    for e in tqdm(augmented_examples):
        a_tag = "-".join([e.answer_pos_tag, e.answer_ner_tag])
        s = e.question_type  # question style (type)
        a_length = e.answer_length
        a_length_bin = value_to_bin(
            a_length,
            InfoConfig.ans_len_min_val,
            InfoConfig.ans_len_max_val,
            InfoConfig.ans_len_bin_width,
        )
        c_tag = "-".join([e.clue_info.clue_pos_tag, e.clue_info.clue_ner_tag])
        dep_dist = e.clue_info.clue_answer_dep_path_len
        dep_dist_bin = value_to_bin(
            dep_dist,
            InfoConfig.clue_dep_dist_min_val,
            InfoConfig.clue_dep_dist_max_val,
            InfoConfig.clue_dep_dist_bin_width,
        )
        sla_tag.append("_".join([s, a_tag]))
        clc_tag_dep_dist.append("_".join([c_tag, str(dep_dist_bin)]))
        ala_tag_a_length.append("_".join([a_tag, str(a_length_bin)]))

    sla_tag = Counter(sla_tag)
    clc_tag_dep_dist = Counter(clc_tag_dep_dist)
    ala_tag_a_length = Counter(ala_tag_a_length)
    sample_probs = {
        "a": ala_tag_a_length,
        "c|a": clc_tag_dep_dist,
        "s|c,a": sla_tag,
    }

    return sample_probs


def create_process_squad_examples(raw_ex: List[SquadExample]):
    """
    Get a list of spaCy processed examples.
    """

    raw_ex = list(enumerate(raw_ex))
    start_index = 0
    end_index = len(raw_ex)
    batch_size = 10000
    num_batches = math.ceil((end_index - start_index) / batch_size)

    def task(j):
        start = start_index + j * batch_size
        end = min(start_index + (j + 1) * batch_size, end_index)
        examples = []
        e: SquadExample
        for pid, e in tqdm(raw_ex[start:end], desc=f"Process {j}", position=j, leave=False):

            context_spacydoc = NLP(e.context_text)
            context_tokens = [token.text for token in context_spacydoc]
            spans = find_token_spans_in_text(e.context_text, context_tokens)

            question_spacydoc = NLP(e.question_text)
            ques_type, ques_type_id = extract_question_type_and_id(
                e.question_text)

            answer_start = e.answer_start
            answer_end = answer_start + len(e.answer_text)

            answer_span = []
            for idx, span in enumerate(spans):
                if not (answer_end <= span[0] or answer_start >= span[1]):
                    answer_span.append(idx)

            y1_in_sent, y2_in_sent = answer_span[0], answer_span[-1]
            answer_in_sent = " ".join(
                context_tokens[y1_in_sent: y2_in_sent + 1])

            clue_info = extract_clue(
                e.context_text, e.question_text, answer_in_sent, y1_in_sent)

            sent_limit = InfoConfig.sent_limit
            clue_token_position = np.where(
                clue_info.padded_selected_clue_binary_ids == 1)[0]

            if len(clue_token_position) > 0 and clue_info.clue_chunk:
                start_idx = clue_token_position[0]
                end_idx = clue_token_position[-1]
                if len(spans) <= end_idx:
                    # num_spans_len_error += 1
                    clue_text, clue_start = None, None
                else:
                    start, end = spans[start_idx][0], spans[end_idx][1]
                    clue_text = e.context_text[start:end]
                    clue_tokenized_text = " ".join(
                        clue_info.clue_chunk[2])
                    if clue_text != clue_tokenized_text:
                        clue_start = e.context_text.find(clue_tokenized_text)
                        clue_text = clue_info.clue_chunk[2][0]
                        if clue_start < 0:
                            continue
                    else:
                        clue_start = start

            else:
                clue_text, clue_start = None, None

            example = SquadProcessedExample(
                context_text=e.context_text,
                question_text=e.question_text,
                answer_text=e.answer_text,
                answer_start=e.answer_start,
                question_type=ques_type,
                clue_text=clue_text,
                clue_start=clue_start,
                para_id= pid
            )
            examples.append(example)
        return examples

    acc = []
    # with mp.Pool(4) as pool:
        # for processed_examples in pool.imap(task, range(num_batches)):
            # acc += processed_examples
    acc = task(0)
    return acc


def pipeline(input_file: str):
    """
    Pipeline for processing squad examples.
    """
    mp.set_start_method("spawn")

    raw_ex = read_squad_examples(input_file)
    raw_ex = raw_ex[:1000]

    processed_ex = create_process_squad_examples(raw_ex)
    print(processed_ex[:6])
    with_info_ex = create_squad_example_with_info(raw_ex[:1000])

    sample_probs = calculate_probability_distribution(with_info_ex)

    return sample_probs


if __name__ == "__main__":
    data = pipeline("botiverse/Theorizer/squad/dataset/train.txt")
    print(data)