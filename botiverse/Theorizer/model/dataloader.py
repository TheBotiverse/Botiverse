from dataclasses import dataclass
import os
from collections import defaultdict, namedtuple
from typing import Dict, List
import torch

from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel

from tqdm import tqdm
import pickle as pkl
import numpy as np
from .utils import *
from botiverse.Theorizer.squad.squad_example import SquadExample, SquadProcessedExample
from transformers import (
    WEIGHTS_NAME,
    CONFIG_NAME,
    AdamW,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    get_linear_schedule_with_warmup,
)

SPECIAL_TOKENS = [
    "<sos>",
    "<eos>",
    "<paragraph>",
    "<clue>",
    "<answer>",
    "<style>",
    "<question>",
    "<pad>",
]
SPECIAL_TOKENS_DICT = {
    "bos_token": "<sos>",
    "eos_token": "<eos>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
    "additional_special_tokens": [
        "<paragraph>",
        "<clue>",
        "<answer>",
        "<style>",
        "<question>",
    ],
}
IGNORE_VALUE_OF_LM_HEADS = -100


@dataclass
class SquadGPT2Example:
    """
    A single example for the SQuAD dataset, processed for GPT-2 training.
    """

    input_ids: List[int]
    attention_mask: List[int] = None
    token_type_ids: List[int] = None
    lm_labels: List[int] = None


def prepare_squad_data_for_gpt2(tokenizer: GPT2Tokenizer, processed_examples: List[SquadProcessedExample]) -> List[SquadGPT2Example]:
    SOS, EOS, PARAGRAPH, CLUE, ANSWER, STYLE, QUESTION = tokenizer.convert_tokens_to_ids(
        SPECIAL_TOKENS[:-1])
    truncated_sequences = 0
    prepared_data = []

    for inst in tqdm(processed_examples):
        # Tokenize context, question, answer, question_type, and clue if available
        tokenized_context = tokenizer.encode(inst.context_text)
        tokenized_question = tokenizer.encode(inst.question_text)
        tokenized_answer = tokenizer.encode(inst.answer_text)
        tokenized_ans_prefix = tokenizer.encode(
            inst.context_text[: inst.answer_start + 1])
        tokenized_qtype = tokenizer.encode(inst.question_type)

        clue_exist = inst.clue_start is not None
        if clue_exist:
            tokenized_clue = tokenizer.encode(inst.clue_text)
            tokenized_clue_prefix = tokenizer.encode(
                inst.context_text[: inst.clue_start + 1])
        else:
            tokenized_clue = []

        # Calculate the total sequence length
        total_seq_len = (
            len(tokenized_context)
            + len(tokenized_answer)
            + len(tokenized_question)
            + len(tokenized_clue)
            + len(tokenized_qtype)
            + 6  # 6 special tokens, without pad or eos
        )

        # Truncate the sequence if it exceeds the GPT-2 model's input size
        if total_seq_len > tokenizer.max_model_input_sizes["gpt2"]:
            tokenized_context = tokenized_context[
                : -1 * (total_seq_len - tokenizer.max_model_input_sizes["gpt2"] + 1)
            ]
            truncated_sequences += 1

        # Calculate answer and clue positions in the tokenized context
        answer_position_tokenized = get_overlap_position(
            tokenized_context, tokenized_answer, tokenized_ans_prefix)
        if clue_exist:
            clue_position_tokenized = get_overlap_position(
                tokenized_context, tokenized_clue, tokenized_clue_prefix)
        else:
            clue_position_tokenized = (None, None)

        # Build input sequence and token_type_ids
        sequence = [SOS] + tokenized_context + [ANSWER] + tokenized_answer + [CLUE] + \
            tokenized_clue + [STYLE] + tokenized_qtype + \
            [QUESTION] + tokenized_question + [EOS]
        token_types = np.full(
            len(sequence), IGNORE_VALUE_OF_LM_HEADS, dtype=int)
        token_types[:len(tokenized_context) + 1] = PARAGRAPH
        token_types[answer_position_tokenized[0] +
                    1:answer_position_tokenized[1] + 1] = ANSWER
        if clue_exist:
            token_types[clue_position_tokenized[0] +
                        1:clue_position_tokenized[1] + 1] = CLUE
        token_types[len(tokenized_context) + len(tokenized_answer) + 2:len(
            tokenized_context) + len(tokenized_answer) + len(tokenized_clue) + 3] = CLUE
        token_types[len(tokenized_context) + len(tokenized_answer) + len(tokenized_clue) + 3:len(
            tokenized_context) + len(tokenized_answer) + len(tokenized_clue) + len(tokenized_qtype) + 4] = STYLE
        token_types[len(tokenized_context) + len(tokenized_answer) +
                    len(tokenized_clue) + len(tokenized_qtype) + 4:-1] = QUESTION

        # Build lm_labels
        lm_labels = np.full(len(sequence), IGNORE_VALUE_OF_LM_HEADS, dtype=int)
        lm_labels[-len(tokenized_question) - 1:-1] = tokenized_question
        lm_labels[-1] = EOS

        # Create a data point
        # TODO: Add attention_mask
        # I wrongly decided to use numpy arrays instead of lists for the data points, so I have to convert them back to lists here
        data_point = SquadGPT2Example(
            input_ids=list(sequence),
            token_type_ids=list(token_types),
            lm_labels=list(lm_labels),
        )
        # Add the data point to the prepared_data list
        prepared_data.append(data_point)

    return prepared_data


def prepare_and_pad_squad_data_for_gpt2(tokenizer: GPT2Tokenizer, processed_examples: List[SquadProcessedExample], max_len: int = None, padding: int = 0) -> List[SquadGPT2Example]:
    """
    Prepare and pad SQuAD data for GPT-2.

    This function tokenizes and processes the input data, builds the input sequence, token_type_ids, and lm_labels suitable for training GPT-2.
    It then pads the sequences to the maximum length in the dataset with the specified padding value.

    Args:
        tokenizer (GPT2Tokenizer): The GPT-2 tokenizer used to tokenize the text.
        processed_examples (List[SquadProcessedExample]): A list of SquadProcessedExample instances containing the processed SQuAD data.
        padding (int, optional): The padding value to use when padding the sequences. Default is 0.

    Returns:
        Dict[str, List[List[int]]]: A dictionary containing the prepared and padded data points with keys 'input_ids', 'token_type_ids', and 'lm_labels'.
    """
    prepared_data = prepare_squad_data_for_gpt2(tokenizer, processed_examples)

    max_l = max(len(x.input_ids) for x in prepared_data) if max_len is None else max_len

    data_point:SquadGPT2Example
    for data_point in prepared_data:
        data_point.input_ids = data_point.input_ids + [padding] * (max_l - len(data_point.input_ids))
        data_point.token_type_ids = data_point.token_type_ids + [padding] * (max_l - len(data_point.token_type_ids))
        data_point.lm_labels = data_point.lm_labels + [IGNORE_VALUE_OF_LM_HEADS] * (max_l - len(data_point.lm_labels))

    return prepared_data


def from_dict_to_squad_processed_example(data: Dict) -> SquadProcessedExample:
    """
    Convert a dictionary to a SquadProcessedExample instance.

    Args:
        data (Dict): A dictionary containing the data to convert.

    Returns:
        SquadProcessedExample: The SquadProcessedExample instance containing the data.
    """
    return SquadProcessedExample(
        context_text=data["paragraph"],
        question_text=data["question"],
        question_type=data["ques_type"],
        answer_text=data["answer"],
        answer_start=data["answer_start"],
        clue_text=data["clue"],
        clue_start=data["clue_start"],
        para_id=data["para_id"],
    )

def read_cached_processed_examples(filepath: str) -> List[SquadProcessedExample]:
    with open(filepath, "rb") as f:
        exmaples = pkl.load(f)
        exmaples = list(map(from_dict_to_squad_processed_example, exmaples))
    return exmaples

def test_getdataset():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_tokens(SPECIAL_TOKENS)
    with open("squad/dataset/train.processed.pkl", "rb") as f:
        data = pkl.load(f)
        data = list(map(from_dict_to_squad_processed_example, data))
    data = prepare_and_pad_squad_data_for_gpt2(tokenizer, data, max_len=512)
    print(data[0])
    print(tokenizer.decode(data[0].input_ids))
    # paragraph = tokenizer.decode(data[0]["paragraph"])
    # print(data[0])
    # print(paragraph)

def test_tokenizer():
    a= [50257, 1026, 318, 257, 30069, 286, 262, 7128, 33955, 379, 406, 454, 8906, 11, 4881, 810, 262, 5283, 5335, 1128, 7241, 306, 4120, 284, 9281, 6206, 324, 5857, 311, 12944, 343, 516, 287, 1248, 3365, 13, 50261, 48615, 6206, 324, 5857, 311, 12944, 343, 516, 50260, 1169, 5283, 5335, 50262, 8727, 50263, 2514, 4150, 750, 262, 5283, 5335, 7910, 1656, 287, 1248, 3365, 287, 406, 454, 8906, 4881, 30, 50258]

    b= [50257, 1026, 318, 257, 30069, 286, 262, 7128, 33955, 379, 406, 454, 8906, 11, 4881, 810, 262, 5283, 5335, 1128, 7241, 306, 4120, 284, 9281, 6206, 324, 5857, 311, 12944, 343, 516, 287, 1248, 3365, 13, 50261, 48615, 6206, 324, 5857, 311, 12944, 343, 516, 50260, 1169, 5283, 5335, 50262, 8727, 50263, 2514, 4150, 750, 262, 5283, 5335, 7910, 1656, 287, 1248, 3365, 287, 406, 454, 8906, 4881, 30,50258]
    print(a==b)
    tokenizer=GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.add_tokens(SPECIAL_TOKENS)
    print(tokenizer.decode([8241]))
    print(tokenizer.decode([8727]))

if __name__ == "__main__":
    test_getdataset()

# TODO: remove this one after testing the above one for correctness, this one is for historical reasons
def prepare_squad_for_gpt2(tokenizer: GPT2Tokenizer, processed_examples: List[SquadProcessedExample], split) -> List[Dict]:

    truncated_sequences = 0
    inst: SquadProcessedExample
    for inst in tqdm(processed_examples):

        tokenized_context = tokenizer.encode(inst.context_text)
        tokenized_question = tokenizer.encode(inst.question_text)
        tokenized_answer = tokenizer.encode(inst.answer_text)
        tokenized_ans_prefix = tokenizer.encode(
            inst.context_text[: inst.answer_start + 1])
        tokenized_qtype = tokenizer.encode(inst.question_type)

        clue_exist = inst.clue_start is not None
        if clue_exist:
            tokenized_clue = tokenizer.encode(inst.clue_text)
            tokenized_clue_prefix = tokenizer.encode(
                inst.context_text[: inst.clue_start + 1])
        else:
            tokenized_clue = []

        total_seq_len = (
            len(tokenized_context)
            + len(tokenized_answer)
            + len(tokenized_question)
            + len(tokenized_clue)
            + len(tokenized_qtype)
            + 6  # 6 special tokens, without pad or eos
        )

        if total_seq_len > tokenizer.max_model_input_sizes["gpt2"]:
            # Heuristic to chop off extra tokens in paragraphs
            tokenized_context = tokenized_context[
                : -1 * (total_seq_len - tokenizer.max_model_input_sizes["gpt2"] + 1)
            ]
            truncated_sequences += 1
            assert (
                len(tokenized_context)
                + len(tokenized_answer)
                + len(tokenized_question)
                + len(tokenized_clue)
                + len(tokenized_qtype)
                + 6
                < tokenizer.max_model_input_sizes["gpt2"]
            )

        ans_prefix_ids = tokenizer.encode(tokenized_ans_prefix)
        answer_position_tokenized = get_overlap_position(
            tokenized_context, tokenized_answer, ans_prefix_ids
        )

        if clue_exist:
            clue_position_tokenized = get_overlap_position(
                tokenized_context, tokenized_clue, tokenized_clue_prefix
            )
        data = {
            "paragraph": tokenized_context,
            "question": tokenized_question,
            "answer": tokenized_answer,
            "answer_position_tokenized": answer_position_tokenized,
            "style": tokenized_qtype,
        }

    return data



