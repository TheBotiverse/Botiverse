from typing import List, Set, Tuple, NewType
import numpy as np
import copy
from nltk.tree import Tree
import nltk
import spacy
import benepar
from dataclasses import dataclass
from .utils import *



@dataclass
class ClueInfo:
    clue_pos_tag: str
    clue_ner_tag: str
    clue_length: int
    clue_chunk: Tuple[str, str, List[str], int, int]
    clue_answer_dep_path_len: int
    padded_selected_clue_binary_ids: np.ndarray


QuestionType = NewType('QuestionType', str)


@dataclass
class SquadAugmentedExample:
    """
    A single training/test example for the Squad-Zhou dataset, after augmenting with clue info and question style.
    """
    context_text: str
    question_text: str
    answer_text: str
    question_type: QuestionType
    answer_pos_tag: str
    answer_ner_tag: str
    answer_length: int
    clue_info: ClueInfo


def __navigate(node: nltk.Tree):
    """
    Recursively navigates through an NLTK parse tree and extracts information about
    the tree depth, number of words, and chunks with their respective positions.

    Args:
        node (nltk.Tree): An NLTK parse tree node, generated from a constituency parser.

    Returns:
        tuple: A tuple containing the following elements:
            - max_depth (int): The maximum depth of the parse tree.
            - word_num (int): The total number of words in the parse tree.
            - chunklist (list): A list of tuples where each tuple contains the chunk's
                                POS tag and a list of its leaves with their respective
                                positions in the sentence.
    """

    """
    Disclaimer: This function is adapted from the original implementation with changes for readability, clarity and documentation
    """
    if type(node) is not Tree:
        return 1, 1, [("word", (node, 0))]

    # process leaf nodes
    for idx, _ in enumerate(node.leaves()):
        tree_location = node.leaf_treeposition(idx)
        non_terminal = node[tree_location[:-1]]
        if type(non_terminal[0]) is not str:
            non_terminal[0] = non_terminal[0] + [idx]
        else:
            non_terminal[0] = [non_terminal[0], idx]

    # process non-leaf nodes
    max_depth, word_num, chunklist = 0, 0, []
    for child in node:
        child_depth, child_num, child_chunklist = __navigate(child)
        max_depth = max(max_depth, child_depth + 1)
        word_num += child_num
        chunklist += child_chunklist

    # process current node
    cur_node_chunk = [(node.label(), node.leaves())]
    chunklist += cur_node_chunk
    return max_depth, word_num, chunklist


def chunks(sentence: str) -> List[Tuple[str, str, List[str], int, int]]:
    """
    Takes a sentence and returns a list of chunks with their respective NER tags,
    POS tags, words, and start and end positions in the sentence.

    Args:
        sentence (str): The input sentence to be parsed and chunked.

    Returns:
        list: A list of tuples where each tuple contains the following elements:
            - chunk_ner_tag (str): The NER tag of the chunk (e.g., 'PERSON', 'ORG').
            - chunk_pos_tag (str): The POS tag of the chunk (e.g., 'NP', 'VP').
            - leaves_without_position (list): A list of words in the chunk.
            - start (int): The start position of the chunk in the sentence.
            - end (int): The end position of the chunk in the sentence.
    """

    """
    Disclaimer: This function is adapted from the original implementation with changes for readability, clarity and documentation
    """
    tree = PARSER.parse(sentence)
    max_depth, node_num, orig_chunklist = __navigate(tree)
    spacy_document = NLP(sentence)
    chunklist = []
    for chunk in orig_chunklist:
        try:
            if chunk[0] == "word":
                continue
            chunk_pos_tag, leaves = chunk
            leaves_without_position = []
            position_list = []

            for v in leaves:
                if type(v) == list:  # Check if v is a list
                    wd = v[0]
                    idx = v[1:]
                    leaves_without_position.append(wd)
                    position_list.append(idx[0])
                else:
                    leaves_without_position.append(v)

            st = position_list[0]
            ed = position_list[-1]
            chunk_ner_tag = "UNK"
            chunk_text = " ".join(leaves_without_position)

            for ent in spacy_document.ents:
                if ent.text == chunk_text or chunk_text in ent.text:
                    chunk_ner_tag = ent.label_

            chunklist.append(
                (chunk_ner_tag, chunk_pos_tag, leaves_without_position, st, ed)
            )
        except:
            continue

    return chunklist


def __dfs(token_list, current_token, current_path, max_depth, related_tokens):
    if len(current_path) > max_depth:
        return
    if current_token in related_tokens and len(related_tokens[current_token]) <= len(current_path):
        return
    related_tokens[current_token] = current_path
    for token in token_list:
        if token.i != current_token:
            continue
        new_path = copy.deepcopy(current_path)
        try:
            new_path.append(token.dep_)
        except:
            continue
        __dfs(token_list, token.head.i, new_path, max_depth, related_tokens)
        for child in token.children:
            new_path = copy.deepcopy(current_path)
            new_path.append(child.head.dep_)
            __dfs(token_list, child.i, new_path, max_depth, related_tokens)


def get_dependency_paths(token_list: List[SpacyToken]):
    """
    Given a list of spaCy tokens, extract the dependency paths between different tokens.

    Args:
        doc (spacy.tokens.Doc): A spaCy document.

    Returns:
        dict: A dictionary mapping token indices to tokens.
        dict: A dictionary mapping token indices to related tokens and their dependency paths.
        list: A list of token texts.
    """
    index_to_token = {token.i: token for token in token_list}

    index_to_related = {}
    tokens = [token.text for token in token_list]

    for token in token_list:
        related_tokens = {}
        __dfs(
            token_list,
            token.i,
            [],
            len(token_list) - 1,
            related_tokens,
        )
        sorted_related = sorted(related_tokens.items(),
                                key=lambda x: len(x[1]))
        index_to_related[token.i] = sorted_related

    return index_to_token, index_to_related, tokens


def __tokenize_and_stem(sentence: str, condition_list: List[bool] = None) -> Tuple[List[str], List[str], List[SpacyToken]]:
    """
    Takes a sentence and returns a list of tokens, a list of stemmed tokens and a list of spacy tokens, only with the content words.

    Args:
        sentence (str): The input sentence to be tokenized and stemmed.

    Returns:
        tuple: A tuple of two lists where the first list contains the tokens and the second list contains the stemmed tokens.
    """

    if condition_list is not None:
        sentence_tokens = [
            token for i, token in enumerate(sentence) if condition_list[i]]
        spacy_tokens = None
        sentence_lemmas = None
    else:
        spacy_doc = NLP(sentence)
        spacy_tokens = [token for token in spacy_doc]
        
        sentence_tokens = [token.text for token in spacy_doc]
        sentence_lemmas = [token.lemma_ for token in spacy_doc]
    return sentence_tokens, sentence_lemmas, spacy_tokens


def __number_of_overlapping_tokens(sentence_tokens1: List[str], sentence_tokens2: List[str], condition_list: List[bool]) -> int:
    conditioned_sentence_1 = [tk for i, tk in enumerate(
        sentence_tokens1) if condition_list[i]]

    conditioned_sentence1_inetersect_sentence2 = [
        tk for tk in conditioned_sentence_1 if tk in sentence_tokens2]
    return len(conditioned_sentence1_inetersect_sentence2)


def extract_clue(sentence: str, question: str, answer: str, answer_start: int, config: InfoConfig = InfoConfig()) -> ClueInfo:
    """
    Given a sentence, question, answer, and the answer's starting position,
    this function extracts information about the clues related to the answer.

    Args:
        sentence (str): The sentence containing the answer.
        question (str): The question being asked.
        answer (str): The correct answer.
        answer_start (int): The starting position of the answer in the sentence.

    Returns:
        A Clue Info object holding all the clue information.
    """

    sentence = sentence.lower()
    question = question.lower()
    chunklist = chunks(sentence)

    sentence_tokens, sentence_lemmas, sentence_spacy_tokens = __tokenize_and_stem(
        sentence)
    question_tokens, question_lemmas, question_spacy_tokens = __tokenize_and_stem(
        question)
    idx2token, idx2related, context_tokens = \
        get_dependency_paths(sentence_spacy_tokens)

    
    # spans = find_token_spans_in_text(sentence, sentence_tokens)
    # answer_end = answer_start + len(answer)
    # answer_span = [idx for idx, span in enumerate(spans) if not (
    #     answer_end <= span[0] or answer_start >= span[1])]
    # y = answer_span[0]
    # answer_start = y

    clue_scores = []
    for chunk in chunklist:
        chunk_ner_tag, chunk_pos_tag, chunk_words, chunk_start, chunk_end = chunk

        chunk_content_words = [
            bool(word.lower() not in FUNCTION_WORDS) for word in chunk_words]
        chunk_text = " ".join(chunk_words).lower()

        chunk_tokens, _, _ = __tokenize_and_stem(
            chunk_words, chunk_content_words)
        chunk_lemmas = sentence_lemmas[chunk_start:chunk_end+1]
        chunk_lemmas = [lemma for i, lemma in enumerate(
            chunk_lemmas) if chunk_content_words[i]]
        no_tc = __number_of_overlapping_tokens(
            chunk_tokens, question_tokens, chunk_content_words)
        no_mc = __number_of_overlapping_tokens(
            chunk_lemmas, question_lemmas, chunk_content_words)
        binary_x = int(chunk_text in question)
        score = 0
        # chunk_lemmas_is_subset_from_question_lemmas = len(set(chunk_lemmas) & set(question_lemmas)) == len(set(chunk_lemmas))
        chunk_lemmas_is_subset_from_question_lemmas = \
            set(chunk_lemmas) <= set(question_lemmas) or \
            set(chunk_tokens) <= set(question_tokens)
        if chunk_lemmas_is_subset_from_question_lemmas and chunk_words[0].lower() not in NOT_BEGIN_TOKENS_FOR_ANSWER_CLUE and sum(map(lambda x: int(x), chunk_content_words)) > 0:
            score = no_tc*2+no_mc+binary_x
        clue_scores.append(score)

    # no clues were found
    padded_selected_clue_binary_ids = np.zeros([InfoConfig.sent_limit], dtype=np.float32)
    if not clue_scores or max(clue_scores) == 0:
        clue_chunk = None
        clue_pos_tag = "UNK"
        clue_ner_tag = "UNK"
        clue_length = 0
        clue_answer_dep_path_len = -1
    else:
        clue_chunk = clue_ner_tag, clue_pos_tag, chunk_words, clue_start, clue_end = \
            chunklist[clue_scores.index(max(clue_scores))]

        clue_answer_dep_path_len = abs(clue_start - answer_start)
        answer_related = idx2related[answer_start]
        clue_length = clue_end - clue_start + 1
        for tk_id, path in answer_related:
            if tk_id == clue_start:
                clue_answer_dep_path_len = len(path)

        if clue_start < InfoConfig.sent_limit and clue_end < InfoConfig.sent_limit:
                padded_selected_clue_binary_ids[clue_start: clue_end + 1] = 1

    return ClueInfo(
        clue_pos_tag=clue_pos_tag,
        clue_ner_tag=clue_ner_tag,
        clue_length=clue_length,
        clue_chunk=clue_chunk,
        clue_answer_dep_path_len=clue_answer_dep_path_len,
        padded_selected_clue_binary_ids=padded_selected_clue_binary_ids
    )


def extract_question_type_and_id(question, config: InfoConfig = InfoConfig()) -> Tuple[QuestionType, int]:
    """
    Given a question string, returns its type and associated id.

    Args:
        question (str): A question string.

    Returns:
        tuple: A tuple containing the question type (str) and its id (int).
    """

    # Split the question into words
    words = question.split()

    # Check if the question is an informational question
    for word in words:
        for i, question_type in enumerate(QUESTION_WORDS):
            if question_type.lower() in word.lower():
                return question_type, Q_TYPE2ID_DICT[question_type]

    # Check if the question is a boolean question
    if words[0].lower() in (q_type.lower() for q_type in BOOL_QUESTION_WORDS):
        return "boolean", Q_TYPE2ID_DICT["boolean"]

    # Return "other" if the question type is not found
    return "other", Q_TYPE2ID_DICT["other"]


def extract_clue_and_question_info(sentence: str, question: str, answer: str, answer_start: int, config: InfoConfig = InfoConfig) -> SquadAugmentedExample:
    """
    Extracts information about the question, answer, and clue from the provided sentence,
    question, and answer.

    Args:
        sentence (str): The sentence containing the answer.
        question (str): The question being asked.
        answer (str): The answer to the question.
        answer_start (int): The character index of the answer's start position in the sentence.
        config: A configuration object containing token limits and clue extraction settings.

    Returns:
        SquadAugmentedExample object containing extracted information about the question, answer, and clue.
    """

    # Process the input sentence and chunks
    sentence_spacydoc = NLP(sentence)
    chunklist = chunks(sentence)

    # Extract answer information
    answer_pos_tag, answer_ner_tag = "UNK", "UNK"
    for chunk in chunklist:
        if answer == " ".join(chunk[2]):
            answer_ner_tag = chunk[0]
            answer_pos_tag = chunk[1]
            break
    answer_length = len(answer.split())

    # Extract question type and id
    question_type, question_id = extract_question_type_and_id(question)

    # Get the answer start token index
    ans_sent_tokens = [token.text for token in sentence_spacydoc]
    spans = find_token_spans_in_text(sentence, ans_sent_tokens)
    answer_end = answer_start + len(answer)
    answer_span = []
    for idx, span in enumerate(spans):
        if not (answer_end <= span[0] or answer_start >= span[1]):
            answer_span.append(idx)
    answer_start_idx = answer_span[0]
    # Extract clue information
    clue_info = extract_clue(sentence, question, answer, answer_start_idx)

    example = SquadAugmentedExample(
        question_text=question,
        context_text=sentence,
        answer_text=answer,
        question_type=question_type,
        answer_pos_tag=answer_pos_tag,
        answer_ner_tag=answer_ner_tag,
        answer_length=answer_length,
        clue_info=clue_info,
    )

    return example


def test_chunks():
    # Test cases
    test_sentences = [
        "Beyoncé Giselle Knowles-Carter is an American singer, songwriter, and actress.",
        "Barack Obama was the 44th President of the United States.",
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France."
    ]

    for sentence in test_sentences:
        print(chunks(sentence))


def test_extract_clue():
    # Test case 1
    test_sentence_1 = "Albert Einstein was a theoretical physicist who developed the theory of relativity."
    test_question_1 = "Who developed the theory of relativity?"
    test_answer_1 = "Albert Einstein"
    test_answer_start_1 = 0

    result_1 = extract_clue(test_sentence_1, test_question_1,
                            test_answer_1, test_answer_start_1)
    print(result_1)

    # Test case 2
    test_sentence_2 = "The capital of France is Paris, which is known for its culture, art, and landmarks."
    test_question_2 = "I apologize for the incomplete response. Here's the complete test case 2:"

    # Test case 2
    test_sentence_2 = "The capital of France is Paris, which is known for its culture, art, and landmarks."
    test_question_2 = "What is the capital of France?"
    test_answer_2 = "Paris"
    test_answer_start_2 = 21

    result_2 = extract_clue(test_sentence_2, test_question_2,
                            test_answer_2, test_answer_start_2)
    print(result_2)


def test_extract_question_type():
    # Test informational question types
    assert extract_question_type_and_id("Who are you?") == ("who", 1)
    assert extract_question_type_and_id("Where do you live?") == ("where", 3)
    assert extract_question_type_and_id("When were you born") == ("when", 4)
    assert extract_question_type_and_id("Why are you here?") == ("why", 5)
    assert extract_question_type_and_id(
        "Which one do you prefer?") == ("which", 6)
    assert extract_question_type_and_id("What is your name?") == ("what", 0)
    assert extract_question_type_and_id("How are you?") == ("how", 2)

    # Test boolean question types
    assert extract_question_type_and_id("Is it ok?") == ("boolean", 7)
    assert extract_question_type_and_id("Can you sleep?") == ("boolean", 7)
    assert extract_question_type_and_id("Should I leave?") == ("boolean", 7)
    assert extract_question_type_and_id(
        "Would you ..., please?") == ("boolean", 7)

    # Test "Other" question type
    assert extract_question_type_and_id(
        "Name the capital of France.") == ("other", 8)


def test_extract_clue_and_question_info():
    # Test case 1: A simple question and answer
    sentence1 = "The quick brown fox jumps over the lazy dog."
    question1 = "What color is the fox?"
    answer1 = "brown"
    answer_start1 = 10
    result1 = extract_clue_and_question_info(
        sentence1, question1, answer1, answer_start1)
    print(result1)
    assert result1.question_type == "what"
    assert result1.answer_pos_tag == "JJ"
    assert result1.answer_ner_tag == "UNK"
    assert result1.clue_info.clue_pos_tag == "ADJ"
    assert result1.clue_info.clue_ner_tag == "UNK"
    print("succeeded")


if __name__ == "__main__":
    # test_chunks()
    # test_extract_clue()
    # test_extract_question_type()
    test_extract_clue_and_question_info()
