import torch
from botiverse.Theorizer.model.finetuned_model import MyGPT2LMHeadModel
from botiverse.Theorizer.model.dataloader import SPECIAL_TOKENS_DICT
from botiverse.Theorizer.squad.sample_data import select_with_default_sampel_probs
from transformers import GPT2Tokenizer
import os

current_file_dir = os.path.dirname(os.path.abspath(__file__))

def __prepare(context):
    sampled_infos = select_with_default_sampel_probs(context)
    instances = []
    for info in sampled_infos["selected_infos"]:
        for style in info["styles"]:
            for clue in info["clues"]:
                instances.append(
                    {
                        "paragraph": sampled_infos["context"],
                        "clue": clue.clue_text,
                        "answer": info["answer"]["answer_text"],
                        "style": style,
                    }
                )
    return instances


def generate(context, max_length=50):
    # Load the fine-tuned model and tokenizer
    
    model_path_or_name = os.path.join(current_file_dir,"model/pretrained-model")
    
    model = MyGPT2LMHeadModel.from_pretrained(
        model_path_or_name, ignore_mismatched_sizes=True
    )
    tokenizer = GPT2Tokenizer.from_pretrained(
        model_path_or_name, **SPECIAL_TOKENS_DICT
    )

    instances = __prepare(context)
    # print(instances)
    qa_dict = {}
    qa_dict["context"] = context
    qa_dict["qa"] = set()
    for inst in instances:
        paragraph = inst["paragraph"]
        clue = inst["clue"]
        answer = inst["answer"]
        style = inst["style"]
        input_sequence = (
            "<sos> "
            + paragraph
            + " <clue> "
            + clue
            + " <answer> "
            + answer
            + " <style> "
            + style
            + " <question> "
            + style
        )

        # Tokenize the input sequence
        input_ids = tokenizer.encode(input_sequence, return_tensors="pt")
        with torch.no_grad():
            # Generate the question
            generated = model.generate(
                input_ids=input_ids,
                max_length=max_length,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                num_beams=5,
                no_repeat_ngram_size=2,
                early_stopping=True,
                temperature=0.7,
            )
        question_start_index = (
            input_ids[0].tolist().index(
                tokenizer.convert_tokens_to_ids("<question>"))
        )

        # Slice the generated tensor to exclude the input sequence
        generated_question = generated[0, question_start_index + 1:]

        # Decode the generated question
        question = tokenizer.decode(
            generated_question, skip_special_tokens=True)

        qa_dict["qa"].add((question, answer))
    qa_dict["qa"] = list(qa_dict["qa"])
    return qa_dict


if __name__ == "__main__":
    context = "Bob is eating a delicious cake in Vancouver."
    qa_dict = generate(context)
    import json
    print(json.dumps(qa_dict,indent=4))
