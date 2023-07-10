from typing import List
from .dataloader import prepare_and_pad_squad_data_for_gpt2, read_cached_processed_data, SquadGPT2Example, SPECIAL_TOKENS
from .finetuned_model import MyGPT2LMHeadModel
import torch
from tqdm import tqdm
from transformers import (
    GPT2Config,
    GPT2TokenizerFast,
    get_linear_schedule_with_warmup,
)
from torch.utils.data import DataLoader, TensorDataset
# Training parameters and configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpu = torch.cuda.device_count()
n_epochs = 10
lr = 5e-4
gradient_accumulation_steps = 4
max_norm = 1.0
train_batch_size = 4
valid_batch_size = 4
padding_length = 512
config = GPT2Config.from_pretrained("gpt2")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
tokenizer.add_tokens(SPECIAL_TOKENS)

model = MyGPT2LMHeadModel.from_pretrained("gpt2", config=config)
if not (next(model.parameters()).is_cuda):
    model.to(device)
model.resize_token_embeddings(len(tokenizer))
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def train(model, tokenizer, train_loader, val_loader, n_epochs, output_dir):
    """
    Train a GPT-2 model on the SQuAD dataset.
    
    Args:
        model: The GPT-2 model to be trained.
        tokenizer: The tokenizer used for tokenizing the data.
        train_loader: The DataLoader containing the training data.
        val_loader: The DataLoader containing the validation data.
        n_epochs: The number of epochs to train the model.
        output_dir: The directory where the model and tokenizer will be saved.
    """
    total_steps = len(train_loader) * n_epochs
    warmup_steps = int(0.1 * total_steps)  # 10% of total steps
    num_training_steps = len(train_loader) * \
        n_epochs // gradient_accumulation_steps

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    global_step = 0
    for epoch in tqdm(range(n_epochs)):
        model.train()
        epoch_loss = 0
        for step, batch in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            if n_gpu == 1:
                batch = tuple(t.to(device) for t in batch)
            cur_input_ids, cur_lm_labels, cur_token_type_ids = batch
            model_outputs = model(
                input_ids=cur_input_ids,
                labels=cur_lm_labels,
                token_type_ids=cur_token_type_ids,
            )
            loss = model_outputs[0]
            loss = loss / gradient_accumulation_steps
            epoch_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            optimizer.step()
            scheduler.step()
            global_step += 1

        epoch_loss /= len(train_loader)

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_loss = 0
            for batch in tqdm(val_loader):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch)
                cur_input_ids, cur_lm_labels, cur_token_type_ids = batch
                model_outputs = model(
                    input_ids=cur_input_ids,
                    labels=cur_lm_labels,
                    token_type_ids=cur_token_type_ids,
                )

                lm_loss = model_outputs[0]
                loss = lm_loss / gradient_accumulation_steps
                val_loss += loss.item()
            val_loss /= len(val_loader)

        # Save final model and tokenizer
        epoch_output_dir = f"{output_dir}_gelu_{epoch}"
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(epoch_output_dir)
        tokenizer.save_pretrained(epoch_output_dir)


if __name__ == "__main__":
    output_dir = "/kaggle/working/model"
    train_examples = read_cached_processed_data(
        "squad/dataset/train.processed.pkl")
    ############################### Training Data ###############################
    train_features: List[SquadGPT2Example] = prepare_and_pad_squad_data_for_gpt2(
        processed_examples=train_examples,
        tokenizer=tokenizer,
        max_len=padding_length
    )
    all_input_ids = torch.tensor(
        [f.input_ids for f in train_features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in train_features], dtype=torch.long)
    all_lm_labels = torch.tensor(
        [f.lm_labels for f in train_features], dtype=torch.long)
    train_data = TensorDataset(
        all_input_ids, all_token_type_ids, all_lm_labels)
    train_dataloader = DataLoader(
        train_data, batch_size=train_batch_size)
    
    ############################### Validation Data ###############################
    valid_examples = read_cached_processed_data(
        "squad/dataset/dev.processed.pkl")
    valid_features: List[SquadGPT2Example] = prepare_and_pad_squad_data_for_gpt2(
        processed_examples=valid_examples,
        tokenizer=tokenizer,
        max_len=padding_length
    )
    all_input_ids = torch.tensor(
        [f.input_ids for f in valid_features], dtype=torch.long)
    all_token_type_ids = torch.tensor(
        [f.token_type_ids for f in valid_features], dtype=torch.long)
    all_lm_labels = torch.tensor(
        [f.lm_labels for f in valid_features], dtype=torch.long)
    valid_data = TensorDataset(
        all_input_ids, all_token_type_ids, all_lm_labels)
    valid_dataloader = DataLoader(
        valid_data, batch_size=valid_batch_size)

    train(model, tokenizer, train_dataloader, valid_dataloader, n_epochs, output_dir)
