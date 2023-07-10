"""
This Module has the run functions for TRIPPY that train and evaluate the model.
"""

import torch
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup


from botiverse.models.TRIPPY.data import prepare_data, Dataset
from botiverse.models.TRIPPY.train import train
from botiverse.models.TRIPPY.evaluate import eval


def run(model, domains, slot_list, label_maps, train_json, dev_json, test_json, device, non_referable_slots, non_referable_pairs, model_path, TRIPPY_config):
    """
    Train and evaluate the TRIPPY model.

    :param model: The TRIPPY model.
    :type model: TRIPPY
    :param domains: The domains to consider in the dataset.
    :type domains: list
    :param slot_list: The list of slots.
    :type slot_list: list
    :param label_maps: The mapping of slot values to their variants.
    :type label_maps: dict
    :param train_json: The path to the training dataset in JSON format.
    :type train_json: str
    :param dev_json: The path to the development dataset in JSON format.
    :type dev_json: str
    :param test_json: The path to the testing dataset in JSON format.
    :type test_json: str
    :param device: The device to train and evaluate the model on.
    :type device: torch.device
    :param non_referable_slots: The slots that are not referable.
    :type non_referable_slots: list
    :param non_referable_pairs: The pairs of slots that are not referable.
    :type non_referable_pairs: list
    :param model_path: The path to save the best model.
    :type model_path: str
    :param TRIPPY_config: The configuration for TRIPPY.
    :type TRIPPY_config: TRIPPYConfig
    """

    n_slots = len(slot_list)

    # train
    print('Preprocessing train set...')
    train_raw_data, train_data = prepare_data(train_json, slot_list, label_maps, TRIPPY_config.tokenizer, TRIPPY_config.max_len, domains, non_referable_slots, non_referable_pairs, TRIPPY_config.multiwoz)
    train_dataset = Dataset(train_data, n_slots, TRIPPY_config.oper2id, slot_list)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    sampler=train_sampler,
                                                    batch_size=TRIPPY_config.train_batch_size)

    # dev
    print('Preprocessing dev set...')
    dev_raw_data, dev_data = prepare_data(dev_json, slot_list, label_maps, TRIPPY_config.tokenizer, TRIPPY_config.max_len, domains, non_referable_slots, non_referable_pairs, TRIPPY_config.multiwoz)
    dev_dataset = Dataset(dev_data, n_slots, TRIPPY_config.oper2id, slot_list)
    dev_data_loader = torch.utils.data.DataLoader(dev_dataset,
                                                  batch_size=TRIPPY_config.dev_batch_size)

    # test
    print('Preprocessing test set...')
    test_raw_data, test_data = prepare_data(test_json, slot_list, label_maps, TRIPPY_config.tokenizer, TRIPPY_config.max_len, domains, non_referable_slots, non_referable_pairs, TRIPPY_config.multiwoz)
    test_dataset = Dataset(test_data, n_slots, TRIPPY_config.oper2id, slot_list)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=TRIPPY_config.test_batch_size)

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": TRIPPY_config.weight_decay,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    # num_train_steps = int(len(train_dataset) / TRAIN_BATCH_SIZE * EPOCHS)
    num_train_steps = len(train_data_loader) * TRIPPY_config.epochs
    num_warmup_steps = int(num_train_steps * TRIPPY_config.warmup_proportion)

    optimizer = AdamW(optimizer_parameters, lr=TRIPPY_config.lr, eps=TRIPPY_config.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps
    )

    best_joint = -1
    for epoch in range(TRIPPY_config.epochs):
        print(f'\nEpoch: {epoch} ---------------------------------------------------------------')
        print('Training the model...')
        train(train_data_loader, model, optimizer, device, scheduler, n_slots, TRIPPY_config.ignore_idx, TRIPPY_config.oper2id)
        print('Evaluating the model on dev set...')
        # jaccard_score, macro_f1_score, all_f1_score = eval_f1_jac(dev_data_loader, model, device, n_slots)
        # joint_goal_acc, states, sentences, indices = eval_joint(dev_raw_data, dev_data, model, device, n_slots, slot_list, label_maps)
        joint_goal_acc, per_slot_acc, macro_f1_score, all_f1_score = eval(dev_raw_data, dev_data, model, device, n_slots, slot_list, label_maps, TRIPPY_config.oper2id, TRIPPY_config.multiwoz)
        # print(f'Joint Goal Acc: {joint_goal_acc}, Jaccard Score: {jaccard_score}, Macro F1 Score: {macro_f1_score}')
        print(f'Joint Goal Acc: {joint_goal_acc}')
        print(f'Per Slot Acc: {per_slot_acc}')
        print(f'Macro F1 Score: {macro_f1_score}')
        print(f'All f1 score = {all_f1_score}')
        if joint_goal_acc > best_joint:
            torch.save(model.state_dict(), model_path)
            best_joint = joint_goal_acc

    print('Loading best model on dev set...')
    model.load_state_dict(torch.load(model_path))
    print('Evaluating the model on test set...')
    joint_goal_acc, per_slot_acc, macro_f1_score, all_f1_score = eval(test_raw_data, test_data, model, device, n_slots, slot_list, label_maps, TRIPPY_config.oper2id, TRIPPY_config.multiwoz)
    print(f'Joint Goal Acc: {joint_goal_acc}')
    print(f'Per Slot Acc: {per_slot_acc}')
    print(f'Macro F1 Score: {macro_f1_score}')
    print(f'All f1 score = {all_f1_score}')
    
    
    

# from botiverse.bots import Theorizer
# from botiverse.models import NeuralNet
# from botiverse.preprocessors import BertEmbedder