import torch
from botiverse.TODS.DNN_DST.config import *
from botiverse.TODS.DNN_DST.data import prepare_data, Dataset
from botiverse.TODS.DNN_DST.train import train
from botiverse.TODS.DNN_DST.evaluate import eval_f1_jac, eval_joint
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

def run(model, domains, slot_list, label_maps, train_json, dev_json, test_json, device):

    n_slots = len(slot_list)

    # train
    print('Preprocessing train set...')
    train_raw_data, train_data = prepare_data(train_json, slot_list, label_maps, TOKENIZER, MAX_LEN, domains)
    train_dataset = Dataset(train_data, n_slots, OPER2ID)
    train_sampler = torch.utils.data.RandomSampler(train_dataset)
    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    sampler=train_sampler,
                                                    batch_size=TRAIN_BATCH_SIZE)

    # dev
    print('Preprocessing dev set...')
    dev_raw_data, dev_data = prepare_data(dev_json, slot_list, label_maps, TOKENIZER, MAX_LEN, domains)
    dev_dataset = Dataset(dev_data, n_slots, OPER2ID)
    dev_data_loader = torch.utils.data.DataLoader(dev_dataset,
                                                  batch_size=DEV_BATCH_SIZE)
    
    # test
    print('Preprocessing test set...')
    test_raw_data, test_data = prepare_data(test_json, slot_list, label_maps, TOKENIZER, MAX_LEN, domains)
    test_dataset = Dataset(test_data, n_slots, OPER2ID)
    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=TEST_BATCH_SIZE)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_dataset) / TRAIN_BATCH_SIZE * EPOCHS)
    optimizer = AdamW(optimizer_parameters, lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_joint = -1
    for epoch in range(EPOCHS):
        print(f'\nEpoch: {epoch} ---------------------------------------------------------------')
        print('Training the model...')
        train(train_data_loader, model, optimizer, device, scheduler, n_slots, IGNORE_IDX)
        print('Evaluating the model on dev set...')
        jaccard_score, macro_f1_score, all_f1_score = eval_f1_jac(dev_data_loader, model, device, n_slots)
        joint_goal_acc, states, sentences, indices = eval_joint(dev_raw_data, dev_data, model, device, n_slots, slot_list, label_maps)
        print(f'Joint Goal Acc: {joint_goal_acc}, Jaccard Score: {jaccard_score}, Macro F1 Score: {macro_f1_score}')
        print(f'All f1 score = {all_f1_score}')
        if joint_goal_acc > best_joint:
            torch.save(model.state_dict(), MODEL_PATH)
            best_joint = joint_goal_acc

    print('Loading best model on dev set...')
    model.load_state_dict(torch.load(MODEL_PATH))
    print('Evaluating the model on test set...')
    jaccard_score, macro_f1_score, all_f1_score = eval_f1_jac(test_data_loader, model, device, n_slots)
    joint_goal_acc, states, sentences, indices = eval_joint(test_raw_data, dev_data, model, device, n_slots, slot_list, label_maps)
    print(f'Joint Goal Acc: {joint_goal_acc}, Jaccard Score: {jaccard_score}, Macro F1 Score: {macro_f1_score}')
    print(f'All f1 score = {all_f1_score}')