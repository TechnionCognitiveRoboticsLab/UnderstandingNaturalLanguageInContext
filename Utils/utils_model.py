import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

actions_model_params = {
    "MODEL": "t5-base",
    "TRAIN_BATCH_SIZE": 32,
    "VALID_BATCH_SIZE": 32,
    "TRAIN_EPOCHS": 20,
    "TRAIN_EPOCHS_GPT": 5,
    "LEARNING_RATE": 1e-4,
    "max_src_len": 512,
    "max_trg_len": 76,
    "num_workers": 0
}

goal_model_params = {
    "MODEL": "t5-small",
    "TRAIN_BATCH_SIZE": 32,
    "VALID_BATCH_SIZE": 32,
    "TRAIN_EPOCHS": 1,
    "VAL_EPOCHS": 1,
    "LEARNING_RATE": 1e-4,
    "max_src_len": 60,
    "max_trg_len": 50,
    "num_workers": 0
}

both_model_params = {
    "MODEL": "t5-base",
    "TRAIN_BATCH_SIZE": 32,
    "VALID_BATCH_SIZE": 32,
    "TRAIN_EPOCHS": 20,
    "TRAIN_EPOCHS_GPT": 5,
    "LEARNING_RATE": 1e-4,
    "max_src_len": 512,
    "max_trg_len": 76,
    "num_workers": 0
}

model_params = {'goal': goal_model_params, 'actions': actions_model_params, 'both': both_model_params}
