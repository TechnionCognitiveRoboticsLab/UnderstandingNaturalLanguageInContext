import os
import torch
import prediction
import preprocess
import transformers
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset
from training import calc_pddl_correct
from Utils import utils_paths, utils_functions
from transformers import GPT2Tokenizer, TrainingArguments, Trainer, GPT2LMHeadModel

transformers.logging.set_verbosity_error()

os.environ["WANDB_DISABLED"] = "True"


#  ############################  Dataset ############################  #

class AlfredDataset(Dataset):
    def __init__(self, src_list, trg_list, tokenizer, max_length, task_type, benchmark_data=False):
        self.input_ids = []
        self.attn_masks = []
        self.labels = []

        tqdm_obj = tqdm(zip(src_list, trg_list), total=len(src_list))
        for src, trg in tqdm_obj:
            if benchmark_data:
                prep_txt = utils_functions.create_sentence_for_gpt(src, '[sep]', trg)
            else:
                sep = '{}:'.format(task_type.capitalize())
                prep_txt = utils_functions.create_sentence_for_gpt(src, sep, trg)
            encodings_dict = tokenizer(prep_txt,
                                       truncation=True,
                                       max_length=max_length,
                                       padding="max_length")

            self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
            self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
            self.labels.append(trg)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx], self.labels[idx]


#  ############################  Pre-Process ############################  #


def preprocess_paper_data():
    datasets = {}
    modes = ['train', 'dev', 'test']
    for m in modes:
        with open('{}Benchmark/alfred.{}.gpt2.txt'.format(utils_paths.data_path, m)) as f:
            lines = f.readlines()
        df = create_df_from_lines(lines)
        df.to_csv('{}Benchmark/{}_data_paper.csv'.format(utils_paths.data_path, m))
        datasets[m] = df
    return datasets


def load_alfred_datasets(datasets, tokenizer, task_type, input_col, paper_data):
    alfred_datasets_dict = {}
    eval_datasets_dict = {}
    max_length = 0

    col = 'actions' if paper_data else 'target_text_{}'.format(task_type)

    for data_mode, df in datasets.items():
        if data_mode == 'train':
            max_length = find_lengths(tokenizer,
                                      df[input_col].tolist(),
                                      df[col].tolist(),
                                      task_type,
                                      benchmark_data=paper_data)

        eval_datasets_dict[data_mode] = {'input_text': df[input_col].tolist(),
                                         'target_text_{}'.format(task_type): df[col].tolist(),
                                         'relations_meta_wo_ids': df['relations_meta_wo_ids'],
                                         'pddl problem': df['pddl problem'],
                                         'task': df['task']}

        alfred_datasets_dict[data_mode] = AlfredDataset(df[input_col],
                                                        df[col],
                                                        tokenizer,
                                                        max_length=max_length,
                                                        task_type=task_type,
                                                        benchmark_data=paper_data)

    return alfred_datasets_dict, eval_datasets_dict, max_length


def find_lengths(gpt_tokenizer, src_list_, trg_list_, task_type, benchmark_data=False):
    combined = []

    for txt, action_seq in tqdm(zip(src_list_, trg_list_)):
        if benchmark_data:
            prep_txt = utils_functions.create_sentence_for_gpt(txt, '[sep]', action_seq)
        else:
            sep = '{}:'.format(task_type.capitalize())
            prep_txt = utils_functions.create_sentence_for_gpt(txt, sep, action_seq)
        combined.append(prep_txt)

    max_length = 0

    for i in tqdm(range(len(combined))):
        src_ = combined[i]
        src_len = len(gpt_tokenizer(src_).data['input_ids'])
        max_length = max(max_length, src_len)

    print('Max src length for GPT-2: {} \n'.format(max_length))
    return max_length


def get_src_and_trg_from_line(line):
    line = line.replace(' [EOS]\n', '')
    line = line.lower()
    split_line = line.split(" [sep] ")
    src, trg = split_line
    if 'noop' in trg:
        trg = trg.replace(' </s> noop </s>', '')
    else:
        if trg[-5:] == ' </s>':
            trg = trg[:-5]

    trg = trg.replace('</s>', ',')
    return src, trg


def create_df_from_lines(lines):
    data = {'task': [], 'actions': []}
    for line in lines:
        src, trg = get_src_and_trg_from_line(line)
        data['task'].append(src)
        data['actions'].append(trg)

    return pd.DataFrame(data)


#  ############################  Model Functions ############################  #

def get_model_and_tokenizer(pt='', model_name="gpt2-medium"):
    torch.manual_seed(42)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name,
                                              bos_token='<|startoftext|>',
                                              eos_token='<|endoftext|>',
                                              pad_token='<|pad|>')

    if pt:
        gpt_model = GPT2LMHeadModel.from_pretrained(pt).cuda()
        return gpt_model, tokenizer

    gpt_model = GPT2LMHeadModel.from_pretrained(model_name).cuda()
    gpt_model.resize_token_embeddings(len(tokenizer))
    return gpt_model, tokenizer


def validate_bs(eval_datasets, tokenizer, model, mode, beam, task_type, bs=1, benchmark_data=False):
    data = []

    model.eval()

    orig_actions = []
    pred_actions = []
    pred_actions_beam = []
    combined_src = []

    src_list = eval_datasets[mode]['input_text']
    trg_list = eval_datasets[mode]['target_text_{}'.format(task_type)]

    tqdm_objects = tqdm(zip(src_list, trg_list), total=len(src_list))

    for src, action_seq in tqdm_objects:
        if benchmark_data:
            prompt = utils_functions.create_sentence_for_gpt(src, '[sep]', '')
        else:
            sep = '{}:'.format(task_type.capitalize())
            prompt = utils_functions.create_sentence_for_gpt(src, sep, '')

        combined_src.append(prompt)
        data.append(prompt)
        orig_actions.append(action_seq)

    max_src_length = 0

    for i in range(len(combined_src)):
        src_ = combined_src[i]
        src_len = len(tokenizer(src_).data['input_ids'])
        max_src_length = max(max_src_length, src_len)

    max_trg_length = 0

    for i in range(len(trg_list)):
        trg_ = trg_list[i]
        trg_len = len(tokenizer(trg_).data['input_ids'])
        max_trg_length = max(max_trg_length, trg_len)

    for i in tqdm(range(0, len(data), bs)):
        batch = data[i: i + bs]
        preds_only_first, preds = validate_batch(model,
                                                 tokenizer,
                                                 batch,
                                                 beam,
                                                 task_type,
                                                 max_trg_length + max_src_length,
                                                 benchmark_data)

        pred_actions.extend(preds_only_first)
        pred_actions_beam.extend(preds)

    return orig_actions, pred_actions, pred_actions_beam


def validate_batch(model, tokenizer, prompt_text, beam, task_type, num_tokens_to_produce=50, benchmark_data=False):
    model.eval()
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=True)
    ids = inputs['input_ids'].cuda()

    output_sequences = model.generate(
        input_ids=ids,
        attention_mask=inputs['attention_mask'].cuda(),
        do_sample=False,
        max_length=num_tokens_to_produce,
        num_beams=beam,
        num_return_sequences=beam
    )
    preds = [utils_functions.get_prediction_from_full_sentence
             (tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True),
              benchmark_data,
              task_type)

             for g in output_sequences]

    preds_only_first = preds

    if beam > 1:
        preds_only_first = [preds[beam * i] for i in range(ids.shape[0])]
        preds = ['|'.join(preds[beam * i:beam * i + beam])
                 for i in range(ids.shape[0])]

    return preds_only_first, preds


def run_full_process(args):
    num_epochs = args.num_epochs
    model_type = args.model_type
    task_type = args.task_type

    train = args.train
    benchmark_data = args.benchmark_data

    batch_size = args.batch_size
    steps = args.steps

    input_type = args.input_type

    test_val_size = args.train_test_split
    train_size = round(1 - test_val_size, 2)

    precision = {}
    recall = {}
    precision_per_value = {}
    recall_per_value = {}

    path = '{}/{}/{}/{}'.format(utils_paths.trained_model_path, model_type, input_type, train_size) if not train else ''
    print('\n{} PATH : {} {}'.format('#'*30, path, '#'*30))
    if not train and not path:
        print('Missing trained model path!'.upper())

    #  ############################  Load Model and Data ############################  #

    model, gpt_tokenizer = get_model_and_tokenizer(pt=path, model_name=model_type)
    gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

    modes = ['train', 'val', 'test', 'test_unseen']
    datasets = {}

    if benchmark_data:
        datasets = preprocess_paper_data()
        input_col = 'task'
    else:
        utils_functions.start_print('Pre-Processing data')
        input_col = 'input_text'

        preprocess.validate_and_split(args, train_size)

        for mode in modes:
            datasets[mode] = preprocess.read_data(mode, args, train_size)

        if args.remove_duplicates:
            datasets = utils_functions.get_clean_datasets(datasets, input_type, model_type, modes, train_size)

        print('Finished Pre-Processing data\n\n')

    alfred_datasets, eval_datasets, max_len = load_alfred_datasets(datasets, gpt_tokenizer, task_type,
                                                                   input_col, benchmark_data)

    val_name = 'val_seen' if 'val_seen' in modes else 'val'
    if train:
        model.train()
        dir_path = '{}/{}/{}/{}/{}'.format(utils_paths.trained_model_path,
                                           model_type,
                                           task_type.capitalize(),
                                           input_type,
                                           train_size)
        utils_paths.check_and_create_dir(dir_path)

        training_args = TrainingArguments(output_dir=dir_path,
                                          num_train_epochs=num_epochs,
                                          logging_steps=steps,
                                          load_best_model_at_end=True,
                                          save_steps=steps,
                                          save_total_limit=2,
                                          per_device_train_batch_size=batch_size,
                                          per_device_eval_batch_size=batch_size,
                                          warmup_steps=100,
                                          weight_decay=0.01,
                                          logging_dir=None,
                                          disable_tqdm=False,
                                          evaluation_strategy=transformers.trainer_utils.IntervalStrategy.STEPS,
                                          eval_steps=steps)

        Trainer(model=model,
                args=training_args,
                train_dataset=alfred_datasets['train'],
                eval_dataset=alfred_datasets[val_name],
                data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                            'attention_mask': torch.stack([f[1] for f in data]),
                                            'labels': torch.stack([f[0] for f in data])}).train()

        print('\n\n\nSaving Model in : {}\n\n\n'.format(dir_path))
        model.save_pretrained(dir_path)
        gpt_tokenizer.save_pretrained(dir_path)

    # modes_for_eval = ['val', 'test', 'test_unseen']
    modes_for_eval = ['test_unseen']
    for mode in modes_for_eval:
        orig, pred, pred_beam = validate_bs(eval_datasets,
                                            gpt_tokenizer,
                                            model,
                                            mode,
                                            args.beam,
                                            task_type,
                                            bs=batch_size*2,
                                            benchmark_data=benchmark_data)

        eval_datasets[mode]['orig_{}'.format(task_type)] = orig
        eval_datasets[mode]['pred_{}_beam'.format(task_type)] = pred_beam
        beam_data_dir = '{}{}/{}/{}/{}/{}/'.format(utils_paths.results_path_csv,
                                                   model_type,
                                                   task_type.capitalize(),
                                                   input_type,
                                                   train_size,
                                                   'Beam_Data')

        utils_paths.check_and_create_dir(beam_data_dir)

        pd.DataFrame(eval_datasets[mode]).to_csv('{}{}_data_with_beam_predictions.csv'. format(beam_data_dir, mode))

        if task_type.lower() == 'actions':
            pddl_correct = 0
            if not args.server:
                pddl_correct = calc_pddl_correct(orig, pred_beam, pd.DataFrame(eval_datasets[mode]),
                                                 mode, model_type, input_type, 'target_text_goal')

            precision_recall_dicts = prediction.accuracy_cal_actions(orig, pred, mode, input_type, train_size,
                                                                     model_type, pddl_correct)
        else:
            precision_recall_dicts = prediction.accuracy_cal_goals(orig, pred, mode, input_type, train_size, model_type)

        per_name = 'predicate_type' if task_type == 'goal' else 'action'

        precision[mode] = precision_recall_dicts['precision']
        recall[mode] = precision_recall_dicts['recall']
        precision_per_value[mode] = precision_recall_dicts['precision_per_{}'.format(per_name)]
        recall_per_value[mode] = precision_recall_dicts['recall_per_{}'.format(per_name)]

        prediction_dir = '{}/{}/{}/{}/{}/Predictions/'.format(utils_paths.results_path_csv,
                                                              model_type,
                                                              task_type.capitalize(),
                                                              input_type,
                                                              train_size)

        utils_paths.check_and_create_dir(prediction_dir)

        prediction_df = pd.DataFrame({'original': orig, 'predicted': pred})
        prediction_df.to_csv('{}{}_predictions.csv'.format(prediction_dir, mode))

    dicts = {'Precision_General': precision,
             'Recall_General': recall,
             'Precision_per_value': precision_per_value,
             'Recall_per_value': recall_per_value}

    print('Dicts: {}'.format(dicts))

    pdf_string = '{}_Model-{}_Epochs-{}_Train_Split-{}'.format(input_type, model_type, num_epochs, train_size)
    utils_functions.create_pdf(pdf_string, dicts, task_type)
