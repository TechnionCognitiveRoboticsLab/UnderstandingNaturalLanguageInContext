import time
import training
import torch.multiprocessing
import preprocess as preprocess
from transformers import T5Tokenizer
from Utils import utils_model, utils_functions, utils_paths

torch.multiprocessing.set_sharing_strategy('file_system')


def run_full_process(args):
    print("\nStart time: {}\n".format(time.strftime("%H:%M:%S", time.localtime())))

    # ARGS #
    input_type = args.input_type
    model_type = args.model_type
    task_type = args.task_type
    num_epochs = args.num_epochs
    beam = args.beam
    test_val_size = args.train_test_split
    train_size = round(1 - test_val_size, 2)
    tokenizer = T5Tokenizer.from_pretrained(model_type)

    datasets = {}
    results_dict = {}

    # Pre-Processing Datasets
    modes = ['train', 'val', 'test', 'test_unseen']

    if args.preprocess_data:
        utils_functions.start_print('Pre-Processing data')

        preprocess.validate_and_split(args, train_size)

        for mode in modes:
            datasets[mode] = preprocess.read_data(mode, args, train_size)

        if args.remove_duplicates:
            datasets = utils_functions.get_clean_datasets(datasets, input_type, model_type, modes, train_size)
        print('Finished Pre-Processing data\n\n')

    else:
        utils_functions.start_print('Reading dataset')
        datasets = preprocess.load_clean_datasets(model_type, input_type, modes, train_size)
        print('Finished Reading dataset\n\n')

    if task_type == 'both':
        utils_functions.start_print('Combining goal and actions datasets')
        data_dir = '{}/{}/{}/{}/'.format(utils_paths.csv_path, model_type, input_type, train_size)
        datasets = {mode: preprocess.create_big_df(data_dir, mode) for mode in datasets}
        print('Finished Reading Goal-Beam dataset\n\n')

    utils_functions.start_print('Finding Lengths')
    src_len, trg_len = preprocess.find_lengths(datasets['train'], task_type, tokenizer)
    new_batch_size = utils_functions.edit_batch_size(src_len, model_type)

    # Update Model Params #
    model_params = utils_model.model_params[task_type]
    model_params["MODEL"] = model_type
    model_params["INPUT"] = input_type
    model_params["TRAIN_BATCH_SIZE"] = new_batch_size
    model_params["VALID_BATCH_SIZE"] = new_batch_size*2
    model_params["TRAIN_EPOCHS"] = num_epochs
    model_params["max_src_len"] = src_len
    model_params["max_trg_len"] = trg_len
    utils_model.model_params[task_type] = model_params

    # Training
    if args.train:
        utils_functions.start_print('Training Model')
        training.train_model(datasets, model_params, task_type, train_size, input_type, args.load_model)
        print('Finished Training Model\n\n')

    # Evaluation
    if args.evaluate:
        utils_functions.start_print('Eval All')
        data_dict = training.get_data_dict(task_type, datasets)
        both = False
        if len(data_dict) > 1:
            both = True
        for tt, data in data_dict.items():
            utils_functions.start_print('Eval All - Task Type {}'.format(tt))
            results_dict[tt] = training.eval_all(data, model_params, tt, input_type,
                                                 model_type, beam, args.server, both, train_size)
        print('Finished Eval All\n\n')

    # Create PDF outputs
    if args.create_pdf_output:
        model_type = '{}_Both'.format(model_type) if task_type == 'both' else model_type
        pdf_str = '{}_Model-{}_Epochs-{}_Train_Split-{}'.format(input_type, model_type, num_epochs, train_size)
        for tt, dicts in results_dict.items():
            utils_functions.create_pdf(pdf_str, dicts, tt)

    print("\nFinish time: {}\n".format(time.strftime("%H:%M:%S", time.localtime())))
