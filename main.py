import gpt2_trainer
import t5_trainer
import argparse

#  action='store_true' -> Default=False
#  action='store_false' -> Default=True

parser = argparse.ArgumentParser()

parser.add_argument('-model', '--model_type', default='gpt2-medium', type=str, required=True, help='Model type')
parser.add_argument('-tt', '--task_type', default='actions', type=str, required=True, help='Task type: Actions or Goal')

parser.add_argument('--benchmark_data', action='store_true', help='Use Benchmark data')
parser.add_argument('--use_meta', action='store_false', help='Use metadata')

parser.add_argument('--server', action='store_false', help='running on the server')

#  Process Arguments
parser.add_argument('--load_model', action='store_true', help='Do train on my pretrained model')

parser.add_argument('--preprocess_data', action='store_false', help='Preprocess Data')
parser.add_argument('--validate_data', action='store_true', help='Validate data with PDDL in Pre-processing')
parser.add_argument('--re_split', action='store_true', help='Re-split the data into train, val and test')
parser.add_argument('--remove_duplicates', action='store_false', help='Remove duplicates between train, val and test')

parser.add_argument('--train', action='store_false', help='Do train')
parser.add_argument('--evaluate', action='store_false', help='Do eval')
parser.add_argument('--create_pdf_output', action='store_false', help='Create PDF output')


#  General Hyper-Parameters
parser.add_argument('-epochs', '--num_epochs', type=int, default=25, help='Number of Epochs')
parser.add_argument('--train_test_split', type=float, default=0.2, help='Size of val+test')

parser.add_argument('-input', '--input_type', type=str, default='task', required=True, help='Input type')

#  GPT Hyper-Parameters
parser.add_argument('-bs', '--batch_size', type=int, default=16, help='Batch size')
parser.add_argument('--eval_bs', type=int, default=256, help='Batch size')
parser.add_argument('--steps', type=int, default=3000, help='Logging, Evaluating and Saving steps')

# Output Params
parser.add_argument('--beam', type=int, default=3, help='Multiple sentences in output')


args = parser.parse_args()

print('\n\n{}{}{}'.format('#'*30, ' ARGS ', '#'*30))
for arg in vars(args):
    print(arg, getattr(args, arg))
print('{}{}'.format('#'*66, '\n\n'))

if 'gpt' in args.model_type:
    gpt2_trainer.run_full_process(args)
elif 't5' in args.model_type:
    t5_trainer.run_full_process(args)
