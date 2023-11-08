import pathlib
from pathlib import Path


def check_and_create_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


abs_path = str(pathlib.Path(__file__).parent.resolve()).split('/')
add = '/data/' if abs_path[1] == 'data' else ''

# Reading-Only paths
domain_path = add + 'Data/Domain/'

# Writing-Only paths
results_path_csv = add + 'Results/CSV/'
results_path_pdf = add + 'Results/PDF/'

# Reading-And-Writing paths
data_path = add + 'Data/'
raw_data_path = add + 'Data/Raw/'
trained_model_path = add + 'Trained_Model/'
# results_path = 'Results/'

jsons_path = add + 'Data/Jsons/'
csv_path = add + 'Data/CSV/'


problem_path = add + 'Data/Problems/'
valid_solutions_path = add + 'Data/Valid/Solutions/'
non_valid_solutions_path = add + 'Data/Non valid/Solutions/'

#  ################################################################  #
check_and_create_dir(results_path_csv)
check_and_create_dir(domain_path)
check_and_create_dir(data_path)
check_and_create_dir(jsons_path)
check_and_create_dir(csv_path)
check_and_create_dir(problem_path)
check_and_create_dir(valid_solutions_path)
check_and_create_dir(non_valid_solutions_path)
