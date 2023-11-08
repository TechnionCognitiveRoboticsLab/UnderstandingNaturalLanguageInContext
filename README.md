# ALFRED-T5
This repository is the implementation of the ICAPS 2023 paper - Understanding Natural Language in Context (https://ojs.aaai.org/index.php/ICAPS/article/view/27248) by Avichai Levy and Erez Karpas.

To reproduce the results presented in the paper:

1. Make sure that VAL plan validator (https://github.com/KCL-Planning/VAL) is installed on your computer.
2. Clone the repository and extract the data file from "Data/CSV.zip" 
This data is the original ALFRED data after the plan validation process and the scene data integration, if you wish to validate the original data again, make sure you download the ALFRED dataset first. However, the scene information will need to be loaded again from the AI2Thor simulator.
3. There are two bash scripts provided, you can edit the scripts to change the hyper-parameter or the model.
4. In terminal run the command: bash script_name.sh (which run the main.py file with required parameters)
5. For the PDDL valid plan scores, after the training process edit and run the script "Create_PDF_with_pddl_acc.py" with the training size that the model was trained on.


All experiments were conducted using the NVIDIA A100 GPU, with 40GB GPU RAM. Tuning the models on the Task, Relations, and Task+Relations inputs types last for 2, 2, and 3 hours on the T5-base model, and 1, 4, and 5 hours on the GPT2-medium model, respectively.
