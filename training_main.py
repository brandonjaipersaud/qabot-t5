""" 
Usage: 
    training_main.py clean <raw-datasets-dir> <clean-dir>
    training_main.py split <clean-datasets-dir> <split-dir> [-t <test-file>]
    training_main.py train <json-path>
    training_main.py gpu-test
    training_main.py print-stats <dataset-path>


Notes:
    The main usage you will need to use is:
        python3.9 training_main.py train <json-path>
    where <json-path> is the path to a config json file.

    This is used for model training/evaluation/prediction

"""

from docopt import docopt
from pathlib import Path
from utils.data_cleaning import *
from model.model import *
from utils.gpu_utils import *

import torch


def main(args):

    print(args)

    if args['clean']:
        raw_path = Path(args['<raw-datasets-dir>'])
        for f in raw_path.iterdir():
            # only clean files ending with csv
            file_type = f.name.split('.')[-1]
            if file_type == 'csv':
                print(f'CLEANING {f.name}')
                cleaned_path = Path(args['<clean-dir>'])
                cleaned_name = 'cleaned_' + f.name 
                cleaned_path = cleaned_path.joinpath(cleaned_name)
                clean_csv(f.as_posix(), cleaned_path.as_posix())

    if args['split']:
        test_file = args['<test-file>']
        test_set = None
        clean_path = Path(args['<clean-datasets-dir>'])
        final_path= args['<split-dir>']
        cleaned_datasets = []
        for f in clean_path.iterdir():
            # only add if not test file
            if not test_file or ( test_file and f.name != test_file ):
                print(f'Adding file {f.name}')
                cleaned_datasets.append(pd.read_csv(f))
            else:
                print(f'Skipping test file {test_file}')
                test_set = pd.read_csv(f)
        # 90% training 10% val  
        combine_and_split(cleaned_datasets, 0.90, 0.5, final_path, test_set=test_set)

    if args['train']:
        
        train(args["<json-path>"])

    if args['gpu-test']:
        # print(cuda.current_device())
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        if device.type == 'cpu':
            print('USING CPU=BAD!')
            return 
        print(f'Using device: {device}')
        #print_gpu_info(device)
        print_gpu_utilization()
        # torch.ones((10000, 10000)).to(device)
        # print_gpu_utilization()

        # print(cuda.list_gpu_processes())

    if args['print-stats']:
        print_stats(args['<dataset-path>'])        

if __name__ == "__main__":
    arguments = docopt(__doc__)
    print(arguments)
    main(arguments)

