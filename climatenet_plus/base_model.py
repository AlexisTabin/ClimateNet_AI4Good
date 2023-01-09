import traceback

from climatenet_plus.climatenet.models.trainer import Trainer
from climatenet_plus.climatenet.utils.data import ClimateDataset, ClimateDatasetLabeled


def train(config):
    model_name = config.architecture
    model = Trainer(config, model_name)

    data_dir = config.data_dir
    train_path = data_dir + 'train/'
    val_path = data_dir + 'val/'
    
    print('train_path : ', train_path)
    print('val_path : ', val_path)

    print('Loading data...')
    train = ClimateDatasetLabeled(train_path, config)
    val = ClimateDatasetLabeled(val_path, config)

    save_dir = config.save_dir
    model.train(train)
    model.evaluate(val)
    model.save_model(save_dir)
    
    
def evaluate(config):
    model_name = config.architecture
    model = Trainer(config, model_name)

    save_dir = config.save_dir
    data_dir = config.data_dir

    inference_path = data_dir + 'test/'

    inference = ClimateDataset(inference_path, config)
    
    model.load_model(save_dir, model_name)

    print('inference_path : ', inference_path)

    try:
        print('evaluating on test data...')
        test = ClimateDatasetLabeled(inference_path, config)
        model.evaluate(test)
    except Exception as e:
        print('error in evaluating on test data...')
        print(e)
        traceback.print_exc()
    
    print("Finished evaluation!")

