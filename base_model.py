import traceback
from os import path

from climatenet_base.analyze_events import analyze_events
from climatenet_base.models.trainer import MODELS, Trainer
from climatenet_base.track_events import track_events
from climatenet_base.utils.data import ClimateDataset, ClimateDatasetLabeled
from climatenet_base.visualize_events import visualize_events


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

    # masks with 1==TC, 2==AR
    # class_masks = model.predict(inference, save_dir=save_dir)
    # event_masks = track_events(class_masks)  # masks with event IDs

    # if config.with_analysis:
    #     try:
    #         analyze_events(event_masks, class_masks, save_dir + 'results/')
    #     except Exception as e:
    #         print("Error when analyzing events : ", e)
    #         # Uncomment if you want to see the traceback of the error
    #         # print('\n'*3)
    #         # print('traceback : ', traceback.format_exc())

    # if config.with_visualization:
    #     try:
    #         print('-'*50)
    #         print('visualizing inference events...')
    #         print('-'*50)
    #         print('\n')
    #         visualize_events(event_masks, inference, save_dir + 'pngs/')
    #     except Exception as e:
    #         print("Error when visualizing events of inference : ", e)
    #         print('\n'*3)
    #         print('traceback : ', traceback.format_exc())
    #         pass

