from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from climatenet.models.upernet import UperNet
from climatenet.utils.utils import Config
from climatenet.track_events import track_events
from climatenet.analyze_events import analyze_events
from climatenet.visualize_events import visualize_events

import traceback
from os import path

def run(model='upernet', checkpoint_path='', data_dir='', save_dir=''):
    config = Config(f'climatenet/trained_{model}/{model}_config.json')
    upernet = UperNet(config)

    train_path = data_dir + 'train/'
    val_path = data_dir + 'val/'
    inference_path = data_dir + 'test/' 

    print('train_path : ', train_path)
    print('val_path : ', val_path)
    print('inference_path : ', inference_path)

    print('Loading data...')
    train = ClimateDatasetLabeled(train_path, config)
    val = ClimateDatasetLabeled(val_path, config)
    inference = ClimateDataset(inference_path, config)

    # upernet.train(train)
    # upernet.evaluate(val)
    # upernet.save_model(checkpoint_path)
    upernet.load_model(checkpoint_path)   
    upernet.evaluate(inference)

    class_masks = upernet.predict(inference, save_dir=save_dir) # masks with 1==TC, 2==AR
    event_masks = track_events(class_masks) # masks with event IDs

    try :
        analyze_events(event_masks, class_masks, save_dir + 'results/')
    except Exception as e:
        print("Error when analyzing events : ", e)
        # Uncomment if you want to see the traceback of the error
        # print('\n'*3)
        # print('traceback : ', traceback.format_exc())

    try : 
        visualize_events(event_masks, inference, save_dir + 'pngs/')
    except Exception as e:
        print("Error when visualizing events : ", e)
        print('\n'*3)
        print('traceback : ', traceback.format_exc())
