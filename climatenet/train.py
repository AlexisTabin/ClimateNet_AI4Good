from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from climatenet.models.upernet.upernet import UperNet
from climatenet.models.cgnet.cgnet import CGNet
from climatenet.models.unet.unet import UNet
from climatenet.utils.utils import Config
from climatenet.track_events import track_events
from climatenet.analyze_events import analyze_events
from climatenet.visualize_events import visualize_events

import traceback
from os import path

string_to_model = {
    'upernet': UperNet,
    'cgnet': CGNet,
    'unet': UNet
}

def run(model_name='upernet', checkpoint_path='', data_dir='', save_dir=''):
    config = Config(f'climatenet/models/{model_name}/{model_name}_config.json')
    model = string_to_model[model_name](config)

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

    
    # model.train(train)
    # model.evaluate(val)
    # model.save_model(checkpoint_path)
    model.load_model(checkpoint_path)   

    try :
        print('evaluating on val set...')
        model.evaluate(val)
    except Exception as e:
        print('Error in evaluating on val set : ', e)
        traceback.print_exc()
        
    try :
        print('evaluating on test data...')
        test = ClimateDatasetLabeled(inference_path, config)
        model.evaluate(test)
    except Exception as e:
        print('error in evaluating on test data...')
        print(e)
        traceback.print_exc()

    class_masks = model.predict(inference, save_dir=save_dir) # masks with 1==TC, 2==AR
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
