import traceback
from os import path

from climatenet.analyze_events import analyze_events
from climatenet.models.trainer import MODELS, Trainer
from climatenet.track_events import track_events
from climatenet.utils.data import ClimateDataset, ClimateDatasetLabeled
from climatenet.utils.utils import Config
from climatenet.visualize_events import visualize_events


def run():
    config = Config('config.json')
    model_name = config.architecture
    model = Trainer(config, model_name)

    data_dir = config.data_dir
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

    save_dir = config.save_dir
    if not config.is_already_trained:
        model.train(train)
        model.evaluate(val)
        model.save_model(save_dir)
    else:
        model.load_model(save_dir)

    try:
        print('evaluating on test data...')
        test = ClimateDatasetLabeled(inference_path, config)
        model.evaluate(test)
    except Exception as e:
        print('error in evaluating on test data...')
        print(e)
        traceback.print_exc()
    

    # masks with 1==TC, 2==AR
    class_masks = model.predict(inference, save_dir=save_dir)
    event_masks = track_events(class_masks)  # masks with event IDs

    if config.with_analysis:
        try:
            analyze_events(event_masks, class_masks, save_dir + 'results/')
        except Exception as e:
            print("Error when analyzing events : ", e)
            # Uncomment if you want to see the traceback of the error
            # print('\n'*3)
            # print('traceback : ', traceback.format_exc())

    if config.with_visualization:
        try:
            print('-'*50)
            print('visualizing inference events...')
            print('-'*50)
            print('\n')
            visualize_events(event_masks, inference, save_dir + 'pngs/')
        except Exception as e:
            print("Error when visualizing events of inference : ", e)
            print('\n'*3)
            print('traceback : ', traceback.format_exc())
            pass


if __name__ == '__main__':
    run()