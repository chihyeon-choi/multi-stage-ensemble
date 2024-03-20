import torch
import argparse
import util
from util import accuracy, log_display, AverageMeter
import os
import datetime
import random
import mlconfig
import loss
import models
import dataset
import shutil
import time
from util import load_model
from util_relabel import evaluate_final_model, loss_separation, get_label_info
from util_relabel import sort_loss_per_class, mean_vector_per_class
from util_relabel import relabel_criterion, relabeling_dataset
from util_train import train
import numpy as np
import torch.nn.functional as F

"""
[Real-World Noisy Labeled Dataset Example]
python main.py --exp_name test
"""

# ArgParse
parser = argparse.ArgumentParser(description='Normalized Loss Functions for Deep Learning with Noisy Labels')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--config_path', type=str, default='configs/')
parser.add_argument('--version', type=str, default='nce+rce')
parser.add_argument('--exp_name', type=str, default="clothing1m")
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--small_loss_ratio', type=float, default=0.1)
parser.add_argument('--relabel_threshold', type=float, default=0.8)
parser.add_argument('--confidence_threshold', type=float, default=0.8)
parser.add_argument('--n_stage', type=int, default=5)
parser.add_argument('--print_freq', type=int, default=50)
args = parser.parse_args()
    

# Set up
if args.exp_name == '' or args.exp_name is None:
    args.exp_name = 'exp_' + datetime.datetime.now()
exp_path = os.path.join(args.exp_name)
log_file_path = os.path.join(exp_path, exp_path)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
checkpoint_path_file = os.path.join(checkpoint_path, 'nce+rce')
util.build_dirs(exp_path)
util.build_dirs(checkpoint_path)

logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")
for arg in vars(args):
    logger.info("%s: %s" % (arg, getattr(args, arg)))

random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    logger.info("Using CUDA!")
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

logger.info("PyTorch Version: %s" % (torch.__version__))
config_file = os.path.join(args.config_path, args.version) + '.yaml'
config = mlconfig.load(config_file)
config.set_immutable()
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))
for key in config:
    logger.info("%s: %s" % (key, config[key]))
    

# clean_or_not : [True, False, True, True, ...] -> True : clean data, False : noisy labeled data
# true_labels : list of true labels [true class of index 1, true class of index 2, ...]
# noisy_labels : list of noisy labels [noisy class of index 1, noisy class of index2, ...]
global data_loader
global test_data_loader

# Dataset load
data_loader = dataset.Clothing1MDataLoader(data_dir = '../datasets/clothing1m',
                                           batch_size = 64, shuffle=True, validation_split=0.0, num_batches=2000, training=True,
                                           num_workers=0, pin_memory=True, seed=8888)

# test_loader : test_data_loader
test_data_loader = dataset.Clothing1MDataLoader(data_dir = '../datasets/clothing1m',
                                           batch_size = 128, shuffle=False, validation_split=0.0, training=False,
                                           num_workers=0).split_validation()
    
class_num = 14
    
def main():
    global data_loader
    global test_data_loader
    # Train each stage model on the progressively refined dataset
    for stage in range(args.n_stage):
        
        # Definition of the model model
        if stage == 0:
            model = config.model()
            model.fc = torch.nn.Linear(2048, 14)
            model = model.to(device)
        
        logger.info("param size = %fMB", util.count_parameters_in_MB(model))
        if args.data_parallel:
            model = torch.nn.DataParallel(model)

        # Defintion of optimizer, scheduler, criterion
        optimizer = config.optimizer(model.parameters())
        scheduler = config.scheduler(optimizer)
        criterion = config.criterion()
            
        starting_epoch = 0
        ENV = {'global_step': 0,
                'best_acc': 0.0,
                'current_acc': 0.0,
                'train_history': [],
                'eval_history': [],
                'train_loss': [],
                'test_loss': []}
        
        # Train model and save the best model on the validation set
        train_acc1, test_acc1, ENV, model = train(starting_epoch,model,optimizer,scheduler,criterion,data_loader, test_data_loader,ENV, stage, logger, config, checkpoint_path_file)

        # load the saved model
        model = load_model(args, config, stage, device)
        # Test accuracy of the current stage model
        final_test_acc = evaluate_final_model(test_data_loader, model, torch.nn.CrossEntropyLoss())
        logger.info('model_acc : {}'.format(final_test_acc))

        if stage < args.n_stage - 1:
            # Sort loss for each class
            sorted_passive_loss_list, sorted_passive_ind_list, image_path_dict = sort_loss_per_class(model, data_loader, criterion)
            # Separate small-loss dataset and large-loss dataset 
            indexes_dict, reference_indexes, ratio_criteria = loss_separation(args.small_loss_ratio, args.relabel_threshold, sorted_passive_loss_list, sorted_passive_ind_list, logger)
            # Compute mean feature vector for each class
            mean_vector_list = mean_vector_per_class(model, data_loader, reference_indexes, indexes_dict, logger)
            # Compute re-labeling confidence value on the each large loss data
            relabel_info = relabel_criterion(model, data_loader, indexes_dict, mean_vector_list, args.confidence_threshold, logger)
            # Refine dataset by re-labeling
            data_loader = relabeling_dataset(data_loader, relabel_info, image_path_dict)
        else:
            logger.info('Successfully trained for total stage')

    test_data_loader = dataset.Clothing1MDataLoader(data_dir = '../datasets/clothing1m',
                                            batch_size = 128, shuffle=False, validation_split=0.0, training=False,
                                            num_workers=0).split_validation()

    # Load all the trained models
    model_dict = {}
    for i in range(args.n_stage):
        model = config.model()
        model.fc = torch.nn.Linear(2048, 14)
        model = model.to(device)
        weight_path = os.path.join(checkpoint_path, 'nce+rce_{}.pth'.format(i))
        checkpoints = torch.load(weight_path)
        model.load_state_dict(checkpoints['model'])
        model_dict[i] = model
        
    # Ensemble model (majority voting)
    correct = 0
    total = 0
    prediction_dict = {}
    # for images, labels in data_loader['test_dataset']:
    for images, labels, _, _ in test_data_loader:
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        with torch.no_grad():
            for i, (_, model) in enumerate(model_dict.items()):
                model.eval()
                pred = model(images)
                softmax = F.softmax(pred[0].data, dim=1)
                confidence, predicted = torch.max(softmax.data,1)
                prediction_dict[i] = (predicted, confidence)
                
            # ensemble rule
            ensemble_predicted = []
            for i in range(labels.size(0)):
                voting_list = [0 for n in range(14)]
                confidence_list = [0 for n in range(14)]
                for temp_predicted, temp_confidence in list(prediction_dict.values()):
                    pred_class = temp_predicted[i].item()
                    voting_list[pred_class] += 1

                    current_confidence = temp_confidence[i].item()
                    if current_confidence > confidence_list[pred_class]:
                        confidence_list[pred_class] = current_confidence
                
                np_voting = np.array(voting_list)
                max_index = np.argwhere(np_voting == np.max(np_voting)).flatten()
                
                if len(max_index) == 1:
                    ensemble_predicted.append(max_index[0])
                else:
                    np_confidence = np.array(confidence_list)
                    max_confidence_class = np.argwhere(np_confidence == np.max(np_confidence)).flatten()[0]
                    ensemble_predicted.append(max_confidence_class)
            
            ensemble_predicted = torch.tensor(ensemble_predicted).to(device, non_blocking=True)
            
            total += labels.size(0)
            correct += (ensemble_predicted == labels).sum().item()
    logger.info('ensemble model accuracy : {}'.format(correct/total))
    
    return
    
    


if __name__ == '__main__':
    main()
