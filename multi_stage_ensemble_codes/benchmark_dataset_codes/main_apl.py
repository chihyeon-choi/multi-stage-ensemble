import torch
import argparse
import util
import os
import datetime
import random
import mlconfig
import loss
import models
import dataset
import shutil
import numpy as np
from util import load_model
from util_relabel import compute_noise_rate, get_label_info
from util_relabel import evaluate_final_model, sort_loss_per_class
from util_relabel import loss_separation, mean_vector_per_class
from util_relabel import relabel_critetion, relabeling_dataset, relabel_acc
import torch.nn.functional as F
from util_train import train

"""
[Benchmark Dataset Example]
For symmetric 20% noise label:
python main.py --dataset mnist --exp_name test1 --noise_ratio 0.2

For asymmetric 20% noise label:
python main.py --dataset cifar10 --exa_name test2 --noise_ratio 0.2 --asym
"""


# ArgParse
parser = argparse.ArgumentParser(description='Multi-stage Ensemble with Refinement for Noisy Labeled Data Classification')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--config_path', type=str, default='configs/')
parser.add_argument('--version', type=str, default='nce+rce')
parser.add_argument('--dataset', type=str, default="cifar10") # mnist, fmnist, cifar10
parser.add_argument('--exp_name', type=str, default="cifar10")
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--asym', action='store_true', default=False)
parser.add_argument('--noise_ratio', type=float, default=0.2)
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
np.random.seed(args.seed)
torch.manual_seed(args.seed)
# torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    # torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    logger.info("Using CUDA!")
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

logger.info("PyTorch Version: %s" % (torch.__version__))
noise_type = 'asym' if args.asym else 'sym'
yaml_name = '{}_{}.yaml'.format(args.dataset,noise_type)
config_file = os.path.join(args.config_path, yaml_name)
config = mlconfig.load(config_file)
config.set_immutable()
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))
for key in config:
    logger.info("%s: %s" % (key, config[key]))


# Dataset load and check the generated noisy labeled dataset
global data_loader
global total_noisy_label
global clean_or_not
if config.dataset.name == 'DatasetGenerator':
    clean_loader = config.dataset(seed=args.seed, noise_rate=0, asym=False)
    data_loader = config.dataset(seed=args.seed, noise_rate=args.noise_ratio, asym=args.asym)
    clean_or_not, total_true_label, total_noisy_label = get_label_info(clean_loader, data_loader, logger)
    class_num = len(np.unique(np.array(total_true_label)))


def main():
    global data_loader
    global total_noisy_label
    global clean_or_not
    
    # Train each stage model on the progressively refined dataset
    for stage in range(args.n_stage):
        # Check the number of training data and noisy labeled data in each class
        # In the case of stage 0, we can check the initially generated noisy labeled dataset
        logger.info('total number of train data : {}'.format(len(clean_or_not)))
        logger.info('total noisy labeled data : {}'.format(len(clean_or_not) - np.sum(clean_or_not)))
        logger.info('actual noise ratio : {}'.format((len(clean_or_not) - np.sum(clean_or_not)) / len(clean_or_not)))
        
        # Definition of the model model
        model = config.model()
        model = model.to(device)
        
        # Update noisy label in each stage
        if stage == 0:
            del data_loader
            data_loader = config.dataset(seed=args.seed, noise_rate=args.noise_ratio, asym=args.asym)
            data_loader = data_loader.getDataLoader()
            total_noisy_label = data_loader['train_dataset'].dataset.train_targets
        else:
            total_noisy_label = data_loader['train_dataset'].dataset.train_targets
        
        # The number of clean label, the ratio of clean label and the number of training data in each class
        num_clean_list, ratio_clean_list, num_class_list = compute_noise_rate(total_true_label, total_noisy_label, logger)
        
        logger.info("param size = %fMB", util.count_parameters_in_MB(model))
        if args.data_parallel:
            model = torch.nn.DataParallel(model)
    
        # Defintion of optimizer, scheduler, criterion
        optimizer = config.optimizer(model.parameters())
        scheduler = config.scheduler(optimizer)
        criterion = config.criterion()
        ENV = {'global_step': 0,
                'best_acc': 0.0,
                'current_acc': 0.0,
                'train_history': [],
                'eval_history': [],
                'train_loss': [],
                'test_loss': []}
        
        # Train model and save the best model on the validation set
        ENV = train(model,optimizer,scheduler,criterion,data_loader,ENV, stage, logger, config, checkpoint_path_file)
        # load the saved model
        model = load_model(args, config, stage, device)
        # Test accuracy of the current stage model
        final_test_acc = evaluate_final_model(data_loader, model, torch.nn.CrossEntropyLoss())
        logger.info('final_model_acc : {}'.format(final_test_acc))
            
    data_loader = config.dataset(seed=args.seed, noise_rate=args.noise_ratio, asym=args.asym)
    data_loader = data_loader.getDataLoader()
    
    # Load all the trained models
    model_dict = {}
    for i in range(args.n_stage):
        model = config.model()
        model = model.to(device)
        weight_path = os.path.join(checkpoint_path, 'nce+rce_{}.pth'.format(i))
        checkpoints = torch.load(weight_path)
        model.load_state_dict(checkpoints['model'])
        model_dict[i] = model
        
    # Ensemble model (majority voting)
    correct = 0
    total = 0
    prediction_dict = {}
    for images, labels in data_loader['test_dataset']:    
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
                voting_list = [0 for n in range(10)]
                confidence_list = [0 for n in range(10)]
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
    # print('ensemble model accuracy : {}'.format(correct/total))
    logger.info('ensemble model accuracy : {}'.format(correct/total))
    
    return
    

if __name__ == '__main__':
    main()
