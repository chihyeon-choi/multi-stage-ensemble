import numpy as np
from util import accuracy, log_display, AverageMeter
import torch
import torch.nn.functional as F
import time
from util import loss_compute
import util

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def evaluate(model, data_loader, criterion, global_step, epoch, stage, logger):
    
    loss_meters_1 = AverageMeter()
    acc_meters_1 = AverageMeter()
    acc5_meters_1 = AverageMeter()
    
    model.eval()
    with torch.no_grad():
        for i, (images, labels, indexes, img_path) in enumerate(data_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            start = time.time()
            
            logits, _ = model(images)
            loss = criterion(logits, labels)
            
            end = time.time()
            time_used = end - start
            
            acc_1, acc_1_5 = accuracy(logits, labels, topk=(1,5))
            loss_meters_1.update(loss.item(), labels.shape[0])
            acc_meters_1.update(acc_1.item(), labels.shape[0])
            acc5_meters_1.update(acc_1_5.item(), labels.shape[0])
        
        
    logger_payload =  {'acc1_avg' : acc_meters_1.avg,
                       'loss' : loss_meters_1.avg}
    
    display = log_display(stage = stage,
                          epoch = epoch,
                          global_step = global_step,
                          time_elapse = time_used,
                          **logger_payload)
    logger.info(display)
    
    return acc_meters_1.avg, loss_meters_1.avg


def train_epoch(model, data_loader, optimizer, ENV, epoch, criterion, stage, logger, config):
    # clean_or_not, total_true_label, total_noisy_label
    
    model.train()
    
    loss_meters_1 = AverageMeter()
    acc_meters_1 = AverageMeter()
    acc5_meters_1 = AverageMeter()
    
    for i, (images, labels, indexes, img_path) in enumerate(data_loader):
        
        images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        ENV['global_step'] += 1
        
        start = time.time()
        
        logits, _ = model(images)
        acc_1, acc_1_5 = accuracy(logits, labels, topk=(1,5))
        
        loss = loss_compute(logits, labels, criterion)
        model.zero_grad()
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_bound)
        optimizer.step()
        
        loss_meters_1.update(loss.item(), labels.shape[0])
        acc_meters_1.update(acc_1.item(), labels.shape[0])
        acc5_meters_1.update(acc_1_5.item(), labels.shape[0])
            
        logger_payload =  {'acc1_avg' : acc_meters_1.avg,
                           'loss' : loss_meters_1.avg,
                           'lr' : optimizer.param_groups[0]['lr'],
                           '|gn|' : grad_norm}
            
        end = time.time()
        time_used = end - start
        
        if ENV['global_step'] % config.log_frequency == 0:
            display = log_display(stage = stage,
                                  epoch = epoch,
                                  global_step = ENV['global_step'],
                                  time_elapse = time_used,
                                  **logger_payload)
            logger.info(display)

    return acc_meters_1.avg, model, ENV, loss_meters_1.avg
    

def train(starting_epoch, model, optimizer, scheduler, criterion, data_loader, test_data_loader, ENV, stage, logger, config, save_path):

    best_test_acc = 0
    
    for epoch in range(starting_epoch, 100):
    # for epoch in range(starting_epoch, 5):
        
        logger.info('='*20 + 'Training' + '='*20)
        
        train_acc1, model, ENV, loss = train_epoch(model,data_loader,optimizer,ENV, epoch, criterion, stage, logger, config)
        
        scheduler.step()
        
        # evaluate
        logger.info('='*20 + 'Eval' + '='*20)
        test_acc1, test_loss_1 = evaluate(model, test_data_loader, torch.nn.CrossEntropyLoss(), ENV['global_step'], epoch, stage, logger)
        
        ENV['train_history'].append(train_acc1)
        ENV['eval_history'].append(test_acc1)
        ENV['current_acc'] = test_acc1
        ENV['best_acc'] = max(ENV['current_acc'], ENV['best_acc'])
        ENV['train_loss'].append(loss)
        ENV['test_loss'].append(test_loss_1)

        if test_acc1 > best_test_acc:
            print('test accuracy improved : {:.5f} => {:.5f}'.format(best_test_acc, test_acc1))
            best_test_acc = test_acc1
            util.save_model(ENV = ENV,
                            epoch = epoch,
                            model = model,
                            optimizer = optimizer,
                            scheduler = scheduler,
                            filename = save_path + '_' + str(stage))
    
    return train_acc1, test_acc1, ENV, model