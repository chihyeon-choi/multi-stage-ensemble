import numpy as np
from util import accuracy, log_display, AverageMeter
import torch
import torch.nn.functional as F
import time
import sys


def get_label_info(noisy_loader, num_class, logger):
    
    noisy_labels = [0 for i in range(num_class)]
    for batch_idx, (data, label, indexs, img_path) in enumerate(noisy_loader):
        
        for path in img_path:
            temp_label = noisy_loader.train_dataset.train_labels[path]
            noisy_labels[temp_label] += 1
    
    print(noisy_labels)

def label_show(total_true_label, total_noisy_label, logger):
    
    cls_num = len(np.unique(total_true_label))
    label_mat = np.zeros((cls_num, cls_num), dtype=np.int64)
    
    for i in range(len(total_true_label)):
        row = total_true_label[i]
        column = total_noisy_label[i]
        label_mat[row, column] += 1
    
    logger.info('\n' + str(label_mat))


def evaluate_final_model(data_loader, final_model, criterion):
    
    device = torch.device('cuda')
    
    loss_meters_1 = AverageMeter()
    acc_meters_1 = AverageMeter()
    acc5_meters_1 = AverageMeter()
    
    final_model.eval()
    with torch.no_grad():
        for i, (images, labels, indexes, img_path) in enumerate(data_loader):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            logits1, _ = final_model(images)
            loss1 = criterion(logits1, labels)
            
            acc_1, acc_1_5 = accuracy(logits1, labels, topk=(1,5))
            loss_meters_1.update(loss1.item(), labels.shape[0])
            acc_meters_1.update(acc_1.item(), labels.shape[0])
            acc5_meters_1.update(acc_1_5.item(), labels.shape[0])
        
    return acc_meters_1.avg


def sort_loss_per_class(model, data_loader, criterion, class_num):
    
    device = torch.device('cuda')

    model.eval()
    
    passive_loss_list = [[] for i in range(class_num)]
    sorted_passive_loss_list = []
    total_ind_list = [[] for i in range(class_num)]
    sorted_passive_ind_list = []
    
    with torch.no_grad():
        for i, (images, labels, indexes, img_path) in enumerate(data_loader):
            
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            ind = list(map(int, indexes))
            
            logits, _ = model(images)
            loss, active_loss, passive_loss = criterion(logits, labels)
            
            loss = loss.tolist()
            active_loss = active_loss.tolist()
            passive_loss = passive_loss.tolist()
            labels = labels.tolist()
            
            for j in range(len(loss)):
                temp_label = labels[j]
                passive_loss_list[temp_label].append(passive_loss[j])
                total_ind_list[temp_label].append(ind[j])
    
    for i in range(len(passive_loss_list)):
        lst_passive_loss = np.array(passive_loss_list[i])
        lst_ind = np.array(total_ind_list[i])
        
        sorted_args = np.argsort(lst_passive_loss)
        sorted_passive_loss_list.append(lst_passive_loss[sorted_args])
        sorted_passive_ind_list.append(lst_ind[sorted_args])
    
    return sorted_passive_loss_list, sorted_passive_ind_list
        

def loss_separation(small_loss_ratio, relabel_threshold, sorted_total_loss_list, sorted_total_ind_list, logger):
    
    print('dividing total lnstances into small loss instance & large loss instance')
    
    indexes_dict = {}
    reference_indexes = []
    ratio_criteria = []
    
    for i in range(len(sorted_total_ind_list)):
        # original ==> num small loss = 전체 데이터에서 small_loss_ratio
        # num_remember : number of small loss instances
        small_num_remember = int(small_loss_ratio * len(sorted_total_ind_list[i]))
        # num_reference_index : number of reference images per class
        num_reference_index = small_num_remember
        
        # small_loss_ind : indexes of small loss instances
        small_loss_ind = sorted_total_ind_list[i][:small_num_remember]
        # reference_ind : indexes of reference instances
        reference_ind = small_loss_ind[:num_reference_index]
        reference_indexes.append(reference_ind)
        # original ==> num small loss = 전체 데이터에서 small_loss_ratio
        
        # large_loss_ind : indexes of large loss instances
        large_loss_criteria = sorted_total_loss_list[i][-1] * relabel_threshold
        ind_criteria = np.where(np.array(sorted_total_loss_list[i]) > large_loss_criteria)[0][0]
        large_num_remember = len(sorted_total_loss_list[i]) - ind_criteria + 1
        large_loss_ind = sorted_total_ind_list[i][-large_num_remember:]
        ratio_criteria.append(ind_criteria/len(sorted_total_loss_list[i]))
        
        key_name_small = 'class_' + str(i) + '_small'
        key_name_large = 'class_' + str(i) + '_large'
        
        indexes_dict[key_name_small] = small_loss_ind
        indexes_dict[key_name_large] = large_loss_ind
        
    total_small_loss = 0
    total_ref_loss = 0
        
    for i in range(len(reference_indexes)):
        total_small_loss += len(indexes_dict['class_'+str(i)+'_small'])
        total_ref_loss += len(reference_indexes[i])
        
        txt_1 = 'class_{}_num_reference : {}'.format(str(i), len(reference_indexes[i]))
        logger.info(txt_1)
        
    txt_4 = 'total_small_loss_number : {}'.format(total_small_loss)
    txt_7 = 'total_ref_number : {}'.format(total_ref_loss)
    logger.info(txt_4)
    logger.info(txt_7)
        
    return indexes_dict, reference_indexes, ratio_criteria

# indexes_dict : {'class_k_small':[class_k_small_index], 'class_k_large':[class_k_large_index]}
# reference_indexes : [[class1 small_loss_index], [class2 small_loss_index], ...]


def mean_vector_per_class(model, data_loader, reference_indexes, indexes_dict, logger):
    
    # reference image dictionary
    ref_image_dict = {}
    for images, labels, indexes, img_path in data_loader:
        ind = list(map(int, indexes))
        
        for i in range(len(reference_indexes)):
            for j in range(len(ind)):
                if ind[j] in reference_indexes[i]:
                    ref_image_dict[ind[j]] = images[j]
    
    # mean vector per class : reference feature vector for re-labeling
    mean_vector_list = []
    
    total_small_loss = 0
    total_ref_loss = 0
    
    model.eval()
    with torch.no_grad():
        for i in range(len(reference_indexes)):
            ref_image = [ref_image_dict[ind] for ind in reference_indexes[i]]
            
            total_small_loss += len(indexes_dict['class_'+str(i)+'_small'])
            total_ref_loss += len(reference_indexes[i])
            
            txt_1 = 'class_{}_num_reference : {}'.format(str(i), len(reference_indexes[i]))
            logger.info(txt_1)
            
            temp_vec = []
            for j in range(len(ref_image)):
                image = ref_image[j]
                image = torch.unsqueeze(image, 0).cuda()
                _, feature_vector = model(image)
                feature_vector = feature_vector.reshape(-1).tolist()
                temp_vec.append(feature_vector)
            temp_vec = torch.tensor(temp_vec)
            mean_vector_list.append(torch.mean(temp_vec, axis=0))
        
    txt_4 = 'total_small_loss_number : {}'.format(total_small_loss)
    txt_7 = 'total_ref_number : {}'.format(total_ref_loss)
    logger.info(txt_4)
    logger.info(txt_7)
    
    return mean_vector_list

def similarity_cosine(tensor_1, tensor_2):
    
    np_1 = tensor_1.cpu().detach().numpy()
    np_2 = tensor_2.cpu().detach().numpy()
    cos_sim = np.dot(np_1, np_2)/(np.linalg.norm(np_1) * np.linalg.norm(np_2))
    
    return cos_sim

def similarity_euclidean(tensor_1, tensor_2):
    
    np_1 = tensor_1.cpu().detach().numpy()
    np_2 = tensor_2.cpu().detach().numpy()
    euclidean_dist = np.linalg.norm(np_1-np_2)
    
    return -euclidean_dist

def distance_euclidean(tensor_1, tensor_2):
    
    np_1 = tensor_1.cpu().detach().numpy()
    np_2 = tensor_2.cpu().detach().numpy()
    euclidean_dist = np.linalg.norm(np_1-np_2)
    
    return euclidean_dist

def flatten(lst):
    result = []
    for item in lst:
        if type(item) == list:
            result += flatten(item)
        else:
            result += [item]
    return result

def relabel_criterion(model, data_loader, indexes_dict, mean_vector_list, relabel_threshold, class_num, logger):
    
    # class_num = 14
    
    large_loss_image = {}
    total_large_index = []
    
    for k,v in indexes_dict.items():
        if 'large' in k:
            total_large_index.append(v.tolist())
    total_large_index = flatten(total_large_index)
            
    for images, labels, indexes, img_path in data_loader:
        ind = list(map(int, indexes))
        for i in range(len(ind)):
            if ind[i] in total_large_index:
                large_loss_image[ind[i]] = images[i]
        
    
    # relabel_info : [[index, original_label, re_label], ...]
    relabel_info = []
    
    # re-labeling using total large loss samples
    with torch.no_grad():
        for i in range(class_num):
            print('re-labeling is proceeding for large loss samples in class {}'.format(i))
            key_name = 'class_' + str(i) + '_large'
            txt_1 = 'number of large loss data in original class : {}'.format(len(indexes_dict[key_name]))
            logger.info(txt_1)
            
            large_loss_ind = [ind for ind in indexes_dict[key_name]]
            large_loss_image_dataset = [large_loss_image[ind] for ind in indexes_dict[key_name]]
            
            for j in range(len(large_loss_image_dataset)):
                temp_index = large_loss_ind[j]
                image = large_loss_image_dataset[j]
                image = torch.unsqueeze(image, 0).cuda()
                _, feature_vector = model(image)
                feature_vector = feature_vector.reshape(-1)
                
                # cos_similarity_list = [similarity_cosine(feature_vector, mean_vector) for mean_vector in mean_vector_list]
                # cos_softmax = F.softmax(torch.Tensor(cos_similarity_list), dim=0)
                eu_similarity_list = [similarity_euclidean(feature_vector, mean_vector) for mean_vector in mean_vector_list]
                eu_softmax = F.softmax(torch.Tensor(eu_similarity_list), dim=0)
                max_value, candidate_class = torch.max(eu_softmax, 0)
                max_value = max_value.item()
                
                # if max_value >= 0.90:
                if max_value >= relabel_threshold:
                    
                    relabel_info.append([temp_index, i, candidate_class])

    return relabel_info
        

def relabeling_dataset(data_loader, relabel_info):
    # relabel_info : [[index, original_label, re_label], ...]
# for k,v in data_loader.train_dataset.train_labels.items():
#     data_loader.train_dataset.train_labels[k] = 1

    for i in range(len(relabel_info)):
        temp_index = relabel_info[i][0]
        assert data_loader.dataset.imgLabel[temp_index] == relabel_info[i][1]
        
        temp_relabeled = relabel_info[i][2].item()
        temp_relabeled = np.int64(temp_relabeled)
    
        data_loader.dataset.imgLabel[temp_index] = temp_relabeled
        
    print('Successfully Constructed re-labeled dataset!')
        
    return data_loader