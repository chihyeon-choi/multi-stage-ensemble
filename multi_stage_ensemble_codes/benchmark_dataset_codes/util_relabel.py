import numpy as np
from util import accuracy, AverageMeter
import torch
import torch.nn.functional as F


def get_label_info(clean_loader, noisy_loader, logger):
    
    clean_loader = clean_loader.getDataLoader()
    noisy_loader = noisy_loader.getDataLoader()
    
    true_labels = clean_loader['train_dataset'].dataset.train_targets
    noisy_labels = noisy_loader['train_dataset'].dataset.train_targets
    
    clean_or_not = np.array(true_labels) == np.array(noisy_labels)
    
    return clean_or_not, true_labels, noisy_labels

def compute_noise_rate(total_true_label, total_noisy_label, logger):
    
    class_num = len(np.unique(np.array(total_true_label)))
    
    # noisy label per class
    noisy = [[] for i in range(class_num)] # [[0, 0, ...], [], [], ...]
    # clean label per class, this is matched with noisy
    clean = [[] for i in range(class_num)]# [[true label, true label, ...], [], [], ...]
    
    # clean data num per class
    num_clean_list = [] # [clean data number in class 0, ...]
    # clean ratio per class
    ratio_clean_list = [] # [clean ratio in class 0, ...]
    # total data num per class
    num_class_list = [] # [total data number in class 0, ... ]
    
    for i in range(len(total_noisy_label)):
        temp_class = total_noisy_label[i]
        noisy[temp_class].append(temp_class)
        clean[temp_class].append(total_true_label[i])
        
    for i in range(class_num):
        temp_noisy = np.array(noisy[i])
        temp_clean = np.array(clean[i])
        temp_comparison = temp_noisy == temp_clean
        num_clean = np.sum(temp_comparison)
        ratio_clean = num_clean / float(len(temp_comparison))
        num_clean_list.append(num_clean)
        ratio_clean_list.append(ratio_clean)
        
    for i in range(class_num):
        num_class_list.append(len(noisy[i]))
        
    return num_clean_list, ratio_clean_list, num_class_list


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
        for images, labels in data_loader['test_dataset']:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            logits1, _ = final_model(images)
            loss1 = criterion(logits1, labels)
            
            acc_1, acc_1_5 = accuracy(logits1, labels, topk=(1,5))
            loss_meters_1.update(loss1.item(), labels.shape[0])
            acc_meters_1.update(acc_1.item(), labels.shape[0])
            acc5_meters_1.update(acc_1_5.item(), labels.shape[0])
        
    return acc_meters_1.avg


def sort_loss_per_class(model, data_loader, total_true_label, criterion):
    
    device = torch.device('cuda')
    
    model.eval()
    class_num = len(np.unique(np.array(total_true_label)))
    
    passive_loss_list = [[] for i in range(class_num)]
    sorted_passive_loss_list = []
    total_ind_list = [[] for i in range(class_num)]
    sorted_passive_ind_list = []
    
    with torch.no_grad():
        for i, (images, labels, indexes) in enumerate(data_loader['train_dataset']):
            
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            ind = indexes.cpu().numpy()
            
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
        

def loss_separation(small_loss_ratio, relabel_threshold, sorted_passive_loss_list, sorted_passive_ind_list, clean_or_not, logger):
    
    print('dividing total lnstances into small loss instance & large loss instance')
    
    indexes_dict = {}
    reference_indexes = []
    ratio_criteria = []
    
    for i in range(len(sorted_passive_ind_list)):
        # num_remember : number of small loss instances
        small_num_remember = int(small_loss_ratio * len(sorted_passive_ind_list[i]))
        # num_reference_index : number of reference images per class
        num_reference_index = small_num_remember
        
        # small_loss_ind : indexes of small loss instances
        small_loss_ind = sorted_passive_ind_list[i][:small_num_remember]
        # reference_ind : indexes of reference instances
        reference_ind = small_loss_ind[:num_reference_index]
        reference_indexes.append(reference_ind)
        
        # large_loss_ind : indexes of large loss instances
        large_loss_criteria = sorted_passive_loss_list[i][-1] * relabel_threshold
        ind_criteria = np.where(np.array(sorted_passive_loss_list[i]) > large_loss_criteria)[0][0]
        large_num_remember = len(sorted_passive_loss_list[i]) - ind_criteria + 1
        large_loss_ind = sorted_passive_ind_list[i][-large_num_remember:]
        ratio_criteria.append(ind_criteria/len(sorted_passive_loss_list[i]))
        
        key_name_small = 'class_' + str(i) + '_small'
        key_name_large = 'class_' + str(i) + '_large'
        
        indexes_dict[key_name_small] = small_loss_ind
        indexes_dict[key_name_large] = large_loss_ind
        
    total_small_loss = 0
    clean_small_loss = 0
    total_ref_loss = 0
    clean_ref_loss = 0
        
    for i in range(len(reference_indexes)):
        total_small_loss += len(indexes_dict['class_'+str(i)+'_small'])
        clean_small_loss += np.sum(clean_or_not[indexes_dict['class_'+str(i)+'_small']])
        total_ref_loss += len(reference_indexes[i])
        clean_ref_loss += np.sum(clean_or_not[reference_indexes[i]])
        
        temp_class_pure_ratio = np.sum(clean_or_not[indexes_dict['class_'+str(i)+'_small']]) / float(len(indexes_dict['class_'+str(i)+'_small']))
        temp_ref_pure_ratio = np.sum(clean_or_not[reference_indexes[i]]) / float(len(reference_indexes[i]))
        
        txt_1 = 'class_{}_num_reference : {}'.format(str(i), len(reference_indexes[i]))
        txt_2 = 'class_{}_small_loss_pure_ratio : {:.5f}'.format(str(i), temp_class_pure_ratio)
        txt_3 = 'class_{}_reference_pure_ratio : {:.5f}'.format(str(i), temp_ref_pure_ratio)
        logger.info(txt_1)
        logger.info(txt_2)
        logger.info(txt_3)
        
    txt_4 = 'total_small_loss_number : {}'.format(total_small_loss)
    txt_5 = 'clean_small_loss_number : {}'.format(clean_small_loss)
    txt_6 = 'small_loss_pure_ratio : {}'.format(clean_small_loss / total_small_loss)
    txt_7 = 'total_ref_number : {}'.format(total_ref_loss)
    txt_8 = 'clean_ref_number : {}'.format(clean_ref_loss)
    txt_9 = 'ref_pure_ratio : {}'.format(clean_ref_loss / total_ref_loss)
    logger.info(txt_4)
    logger.info(txt_5)
    logger.info(txt_6)
    logger.info(txt_7)
    logger.info(txt_8)
    logger.info(txt_9)
        
    return indexes_dict, reference_indexes, ratio_criteria

# indexes_dict : {'class_k_small':[class_k_small_index], 'class_k_large':[class_k_large_index]}
# reference_indexes : [[class1 small_loss_index], [class2 small_loss_index], ...]


def mean_vector_per_class(model, data_loader, reference_indexes, clean_or_not, indexes_dict, logger):
    
    # reference image dictionary
    ref_image_dict = {}
    for images, labels, indexes in data_loader['train_dataset']:
        ind = indexes.cpu().numpy()
        
        for i in range(len(reference_indexes)):
            for j in range(len(ind)):
                if ind[j] in reference_indexes[i]:
                    ref_image_dict[ind[j]] = images[j]
    
    # mean vector per class : reference feature vector for re-labeling
    mean_vector_list = []
    
    total_small_loss = 0
    clean_small_loss = 0
    total_ref_loss = 0
    clean_ref_loss = 0
    
    model.eval()
    with torch.no_grad():
        for i in range(len(reference_indexes)):
            ref_image = [ref_image_dict[ind] for ind in reference_indexes[i]]
            
            total_small_loss += len(indexes_dict['class_'+str(i)+'_small'])
            clean_small_loss += np.sum(clean_or_not[indexes_dict['class_'+str(i)+'_small']])
            total_ref_loss += len(reference_indexes[i])
            clean_ref_loss += np.sum(clean_or_not[reference_indexes[i]])
            
            temp_class_pure_ratio = np.sum(clean_or_not[indexes_dict['class_'+str(i)+'_small']]) / float(len(indexes_dict['class_'+str(i)+'_small']))
            temp_ref_pure_ratio = np.sum(clean_or_not[reference_indexes[i]]) / float(len(reference_indexes[i]))
            
            txt_1 = 'class_{}_num_reference : {}'.format(str(i), len(reference_indexes[i]))
            txt_2 = 'class_{}_small_loss_pure_ratio : {:.5f}'.format(str(i), temp_class_pure_ratio)
            txt_3 = 'class_{}_reference_pure_ratio : {:.5f}'.format(str(i), temp_ref_pure_ratio)
            logger.info(txt_1)
            logger.info(txt_2)
            logger.info(txt_3)
            
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
    txt_5 = 'clean_small_loss_number : {}'.format(clean_small_loss)
    txt_6 = 'small_loss_pure_ratio : {}'.format(clean_small_loss / total_small_loss)
    txt_7 = 'total_ref_number : {}'.format(total_ref_loss)
    txt_8 = 'clean_ref_number : {}'.format(clean_ref_loss)
    txt_9 = 'ref_pure_ratio : {}'.format(clean_ref_loss / total_ref_loss)
    logger.info(txt_4)
    logger.info(txt_5)
    logger.info(txt_6)
    logger.info(txt_7)
    logger.info(txt_8)
    logger.info(txt_9)
    
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

def relabel_critetion(model, data_loader, indexes_dict, mean_vector_list, total_true_label, relabel_threshold, logger):
    
    class_num = len(np.unique(np.array(total_true_label)))
    
    large_loss_image = {}
    total_large_index = []
    
    for k,v in indexes_dict.items():
        if 'large' in k:
            total_large_index.append(v.tolist())
    total_large_index = flatten(total_large_index)
            
    for images, labels, indexes in data_loader['train_dataset']:
        ind = indexes.cpu().numpy()
        for i in range(len(ind)):
            if ind[i] in total_large_index:
                large_loss_image[ind[i]] = images[i]
        
    
    # relabel_info : [[index, original_label, re_label, ture_label], ...]
    relabel_info = []
    
    # re-labeling using total large loss samples
    with torch.no_grad():
        for i in range(class_num):
            print('re-labeling is proceeding for large loss samples in class {}'.format(i))
            key_name = 'class_' + str(i) + '_large'
            txt_1 = 'number of large loss data in original class : {}'.format(len(indexes_dict[key_name]))
            logger.info(txt_1)
            
            large_loss_ind = [ind for ind in indexes_dict[key_name]]
            large_loss_true_label = [total_true_label[ind] for ind in indexes_dict[key_name]]
            large_loss_image_dataset = [large_loss_image[ind] for ind in indexes_dict[key_name]]
            
            for j in range(len(large_loss_image_dataset)):
                temp_index = large_loss_ind[j]
                image = large_loss_image_dataset[j]
                image = torch.unsqueeze(image, 0).cuda()
                _, feature_vector = model(image)
                feature_vector = feature_vector.reshape(-1)
                
                eu_similarity_list = [similarity_euclidean(feature_vector, mean_vector) for mean_vector in mean_vector_list]
                eu_softmax = F.softmax(torch.Tensor(eu_similarity_list), dim=0)
                max_value, candidate_class = torch.max(eu_softmax, 0)
                max_value = max_value.item()
                
                # if max_value >= 0.90:
                if max_value >= relabel_threshold:
                    
                    temp_true_label = large_loss_true_label[j]
                    relabel_info.append([temp_index, i, candidate_class, temp_true_label])

    return relabel_info
        

def relabeling_dataset(data_loader, relabel_info, clean_or_not):
    
    original_dataset = data_loader['train_dataset'].dataset
    batch_size = data_loader['train_dataset'].batch_size
    num_workers = data_loader['train_dataset'].num_workers
    pin_memory = data_loader['train_dataset'].pin_memory
    target_label = data_loader['train_dataset'].dataset.train_targets
    
    for i in range(len(relabel_info)):
        temp_index = relabel_info[i][0]
        assert data_loader['train_dataset'].dataset.train_targets[temp_index] == relabel_info[i][1]
        
        temp_relabeled = relabel_info[i][2].item()
        temp_relabeled = np.int64(temp_relabeled)
        target_label[temp_index] = temp_relabeled
        
        if temp_relabeled == relabel_info[i][-1]:
            clean_or_not[temp_index] = True
        else:
            clean_or_not[temp_index] = False
    
    original_dataset.train_targets = target_label
    relabeled_dataset = original_dataset
    data_loader['train_dataset'] = torch.utils.data.DataLoader(dataset = relabeled_dataset,
                                                               batch_size = batch_size,
                                                               shuffle = False,
                                                               pin_memory = pin_memory,
                                                               num_workers = num_workers)
    
    data_loaders = {}
    data_loaders['train_dataset'] = data_loader['train_dataset']
    data_loaders['valid_dataset'] = data_loader['valid_dataset']
    data_loaders['test_dataset'] = data_loader['test_dataset']
        
    print('Successfully Constructed re-labeled dataset!')
        
    return data_loaders, clean_or_not
        

# relabel_info : [[index, original_label, re_label, ture_label], ...]
def relabel_acc(relabel_info, num_class_list, ratio_clean_list, logger):
    
    num_classes = len(num_class_list)
    
    num_relabeled = [0 for i in range(num_classes)]
    num_clean = [0 for i in range(num_classes)]
    num_relabeled_total = [0 for i in range(num_classes)]
    num_total_clean = [0 for i in range(num_classes)]

    for i in range(num_classes):
        temp_num = 0
        temp_clean = 0
        temp_diff_num = 0
        temp_diff_clean = 0
        for j in range(len(relabel_info)):
            if relabel_info[j][2] == i:
                temp_num += 1
                if relabel_info[j][-1] == i:
                    temp_clean += 1
                    
            if relabel_info[j][2] == i and relabel_info[j][1] != i:
                temp_diff_num += 1
                if relabel_info[j][-1] == i:
                    temp_diff_clean += 1
                    
        num_relabeled[i] = temp_num
        num_clean[i] = temp_clean
        num_relabeled_total[i] = temp_diff_num
        num_total_clean[i] = temp_diff_clean
        
        if temp_num == 0:
            temp_num = 1
        if temp_diff_num == 0:
            temp_diff_num = 1
        
        txt_1 = 'Re-labeled class : {}, Re-labeling precision : {:.4f}'.format(i, temp_diff_clean/temp_diff_num)
        txt_2 = 'the number of re-labeled data as {} from different class : {}'.format(i, temp_diff_num)
        txt_3 = 'the number of correctly re-labeled data from different class : {}'.format(temp_diff_clean)
        logger.info(txt_1)
        logger.info(txt_2)
        logger.info(txt_3)
    
    numerator = sum(num_total_clean)
    denominator = sum(num_relabeled_total)
    if denominator == 0:
        denominator = 1
    stage_relabel_acc = numerator / denominator
    txt_8 = 'total re-labeling accuracy(only from diffrent class) : {:.3f}[{}/{}]'.format(stage_relabel_acc, numerator, denominator)
    logger.info(txt_8)
    
    for i in range(num_classes):
        temp_relabeled_data = 0
        for j in range(len(relabel_info)):
            if relabel_info[j][1] == 1:
                temp_relabeled_data += 1
        temp_relabeled_ratio = temp_relabeled_data / num_class_list[i]
        txt_6 = 'noise ratio in class {} : {:.5f}'.format(i, 1-ratio_clean_list[i])
        txt_7 = 'Re-labeled data ratio in class {} : {:.5f}'.format(i, temp_relabeled_ratio)
        logger.info(txt_6)
        logger.info(txt_7)