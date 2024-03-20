from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm
from numpy.testing import assert_array_almost_equal
import numpy as np
import os
import torch
import random
import mlconfig
import copy


def fix_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def build_for_cifar100(size, noise):
    """ random flip between two random classes.
    """
    assert(noise >= 0.) and (noise <= 1.)

    P = (1. - noise) * np.eye(size)
    for i in np.arange(size - 1):
        P[i, i+1] = noise

    # adjust last row
    P[size-1, 0] = noise

    assert_array_almost_equal(P.sum(axis=1), 1, 1)
    return P


def train_val_split(base_dataset, num_classes, seed, is_mnist):

    fix_seed(seed)
    # num_classes = 10
    base_dataset = np.array(base_dataset)

    if not is_mnist:
        train_n = int(len(base_dataset)*0.9/num_classes)
    train_idxs = []
    val_idxs = []

    for i in range(num_classes):
        idxs = np.where(base_dataset == i)[0]
        if is_mnist:
            train_n = int(len(idxs)*0.9)
        np.random.shuffle(idxs)
        train_idxs.extend(idxs[:train_n])
        val_idxs.extend(idxs[train_n:])
    np.random.shuffle(train_idxs)
    np.random.shuffle(val_idxs)

    return train_idxs, val_idxs

def multiclass_noisify(y, P, seed, random_state):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """
    fix_seed(seed)
    assert P.shape[0] == P.shape[1]
    assert np.max(y) < P.shape[0]

    # row stochastic matrix
    assert_array_almost_equal(P.sum(axis=1), np.ones(P.shape[1]))
    assert (P >= 0.0).all()

    m = y.shape[0]
    new_y = y.copy()
    flipper = np.random.RandomState(random_state)

    for idx in np.arange(m):
        i = y[idx]
        # draw a vector with only an 1
        flipped = flipper.multinomial(1, P[i, :], 1)[0]
        new_y[idx] = np.where(flipped == 1)[0]

    return new_y


def other_class(n_classes, current_class):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: one random class that != class_ind
    """

    if current_class < 0 or current_class >= n_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_class_list = list(range(n_classes))
    other_class_list.remove(current_class)
    other_class = np.random.choice(other_class_list)
    return other_class


class MNISTNoisy(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, nosiy_rate=0.0, asym=False, seed=0, version='train'):
        super(MNISTNoisy, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.targets = self.targets.numpy()
        self.version = version

        fix_seed(seed)
        train_set_index, valid_set_index = train_val_split(datasets.MNIST(root, train=True).targets, 10, seed, True)

        temp_targets = copy.deepcopy(self.targets)
        temp_targets = np.array(temp_targets)

        if self.version == 'train':
            self.train_imgs = self.data[train_set_index]
            self.train_targets = list(temp_targets[train_set_index])
            self.train_targets = np.array(self.train_targets)
            
        if self.version == 'valid':
            self.valid_imgs = self.data[valid_set_index]
            self.valid_targets = list(temp_targets[valid_set_index])
            self.valid_targets = np.array(self.valid_targets)
            

        if asym:
            P = np.eye(10)
            n = nosiy_rate

            P[7, 7], P[7, 1] = 1. - n, n
            # 2 -> 7
            P[2, 2], P[2, 7] = 1. - n, n

            # 5 <-> 6
            P[5, 5], P[5, 6] = 1. - n, n
            P[6, 6], P[6, 5] = 1. - n, n

            # 3 -> 8
            P[3, 3], P[3, 8] = 1. - n, n

            if self.version == 'train':
                fix_seed(seed)
                y_train_noisy = multiclass_noisify(self.train_targets, P, seed, seed)
                actual_noise = (y_train_noisy != self.train_targets).mean()
                assert actual_noise > 0.0
                self.train_targets = y_train_noisy
            if self.version == 'valid':
                fix_seed(seed)
                y_valid_noisy = multiclass_noisify(self.valid_targets, P, seed, seed)
                actual_noise = (y_valid_noisy != self.valid_targets).mean()
                assert actual_noise > 0.0
                self.valid_targets = y_valid_noisy

        else:
            if self.version == 'train':
                fix_seed(seed)
                n_samples = len(self.train_targets)
                n_noisy = int(nosiy_rate * n_samples)
                
                class_index = [np.where(np.array(self.train_targets) == i)[0] for i in range(10)]
                class_noisy = int(n_noisy / 10)
                noisy_idx = []
                for d in range(10):
                    noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                    noisy_idx.extend(noisy_class_index)
                    print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
                for i in noisy_idx:
                    self.train_targets[i] = other_class(n_classes=10, current_class=self.train_targets[i])
                for i in range(10):
                    n_noisy = np.sum(np.array(self.train_targets) == i)
                    print('Noisy class %s, has %s samples.' % (i, n_noisy))
                    
            if self.version == 'valid':
                fix_seed(seed)
                n_samples = len(self.valid_targets)
                n_noisy = int(nosiy_rate * n_samples)
                
                class_index = [np.where(np.array(self.valid_targets) == i)[0] for i in range(10)]
                class_noisy = int(n_noisy / 10)
                noisy_idx = []
                for d in range(10):
                    noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                    noisy_idx.extend(noisy_class_index)
                    print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
                for i in noisy_idx:
                    self.valid_targets[i] = other_class(n_classes=10, current_class=self.valid_targets[i])
                for i in range(10):
                    n_noisy = np.sum(np.array(self.valid_targets) == i)
                    print('Noisy class %s, has %s samples.' % (i, n_noisy))

        return
        
    
    def __getitem__(self, index):

        if self.version == 'train':
            img, target = self.train_imgs[index], int(self.train_targets[index])
        if self.version == 'valid':
            img, target = self.valid_imgs[index], int(self.valid_targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.version == 'train':
            return len(self.train_imgs)
        else:
            return len(self.valid_imgs)


class FMNISTNoisy(datasets.FashionMNIST):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, nosiy_rate=0.0, asym=False, seed=0, version='train'):
        super(FMNISTNoisy, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.targets = self.targets.numpy()
        self.version = version

        fix_seed(seed)
        train_set_index, valid_set_index = train_val_split(datasets.FashionMNIST(root, train=True).targets, 10, seed, True)

        temp_targets = copy.deepcopy(self.targets)
        temp_targets = np.array(temp_targets)

        if self.version == 'train':
            self.train_imgs = self.data[train_set_index]
            self.train_targets = list(temp_targets[train_set_index])
            self.train_targets = np.array(self.train_targets)
            
            
        if self.version == 'valid':
            self.valid_imgs = self.data[valid_set_index]
            self.valid_targets = list(temp_targets[valid_set_index])
            self.valid_targets = np.array(self.valid_targets)
            
        if asym:
            P = np.eye(10)
            n = nosiy_rate

            # 0 -> 6
            P[0, 0], P[0, 6] = 1. - n, n

            # 2 -> 4
            P[2, 2], P[2, 4] = 1. - n, n

            # 5 <-> 7
            P[5, 5], P[5, 7] = 1. - n, n
            P[7, 7], P[7, 5] = 1. - n, n
            if self.version == 'train':
                fix_seed(seed)
                y_train_noisy = multiclass_noisify(self.train_targets, P, seed, seed)
                actual_noise = (y_train_noisy != self.train_targets).mean()
                assert actual_noise > 0.0
                self.train_targets = y_train_noisy
            if self.version == 'valid':
                fix_seed(seed)
                y_valid_noisy = multiclass_noisify(self.valid_targets, P, seed, seed)
                actual_noise = (y_valid_noisy != self.valid_targets).mean()
                assert actual_noise > 0.0
                self.valid_targets = y_valid_noisy
        else:
            if self.version == 'train':
                fix_seed(seed)
                n_samples = len(self.train_targets)
                n_noisy = int(nosiy_rate * n_samples)
                
                class_index = [np.where(np.array(self.train_targets) == i)[0] for i in range(10)]
                class_noisy = int(n_noisy / 10)
                noisy_idx = []
                for d in range(10):
                    noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                    noisy_idx.extend(noisy_class_index)
                    print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
                for i in noisy_idx:
                    self.train_targets[i] = other_class(n_classes=10, current_class=self.train_targets[i])
                for i in range(10):
                    n_noisy = np.sum(np.array(self.train_targets) == i)
                    print('Noisy class %s, has %s samples.' % (i, n_noisy))                    

            if self.version == 'valid':
                fix_seed(seed)
                n_samples = len(self.valid_targets)
                n_noisy = int(nosiy_rate * n_samples)
                print("[Valid] %d Noisy samples" % (n_noisy))
                class_index = [np.where(np.array(self.valid_targets) == i)[0] for i in range(10)]
                class_noisy = int(n_noisy / 10)
                noisy_idx = []
                for d in range(10):
                    noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                    noisy_idx.extend(noisy_class_index)
                    print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
                for i in noisy_idx:
                    self.valid_targets[i] = other_class(n_classes=10, current_class=self.valid_targets[i])
                for i in range(10):
                    n_noisy = np.sum(np.array(self.valid_targets) == i)
                    print('Noisy class %s, has %s samples.' % (i, n_noisy))
                    
            return
        
    
    def __getitem__(self, index):

        if self.version == 'train':
            img, target = self.train_imgs[index], int(self.train_targets[index])
        if self.version == 'valid':
            img, target = self.valid_imgs[index], int(self.valid_targets[index])


        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        if self.version == 'train':
            return len(self.train_imgs)
        else:
            return len(self.valid_imgs)


class cifar10Nosiy(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, nosiy_rate=0.0, asym=False, seed=888, version='train'):
        super(cifar10Nosiy, self).__init__(root, transform=transform, target_transform=target_transform, download=True)
        self.download = download
        self.version = version

        fix_seed(seed)
        train_set_index, valid_set_index = train_val_split(datasets.CIFAR10(root, train=True).targets, 10, seed, False)
        # print('seed : {}'.format(seed))
        # print('[train index] : {}'.format(train_set_index[:10]))
        # print('[valid index] : {}'.format(valid_set_index[:10]))

        temp_targets = copy.deepcopy(self.targets)
        temp_targets = np.array(temp_targets)

        if self.version == 'train':
            self.train_imgs = self.data[train_set_index]
            self.train_targets = list(temp_targets[train_set_index])
            self.train_targets = np.array(self.train_targets)

        if self.version == 'valid':
            self.valid_imgs = self.data[valid_set_index]
            self.valid_targets = list(temp_targets[valid_set_index])
            self.valid_targets = np.array(self.valid_targets)

        if asym:
            if self.version == 'train':
                fix_seed(seed)
                for i in range(10):
                    indices = np.where(np.array(self.train_targets) == i)[0]
                    np.random.shuffle(indices)
                    for j, idx in enumerate(indices):
                        if j < nosiy_rate * len(indices):
                            if i == 9: # truck -> automobile
                                self.train_targets[idx] = 1
                            elif i == 2: # bird -> airplane
                                self.train_targets[idx] = 0
                            elif i == 3: # cat -> dog
                                self.train_targets[idx] = 5
                            elif i == 5: # dog -> cat
                                self.train_targets[idx] = 3
                            elif i == 4: # deer -> horse
                                self.train_targets[idx] = 7

            if self.version == 'valid':
                fix_seed(seed)
                for i in range(10):
                    indices = np.where(np.array(self.valid_targets) == i)[0]
                    np.random.shuffle(indices)
                    for j, idx in enumerate(indices):
                        if j < nosiy_rate * len(indices):
                            if i == 9: # truck -> automobile
                                self.valid_targets[idx] = 1
                            elif i == 2: # bird -> airplane
                                self.valid_targets[idx] = 0
                            elif i == 3: # cat -> dog
                                self.valid_targets[idx] = 5
                            elif i == 5: # dog -> cat
                                self.valid_targets[idx] = 3
                            elif i == 4: # deer -> horse
                                self.valid_targets[idx] = 7
            return

        elif nosiy_rate > 0:
            if self.version == 'train':
                fix_seed(seed)
                n_samples = len(self.train_targets)
                n_noisy = int(nosiy_rate * n_samples)
                class_index = [np.where(np.array(self.train_targets) == i)[0] for i in range(10)]
                class_noisy = int(n_noisy/10)
                noisy_idx = []
                for d in range(10):
                    noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                    noisy_idx.extend(noisy_class_index)
                    print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
                for i in noisy_idx:
                    self.train_targets[i] = other_class(n_classes=10, current_class=self.train_targets[i])
                for i in range(10):
                    n_noisy = np.sum(np.array(self.train_targets) == i)
                    print('Noisy class %s, has %s samples.' % (i, n_noisy))

                # print('noisy_idx : {}, {}'.format(noisy_idx[:5], noisy_idx[-5:]))
                # print('noisy labels : {}'.format(self.train_targets[:20]))
                # import sys
                # sys.exit()

            if self.version == 'valid':
                fix_seed(seed)
                n_samples = len(self.valid_targets)
                n_noisy = int(nosiy_rate * n_samples)
                class_index = [np.where(np.array(self.valid_targets) == i)[0] for i in range(10)]
                class_noisy = int(n_noisy / 10)
                noisy_idx = []
                for d in range(10):
                    noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                    noisy_idx.extend(noisy_class_index)
                    print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
                for i in noisy_idx:
                    self.valid_targets[i] = other_class(n_classes=10, current_class=self.valid_targets[i])
                for i in range(10):
                    n_noisy = np.sum(np.array(self.valid_targets) == i)
                    print('Noisy class %s, has %s samples.' % (i, n_noisy))
            return

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.version == 'train':
            img, target = self.train_imgs[index], self.train_targets[index]
        if self.version == 'valid':
            img, target = self.valid_imgs[index], self.valid_targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target, index

    def __len__(self):
        if self.version == 'train':
            return len(self.train_imgs)
        else:
            return len(self.valid_imgs)


class cifar100Nosiy(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, nosiy_rate=0.0, asym=False, seed=888, version='train'):
        super(cifar100Nosiy, self).__init__(root, transform=transform, target_transform=target_transform, download=True)
        self.download = download
        self.version = version

        fix_seed(seed)
        train_set_index, valid_set_index = train_val_split(datasets.CIFAR100(root, train=True).targets, 100, seed, False)
        # print('seed : {}'.format(seed))
        # print('[train index] : {}'.format(train_set_index[:10]))
        # print('[valid index] : {}'.format(valid_set_index[:10]))

        temp_targets = copy.deepcopy(self.targets)
        temp_targets = np.array(temp_targets)

        if self.version == 'train':
            self.train_imgs = self.data[train_set_index]
            self.train_targets = list(temp_targets[train_set_index])
            self.train_targets = np.array(self.train_targets)

        if self.version == 'valid':
            self.valid_imgs = self.data[valid_set_index]
            self.valid_targets = list(temp_targets[valid_set_index])
            self.valid_targets = np.array(self.valid_targets)

        if asym:
            if self.version == 'train':
                """mistakes are inside the same superclass of 10 classes, e.g. 'fish'
                """
                fix_seed(seed)
                nb_classes = 100
                P = np.eye(nb_classes)
                n = nosiy_rate
                nb_superclasses = 20
                nb_subclasses = 5

                if n > 0.0:
                    for i in np.arange(nb_superclasses):
                        init, end = i * nb_subclasses, (i+1) * nb_subclasses
                        P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

                        y_train_noisy = multiclass_noisify(np.array(self.train_targets), P, seed, seed)
                        actual_noise = (y_train_noisy != np.array(self.train_targets)).mean()
                    assert actual_noise > 0.0
                    print('Actual noise %.2f' % actual_noise)
                    self.train_targets = y_train_noisy.tolist()

            if self.version == 'valid':
                """mistakes are inside the same superclass of 10 classes, e.g. 'fish'
                """
                fix_seed(seed)
                nb_classes = 100
                P = np.eye(nb_classes)
                n = nosiy_rate
                nb_superclasses = 20
                nb_subclasses = 5

                if n > 0.0:
                    for i in np.arange(nb_superclasses):
                        init, end = i * nb_subclasses, (i+1) * nb_subclasses
                        P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

                        y_valid_noisy = multiclass_noisify(np.array(self.valid_targets), P, seed, seed)
                        actual_noise = (y_valid_noisy != np.array(self.valid_targets)).mean()
                    assert actual_noise > 0.0
                    print('Actual noise %.2f' % actual_noise)
                    self.valid_targets = y_valid_noisy.tolist()

            return

        elif nosiy_rate > 0:
            if self.version == 'train':
                fix_seed(seed)
                n_samples = len(self.train_targets)
                n_noisy = int(nosiy_rate * n_samples)
                class_index = [np.where(np.array(self.train_targets) == i)[0] for i in range(100)]
                class_noisy = int(n_noisy/100)
                noisy_idx = []
                for d in range(100):
                    noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                    noisy_idx.extend(noisy_class_index)
                    print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
                for i in noisy_idx:
                    self.train_targets[i] = other_class(n_classes=100, current_class=self.train_targets[i])
                for i in range(100):
                    n_noisy = np.sum(np.array(self.train_targets) == i)
                    print('Noisy class %s, has %s samples.' % (i, n_noisy))

            if self.version == 'valid':
                fix_seed(seed)
                n_samples = len(self.valid_targets)
                n_noisy = int(nosiy_rate * n_samples)
                class_index = [np.where(np.array(self.valid_targets) == i)[0] for i in range(100)]
                class_noisy = int(n_noisy / 100)
                noisy_idx = []
                for d in range(100):
                    noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                    noisy_idx.extend(noisy_class_index)
                    print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
                for i in noisy_idx:
                    self.valid_targets[i] = other_class(n_classes=100, current_class=self.valid_targets[i])
                for i in range(100):
                    n_noisy = np.sum(np.array(self.valid_targets) == i)
                    print('Noisy class %s, has %s samples.' % (i, n_noisy))
            return

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.version == 'train':
            img, target = self.train_imgs[index], self.train_targets[index]
        if self.version == 'valid':
            img, target = self.valid_imgs[index], self.valid_targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target, index

    def __len__(self):
        if self.version == 'train':
            return len(self.train_imgs)
        else:
            return len(self.valid_imgs)


@mlconfig.register
class DatasetGenerator():
    def __init__(self,
                 train_batch_size=128,
                 eval_batch_size=256,
                 data_path='data/',
                 seed=123,
                 num_of_workers=4,
                 asym=False,
                 dataset_type='CIFAR10',
                 is_cifar100=False,
                 cutout_length=16,
                 noise_rate=0.4):
        self.seed = seed
        np.random.seed(seed)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_path = data_path
        self.num_of_workers = num_of_workers
        self.cutout_length = cutout_length
        self.noise_rate = noise_rate
        self.dataset_type = dataset_type
        self.asym = asym
        self.data_loaders = self.loadData()
        return

    def getDataLoader(self):
        return self.data_loaders

    def loadData(self):
        if self.dataset_type == 'MNIST':
            MEAN = [0.1307]
            STD = [0.3081]
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)])

            train_dataset = MNISTNoisy(root=self.data_path,
                                       train=True,
                                       transform=train_transform,
                                       download=True,
                                       asym=self.asym,
                                       seed=self.seed,
                                       nosiy_rate=self.noise_rate,
                                       version='train')

            valid_dataset = MNISTNoisy(root=self.data_path,
                                       train=True,
                                       transform=train_transform,
                                       download=True,
                                       asym=self.asym,
                                       seed=self.seed,
                                       nosiy_rate=self.noise_rate,
                                       version='valid')

            test_dataset = datasets.MNIST(root=self.data_path,
                                          train=False,
                                          transform=test_transform,
                                          download=True)

        if self.dataset_type == 'FMNIST':
            MEAN = [0.5]
            STD = [0.5]
            train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)])

            train_dataset = FMNISTNoisy(root=self.data_path,
                                       train=True,
                                       transform=train_transform,
                                       download=True,
                                       asym=self.asym,
                                       seed=self.seed,
                                       nosiy_rate=self.noise_rate,
                                       version='train')

            valid_dataset = FMNISTNoisy(root=self.data_path,
                                       train=True,
                                       transform=train_transform,
                                       download=True,
                                       asym=self.asym,
                                       seed=self.seed,
                                       nosiy_rate=self.noise_rate,
                                       version='valid')

            test_dataset = datasets.FashionMNIST(root=self.data_path,
                                                train=False,
                                                transform=test_transform,
                                                download=True)

        elif self.dataset_type == 'CIFAR10':
            CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
            CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            train_dataset = cifar10Nosiy(root=self.data_path,
                                         train=True,
                                         transform=train_transform,
                                         download=True,
                                         asym=self.asym,
                                         nosiy_rate=self.noise_rate,
                                         version='train')

            valid_dataset = cifar10Nosiy(root=self.data_path,
                                         train=True,
                                         transform=train_transform,
                                         download=True,
                                         asym=self.asym,
                                         nosiy_rate=self.noise_rate,
                                         version='valid')

            test_dataset = datasets.CIFAR10(root=self.data_path,
                                            train=False,
                                            transform=test_transform,
                                            download=True)

        elif self.dataset_type == 'CIFAR100':
            CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
            CIFAR_STD = [0.2673, 0.2564, 0.2762]

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

            train_dataset = cifar100Nosiy(root=self.data_path,
                                         train=True,
                                         transform=train_transform,
                                         download=True,
                                         asym=self.asym,
                                         nosiy_rate=self.noise_rate,
                                         version='train')

            valid_dataset = cifar100Nosiy(root=self.data_path,
                                         train=True,
                                         transform=train_transform,
                                         download=True,
                                         asym=self.asym,
                                         nosiy_rate=self.noise_rate,
                                         version='valid')

            test_dataset = datasets.CIFAR100(root=self.data_path,
                                            train=False,
                                            transform=test_transform,
                                            download=True)                                            


        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=train_dataset,
                                                   batch_size=self.train_batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=self.num_of_workers)
        
        data_loaders['valid_dataset'] = DataLoader(dataset=valid_dataset,
                                                   batch_size=self.train_batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=self.num_of_workers)

        data_loaders['test_dataset'] = DataLoader(dataset=test_dataset,
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.num_of_workers)

        print("Num of train %d" % (len(train_dataset)))
        print("Num of valid %d" % (len(valid_dataset)))
        print("Num of test %d" % (len(test_dataset)))

        return data_loaders


class Clothing1MDataset:
    def __init__(self, path, type='train', transform=None, target_transform=None):
        self.path = path
        if type == 'test':
            flist = os.path.join(path, "annotations/clean_test.txt")
        elif type == 'valid':
            flist = os.path.join(path, "annotations/clean_val.txt")
        elif type == 'train':
            flist = os.path.join(path, "annotations/noisy_train.txt")
        else:
            raise('Unknown type')

        self.imlist = self.flist_reader(flist)
        self.transform = transform

    def __len__(self):
        return len(self.imlist)

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = Image.open(impath).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def flist_reader(self, flist):
        imlist = []
        with open(flist, 'r') as rf:
            for line in rf.readlines():
                row = line.split(" ")
                impath = self.path + row[0]
                imlabel = row[1]
                imlist.append((impath, int(imlabel)))
        return imlist


@mlconfig.register
class Clothing1MDatasetLoader:
    def __init__(self, train_batch_size=128, eval_batch_size=256, data_path='data/', num_of_workers=4, use_cutout=True, cutout_length=112):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_path = data_path
        self.num_of_workers = num_of_workers
        self.use_cutout = use_cutout
        self.cutout_length = cutout_length
        self.data_loaders = self.loadData()

    def getDataLoader(self):
        return self.data_loaders

    def loadData(self):
        MEAN = [0.485, 0.456, 0.406]
        STD = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
         ])
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD)
        ])
        if self.use_cutout:
            print('Using Cutout')
            train_transform.transforms.append(Cutout(self.cutout_length))

        train_dataset = Clothing1MDataset(path=self.data_path,
                                          type='train',
                                          transform=train_transform)

        test_dataset = Clothing1MDataset(path=self.data_path,
                                         type='test',
                                         transform=test_transform)

        valid_dataset = Clothing1MDataset(path=self.data_path,
                                          type='valid',
                                          transform=test_transform)

        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=train_dataset,
                                                   batch_size=self.train_batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=self.num_of_workers)

        data_loaders['test_dataset'] = DataLoader(dataset=test_dataset,
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.num_of_workers)

        data_loaders['valid_dataset'] = DataLoader(dataset=valid_dataset,
                                                   batch_size=self.eval_batch_size,
                                                   shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=self.num_of_workers)
        return data_loaders


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img