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
from base.base_data_loader import BaseDataLoader
from itertools import repeat


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


def multiclass_noisify(y, P, random_state=0):
    """ Flip classes according to transition probability matrix T.
    It expects a number between 0 and the number of classes - 1.
    """

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
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, nosiy_rate=0.0, asym=False, seed=0):
        super(MNISTNoisy, self).__init__(root, transform=transform, target_transform=target_transform, download=download)
        self.targets = self.targets.numpy()
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

            y_train_noisy = multiclass_noisify(self.targets, P=P, random_state=seed)
            actual_noise = (y_train_noisy != self.targets).mean()
            assert actual_noise > 0.0
            print('Actual noise %.2f' % actual_noise)
            self.targets = y_train_noisy

        else:
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
            class_noisy = int(n_noisy / 10)
            noisy_idx = []
            for d in range(10):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=10, current_class=self.targets[i])
            print(len(noisy_idx))

        print("Print noisy label generation statistics:")
        for i in range(10):
            n_noisy = np.sum(np.array(self.targets) == i)
            print("Noisy class %s, has %s samples." % (i, n_noisy))
        return
        
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class cifar10Nosiy(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, nosiy_rate=0.0, asym=False):
        super(cifar10Nosiy, self).__init__(root, transform=transform, target_transform=target_transform, download=True)
        self.download = download
        if asym:
            # automobile < - truck, bird -> airplane, cat <-> dog, deer -> horse
            source_class = [9, 2, 3, 5, 4]
            target_class = [1, 0, 5, 3, 7]
            for s, t in zip(source_class, target_class):
                cls_idx = np.where(np.array(self.targets) == s)[0]
                n_noisy = int(nosiy_rate * cls_idx.shape[0])
                noisy_sample_index = np.random.choice(cls_idx, n_noisy, replace=False)
                for idx in noisy_sample_index:
                    self.targets[idx] = t
            return
        elif nosiy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
            class_noisy = int(n_noisy / 10)
            noisy_idx = []
            for d in range(10):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=10, current_class=self.targets[i])
            print(len(noisy_idx))
            print("Print noisy label generation statistics:")
            for i in range(10):
                n_noisy = np.sum(np.array(self.targets) == i)
                print("Noisy class %s, has %s samples." % (i, n_noisy))
            return

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        
        return img, target, index


class cifar100Nosiy(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=True, nosiy_rate=0.0, asym=False, seed=0):
        super(cifar100Nosiy, self).__init__(root, download=download, transform=transform, target_transform=target_transform)
        self.download = download
        if asym:
            """mistakes are inside the same superclass of 10 classes, e.g. 'fish'
            """
            nb_classes = 100
            P = np.eye(nb_classes)
            n = nosiy_rate
            nb_superclasses = 20
            nb_subclasses = 5

            if n > 0.0:
                for i in np.arange(nb_superclasses):
                    init, end = i * nb_subclasses, (i+1) * nb_subclasses
                    P[init:end, init:end] = build_for_cifar100(nb_subclasses, n)

                    y_train_noisy = multiclass_noisify(np.array(self.targets), P=P, random_state=seed)
                    actual_noise = (y_train_noisy != np.array(self.targets)).mean()
                assert actual_noise > 0.0
                print('Actual noise %.2f' % actual_noise)
                self.targets = y_train_noisy.tolist()
            return
        elif nosiy_rate > 0:
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(100)]
            class_noisy = int(n_noisy / 100)
            noisy_idx = []
            for d in range(100):
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=100, current_class=self.targets[i])
            print(len(noisy_idx))
            print("Print noisy label generation statistics:")
            for i in range(100):
                n_noisy = np.sum(np.array(self.targets) == i)
                print("Noisy class %s, has %s samples." % (i, n_noisy))
            return
        
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
            
        
        return img, target, index


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
                                       nosiy_rate=self.noise_rate)

            test_dataset = datasets.MNIST(root=self.data_path,
                                          train=False,
                                          transform=test_transform,
                                          download=True)

        elif self.dataset_type == 'CIFAR100':
            CIFAR_MEAN = [0.5071, 0.4865, 0.4409]
            CIFAR_STD = [0.2673, 0.2564, 0.2762]

            train_transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(20),
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
                                          seed=self.seed,
                                          nosiy_rate=self.noise_rate)

            test_dataset = datasets.CIFAR100(root=self.data_path,
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
                                         nosiy_rate=self.noise_rate)

            test_dataset = datasets.CIFAR10(root=self.data_path,
                                            train=False,
                                            transform=test_transform,
                                            download=True)
        else:
            raise("Unknown Dataset")

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

        print("Num of train %d" % (len(train_dataset)))
        print("Num of test %d" % (len(test_dataset)))

        return data_loaders


class Clothing1MDataset:
    def __init__(self, path, type='train', transform=None, target_transform=None):
        self.path = path
        if type == 'test':
            flist = os.path.join(path, "annotations/clean_test_key_list.txt")
        elif type == 'valid':
            flist = os.path.join(path, "annotations/clean_val_key_list.txt")
        elif type == 'train':
            flist = os.path.join(path, "annotations/noisy_train_key_list.txt")
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

def fix_seed(seed=777):
    np.random.seed(seed)
    random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    

class Clothing1MDataLoader(BaseDataLoader):
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_batches=0, training=True, num_workers=4, pin_memory=True, seed=8888):

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_batches = num_batches
        self.training = training

        self.transform_train = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),                
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),                     
            ]) 
        self.transform_val = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.6959, 0.6537, 0.6371),(0.3113, 0.3192, 0.3214)),
            ])     

        self.data_dir = data_dir
        self.train_dataset, self.val_dataset = get_clothing1m('../datasets/clothing1m', num_samples=self.num_batches*self.batch_size, train=training,
#         self.train_dataset, self.val_dataset = get_clothing1m(config['data_loader']['args']['data_dir'], cfg_trainer, num_samples=260000, train=training,
                transform_train=self.transform_train, transform_val=self.transform_val, seed=seed)

        super().__init__(self.train_dataset, batch_size, shuffle, validation_split, num_workers, pin_memory,
                         val_dataset = self.val_dataset)    

def inf_loop(data_loader):
    for loader in repeat(data_loader):
        yield from loader

def get_clothing1m(data_path, num_samples=128000, train=True, transform_train=None, transform_val=None, seed=8888):
    
    if train:
        fix_seed(seed)
        train_dataset = Clothing1M_Dataset(data_path, num_samples=num_samples, train=train, transform=transform_train, seed=seed)
        val_dataset = Clothing1M_Dataset(data_path, val=train, transform=transform_val)
        print(f"Train : {len(train_dataset)} Val: {len(val_dataset)}")
        
    else:
        fix_seed(seed)
        train_dataset = []
        val_dataset = Clothing1M_Dataset(data_path, test=(not train), transform=transform_val)
        print(f"Test: {len(val_dataset)}")
        
    return train_dataset, val_dataset

class Clothing1M_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path='data/', num_samples=128000, train=False, val=False, test=False, transform=None, num_class=14, seed=8888):
        
        fix_seed(seed)
        self.path = data_path
        self.transform = transform
        self.train_labels = {}
        self.test_labels = {}
        self.val_labels = {}
        
        self.train = train
        self.val = val
        self.test = test
        
        with open(os.path.join(self.path, "annotations/noisy_label_kv.txt"), 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/'%self.path + entry[0][7:]
                self.train_labels[img_path] = int(entry[1])
        
        with open(os.path.join(self.path, "annotations/clean_label_kv.txt"), 'r') as f:
            lines = f.read().splitlines()
            for l in lines:
                entry = l.split()
                img_path = '%s/'%self.path + entry[0][7:]
                self.test_labels[img_path] = int(entry[1])
                
        if train:
            train_imgs = []
            with open(os.path.join(self.path, "annotations/noisy_train_key_list.txt"), 'r') as f:
                lines = f.read().splitlines()
                for i,l in enumerate(lines):
                    img_path = '%s/'%self.path + l[7:]
                    train_imgs.append((i,img_path))
            self.num_raw_example = len(train_imgs)
            random.shuffle(train_imgs)
            class_num = torch.zeros(num_class)
            self.train_imgs = []
            self.train_labels_ = []
            for id_raw, impath in train_imgs:
                label = self.train_labels[impath]
                if class_num[label] < (num_samples/14) and len(self.train_imgs)<num_samples:
                    self.train_imgs.append((id_raw,impath))
                    self.train_labels_.append(int(label))
                    class_num[label] += 1
            random.shuffle(self.train_imgs)
            self.train_imgs = np.array(self.train_imgs)
            self.train_labels_ = np.array(self.train_labels_)
            
        elif test:
            self.test_imgs = []
            with open(os.path.join(self.path, "annotations/clean_test_key_list.txt"), 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.path + l[7:]
                    self.test_imgs.append(img_path)
                    
        elif val:
            self.val_imgs = []
            with open(os.path.join(self.path, "annotations/clean_val_key_list.txt"), 'r') as f:
                lines = f.read().splitlines()
                for l in lines:
                    img_path = '%s/'%self.path + l[7:]
                    self.val_imgs.append(img_path)
                    
    def __getitem__(self, index):
        if self.train:
            id_raw, img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
        elif self.val:
            img_path = self.val_imgs[index]
            target = self.test_labels[img_path]
        elif self.test:
            img_path = self.test_imgs[index]
            target = self.test_labels[img_path]
        image = Image.open(img_path).convert('RGB')
        if self.train:
            img0 = self.transform(image)
        if self.test or self.val:
            img = self.transform(image)
            return img, target, index, img_path
        else:
            return img0, target, id_raw, img_path
        
    def __len__(self):
        if self.test:
            return len(self.test_imgs)
        if self.val:
            return len(self.val_imgs)
        else:
            return len(self.train_imgs)
        


## animal-10n
def LoadImg(path):
    return Image.open(path).convert('RGB')

class Animal10NDataset:
    def __init__(self, path, type='train', transform=None, target_transform=None):
        self.imgDir = os.path.join(path, 'training') if type=='train' else os.path.join(path, 'testing')
        self.clsList = sorted(os.listdir(self.imgDir))
        self.nbCls = len(self.clsList)
        self.cls2Idx = dict(zip(self.clsList, range(self.nbCls)))
        self.imgPth = []
        self.imgLabel = []
        for cls in self.clsList:
            imgList = sorted(os.listdir(os.path.join(self.imgDir, cls)))
            self.imgPth = self.imgPth + [os.path.join(self.imgDir, cls, img) for img in imgList]
            self.imgLabel = self.imgLabel + [self.cls2Idx[cls] for i in range(len(imgList))]

        self.nbImg = len(self.imgPth)
        self.dataTransform = transform

    def __len__(self):
        return self.nbImg

    def __getitem__(self, index):
        img = LoadImg(self.imgPth[index])
        img = self.dataTransform(img)
        target = self.imgLabel[index]

        return img, target, index, self.imgPth[index]


@mlconfig.register
class Animal10NDatasetLoader:
    def __init__(self, train_batch_size=128, eval_batch_size=256, data_dir='../dataset/Animal10N/raw_image', num_of_workers=4):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.data_path = data_dir
        self.num_of_workers = num_of_workers
        self.data_loaders = self.loadData()

    def getDataLoader(self):
        return self.data_loaders

    def loadData(self):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
         ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
        # print('data_path',self.data_path)
        train_dataset = Animal10NDataset(path=self.data_path,
                                          type='train',
                                          transform=train_transform)

        test_dataset = Animal10NDataset(path=self.data_path,
                                         type='test',
                                         transform=test_transform)
        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=train_dataset,
                                                   batch_size=self.train_batch_size,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=self.num_of_workers)

        data_loaders['valid_dataset'] = DataLoader(dataset=test_dataset,
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.num_of_workers)                                        

        data_loaders['test_dataset'] = DataLoader(dataset=test_dataset,
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.num_of_workers)

        return data_loaders


class WebVisionDataset:
    def __init__(self, path, file_name='webvision_mini_train', transform=None, target_transform=None):
        self.target_list = []
        self.path = path
        self.load_file(os.path.join(path, file_name))
        self.transform = transform
        self.target_transform = target_transform
        return

    def load_file(self, filename):
        f = open(filename, "r")
        for line in f:
            train_file, label = line.split()
            self.target_list.append((train_file, int(label)))
        f.close()
        return

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, index):
        impath, target = self.target_list[index]
        img = Image.open(os.path.join(self.path, impath)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


@mlconfig.register
class WebVisionDatasetLoader:
    def __init__(self, setting='mini', train_batch_size=128, eval_batch_size=256, train_data_path='data/', valid_data_path='data/', num_of_workers=4):
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.train_data_path = train_data_path
        self.valid_data_path = valid_data_path
        self.num_of_workers = num_of_workers
        self.setting = setting
        self.data_loaders = self.loadData()

    def getDataLoader(self):
        return self.data_loaders

    def loadData(self):
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]
        train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ColorJitter(brightness=0.4,
                                                                     contrast=0.4,
                                                                     saturation=0.4,
                                                                     hue=0.2),
                                              transforms.ToTensor(),
                                              transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

        test_transform = transforms.Compose([transforms.Resize(256),
                                             transforms.CenterCrop(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

        if self.setting == 'mini':
            train_dataset = WebVisionDataset(path=self.train_data_path,
                                             file_name='webvision_mini_train.txt',
                                             transform=train_transform)

            test_dataset = ImageNetMini(root=self.valid_data_path,
                                        split='val',
                                        transform=test_transform)

        elif self.setting == 'full':
            train_dataset = WebVisionDataset(path=self.train_data_path,
                                             file_name='train_filelist_google.txt',
                                             transform=train_transform)

            test_dataset = WebVisionDataset(path=self.valid_data_path,
                                            file_name='val_filelist.txt',
                                            transform=test_transform)

        elif self.setting == 'full_imagenet':
            train_dataset = WebVisionDataset(path=self.train_data_path,
                                             file_name='train_filelist_google',
                                             transform=train_transform)

            test_dataset = datasets.ImageNet(root=self.valid_data_path,
                                             split='val',
                                             transform=test_transform)

        else:
            raise(NotImplementedError)

        data_loaders = {}

        print('Training Set Size %d' % (len(train_dataset)))
        print('Test Set Size %d' % (len(test_dataset)))

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

        return data_loaders


class ImageNetMini(datasets.ImageNet):
    def __init__(self, root, split='val', download=False, **kwargs):
        super(ImageNetMini, self).__init__(root, download=download, split=split, **kwargs)
        self.new_targets = []
        self.new_images = []
        for i, (file, cls_id) in enumerate(self.imgs):
            if cls_id <= 49:
                self.new_targets.append(cls_id)
                self.new_images.append((file, cls_id))
                print((file, cls_id))
        self.imgs = self.new_images
        self.targets = self.new_targets
        self.samples = self.imgs
        print(len(self.samples))
        print(len(self.targets))
        return


class NosieImageNet(datasets.ImageNet):
    def __init__(self, root, split='train', seed=999, download=False, target_class_num=200, nosiy_rate=0.4, **kwargs):
        super(NosieImageNet, self).__init__(root, download=download, split=split, **kwargs)
        random.seed(seed)
        np.random.seed(seed)
        self.new_idx = random.sample(list(range(0, 1000)), k=target_class_num)
        print(len(self.new_idx), len(self.imgs))
        self.new_imgs = []
        self.new_targets = []

        for file, cls_id in self.imgs:
            if cls_id in self.new_idx:
                new_idx = self.new_idx.index(cls_id)
                self.new_imgs.append((file, new_idx))
                self.new_targets.append(new_idx)
        self.imgs = self.new_imgs
        self.targets = self.new_targets
        print(min(self.targets), max(self.targets))
        # Noise
        if split == 'train':
            n_samples = len(self.targets)
            n_noisy = int(nosiy_rate * n_samples)
            print("%d Noisy samples" % (n_noisy))
            class_index = [np.where(np.array(self.targets) == i)[0] for i in range(target_class_num)]
            class_noisy = int(n_noisy / target_class_num)
            noisy_idx = []
            for d in range(target_class_num):
                print(len(class_index[d]), d)
                noisy_class_index = np.random.choice(class_index[d], class_noisy, replace=False)
                noisy_idx.extend(noisy_class_index)
                print("Class %d, number of noisy % d" % (d, len(noisy_class_index)))
            for i in noisy_idx:
                self.targets[i] = other_class(n_classes=target_class_num, current_class=self.targets[i])
                (file, old_idx) = self.imgs[i]
                self.imgs[i] = (file, self.targets[i])
            print(len(noisy_idx))
            print("Print noisy label generation statistics:")
            for i in range(target_class_num):
                n_noisy = np.sum(np.array(self.targets) == i)
                print("Noisy class %s, has %s samples." % (i, n_noisy))

        self.samples = self.imgs


class ImageNetDatasetLoader:
    def __init__(self,
                 batchSize=128,
                 eval_batch_size=256,
                 dataPath='data/',
                 seed=999,
                 target_class_num=200,
                 nosiy_rate=0.4,
                 numOfWorkers=4):
        self.batchSize = batchSize
        self.eval_batch_size = eval_batch_size
        self.dataPath = dataPath
        self.numOfWorkers = numOfWorkers
        self.seed = seed
        self.target_class_num = target_class_num
        self.nosiy_rate = nosiy_rate
        self.data_loaders = self.loadData()

    def getDataLoader(self):
        return self.data_loaders

    def loadData(self):
        IMAGENET_MEAN = [0.485, 0.456, 0.406]
        IMAGENET_STD = [0.229, 0.224, 0.225]

        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4,
                                   contrast=0.4,
                                   saturation=0.4,
                                   hue=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

        train_dataset = NosieImageNet(root=self.dataPath,
                                      split='train',
                                      nosiy_rate=self.nosiy_rate,
                                      target_class_num=self.target_class_num,
                                      seed=self.seed,
                                      transform=train_transform,
                                      download=True)

        test_dataset = NosieImageNet(root=self.dataPath,
                                     split='val',
                                     nosiy_rate=self.nosiy_rate,
                                     target_class_num=self.target_class_num,
                                     seed=self.seed,
                                     transform=test_transform,
                                     download=True)

        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=train_dataset,
                                                   batch_size=self.batchSize,
                                                   shuffle=True,
                                                   pin_memory=True,
                                                   num_workers=self.numOfWorkers)

        data_loaders['test_dataset'] = DataLoader(dataset=test_dataset,
                                                  batch_size=self.batchSize,
                                                  shuffle=False,
                                                  pin_memory=True,
                                                  num_workers=self.numOfWorkers)
        return data_loaders


def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for data, _ in tqdm(loader):

        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3])
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


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
