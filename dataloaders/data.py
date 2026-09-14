import os
import yaml
import pickle
import numpy as np

from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from sklearn.model_selection import train_test_split

from shutil import move, rmtree


class iData(object):
    common_trsf = []
    train_trsf = []
    test_trsf = []
    class_order = None

# 编写的一个基础数据类，包含了数据的下载和预处理等功能，
# 其他数据集类可以继承这个基础类并重写相应的方法来实现特定数据集的处理逻辑。
if __name__ == "__main__":
    idata = iData()
    print(idata.common_trsf)

class iCIFAR10(iData):
    use_path = False  # whether to use custom path or not
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
        ),
    ]

    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip()
    ]

    test_trsf = [
        transforms.Resize(224)
    ]
    
    class_order = np.arange(10).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(10).tolist()
        self.class_order = class_order

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR10(self.args['data_path'], train=True, download=True)
        test_dataset = datasets.cifar.CIFAR10(self.args['data_path'], train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iCIFAR100(iData):
    use_path = False  # whether to use custom path or not
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
        ),
    ]

    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]

    test_trsf = [
        transforms.Resize(224),
    ]
    
    # result = [i for i in range(100)]，还是采用上面的方法吧，直接生成一个列表，导入numpy更加复杂
    class_order = np.arange(100).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(100).tolist()
        self.class_order = class_order

    def download_data(self):
        train_dataset = datasets.cifar.CIFAR100(self.args['data_path'], train=True, download=True)
        test_dataset = datasets.cifar.CIFAR100(self.args['data_path'], train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(
            train_dataset.targets
        )
        self.test_data, self.test_targets = test_dataset.data, np.array(
            test_dataset.targets
        )


class iIMAGENET_R(iData):
    use_path = True
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)
        ),
    ]

    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip()
    ]

    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224)
    ]
    
    class_order = np.arange(200).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(200).tolist()
        self.class_order = class_order

    def download_data(self):
        # load splits from config file
        if not os.path.exists(os.path.join(self.args['data_path'], 'train')) and not os.path.exists(os.path.join(self.args['data_path'], 'test')):
            self.dataset = datasets.ImageFolder(self.args['data_path'], transform=None)
            train_idx, val_idx = split_train_test_idx(self.dataset)
            self.train_file_list = [self.dataset.imgs[i][0] for i in train_idx]
            self.test_file_list = [self.dataset.imgs[i][0] for i in val_idx]
            split_train_test_path(self.args['data_path'], self.dataset.classes, self.train_file_list, self.test_file_list)
            print("Dataset split completed.")

        train_data_config = datasets.ImageFolder(os.path.join(self.args['data_path'], 'train')).samples
        test_data_config = datasets.ImageFolder(os.path.join(self.args['data_path'], 'test')).samples
        self.train_data = np.array([config[0] for config in train_data_config])
        self.train_targets = np.array([config[1] for config in train_data_config])
        self.test_data = np.array([config[0] for config in test_data_config])
        self.test_targets = np.array([config[1] for config in test_data_config])


class iIMAGENET_A(iData):
    use_path = True
    common_trsf = [
        transforms.ToTensor()
    ]

    train_trsf = [
            transforms.RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3./4., 4./3.)),
            transforms.RandomHorizontalFlip(p=0.5)
    ]

    test_trsf = [
        # transforms.Resize(256, interpolation=3),
        # warning fix
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224)
    ]
    
    class_order = np.arange(200).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(200).tolist()
        self.class_order = class_order

    def download_data(self):
        train_dir = os.path.join(self.args['data_path'], 'train')
        test_dir = os.path.join(self.args['data_path'], 'test')

        if not os.path.exists(train_dir) and not os.path.exists(test_dir):
            self.dataset = datasets.ImageFolder(self.args['data_path'], transform=None)
            train_idx, val_idx = split_train_test_idx(self.dataset)
            self.train_file_list = [self.dataset.imgs[i][0] for i in train_idx]
            self.test_file_list = [self.dataset.imgs[i][0] for i in val_idx]
            split_train_test_path(self.args['data_path'], self.dataset.classes, self.train_file_list, self.test_file_list)
            print("Dataset split completed.")

        train_data_config = datasets.ImageFolder(train_dir).samples
        test_data_config = datasets.ImageFolder(test_dir).samples
        self.train_data = np.array([config[0] for config in train_data_config])
        self.train_targets = np.array([config[1] for config in train_data_config])
        self.test_data = np.array([config[0] for config in test_data_config])
        self.test_targets = np.array([config[1] for config in test_data_config])


class iDomainNet(iData):
    use_path = True
    common_trsf = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    ]

    train_trsf = [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
    ]

    test_trsf = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]

    class_order = np.arange(345).tolist()
    
    def __init__(self, args):
        self.args = args
        class_order = np.arange(345).tolist()
        self.class_order = class_order
        self.domain_names = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch", ]

    def download_data(self):
        # load splits from config file
        train_pkl_path = 'dataloaders/splits/domainnet_train.pkl'
        train_yaml_path = 'dataloaders/splits/domainnet_train.yaml'
        if os.path.exists(train_pkl_path):
            with open(train_pkl_path, 'rb') as f:
                train_data_config = pickle.load(f)
        else:
            with open(train_yaml_path, 'r') as f:
                train_data_config = yaml.load(f, Loader=yaml.Loader)
            with open(train_pkl_path, 'wb') as f:
                pickle.dump(train_data_config, f)
        
        test_pkl_path = 'dataloaders/splits/domainnet_test.pkl'
        test_yaml_path = 'dataloaders/splits/domainnet_test.yaml'
        if os.path.exists(test_pkl_path):
            with open(test_pkl_path, 'rb') as f:
                test_data_config = pickle.load(f)
        else:
            with open(test_yaml_path, 'r') as f:
                test_data_config = yaml.load(f, Loader=yaml.Loader)
            with open(test_pkl_path, 'wb') as f:
                pickle.dump(test_data_config, f)

        self.train_data = np.array(train_data_config['data'])
        self.train_targets = np.array(train_data_config['targets'])
        self.test_data = np.array(test_data_config['data'])
        self.test_targets = np.array(test_data_config['targets'])


class iCUB(iData):
    use_path = True  # whether to use custom path or not
    common_trsf = [transforms.ToTensor()]

    train_trsf = [
            transforms.RandomResizedCrop(224, scale=(0.05, 1.0), ratio=(3./4., 4./3.)),
            transforms.RandomHorizontalFlip(p=0.5)
    ]
    test_trsf = [
        # transforms.Resize(256, interpolation=3), 
        # warning fix
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(224)
    ]
    
    class_order = np.arange(200).tolist()

    def __init__(self, args):
        self.args = args
        class_order = np.arange(200).tolist()
        self.class_order = class_order

    def download_data(self):
        train_dir = os.path.join(self.args['data_path'], 'train')
        test_dir = os.path.join(self.args['data_path'], 'test')

        if not os.path.exists(train_dir) and not os.path.exists(test_dir):
            self.dataset = datasets.ImageFolder(self.args['data_path'], transform=None)
            train_idx, val_idx = split_train_test_idx(self.dataset)
            self.train_file_list = [self.dataset.imgs[i][0] for i in train_idx]
            self.test_file_list = [self.dataset.imgs[i][0] for i in val_idx]
            split_train_test_path(self.args['data_path'], self.dataset.classes, self.train_file_list, self.test_file_list)
            print("Dataset split completed.")
            
        train_data_config = datasets.ImageFolder(train_dir).samples
        test_data_config = datasets.ImageFolder(test_dir).samples
        self.train_data = np.array([config[0] for config in train_data_config])
        self.train_targets = np.array([config[1] for config in train_data_config])
        self.test_data = np.array([config[0] for config in test_data_config])
        self.test_targets = np.array([config[1] for config in test_data_config])


def build_transform(is_train, args):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        
        transform = [
            transforms.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
        return transform

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(
            # transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
            # warning fix
            transforms.Resize(size, interpolation=InterpolationMode.BICUBIC),
        )
        t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    
    # return transforms.Compose(t)
    return t


def split_train_test_idx(dataset, test_size=0.2, seed=42):
    labels = [dataset[i][1] for i in range(len(dataset))]
    train_idx, val_idx = train_test_split(
        range(len(dataset)), 
        test_size=test_size, 
        stratify=labels, 
        random_state=seed
    )
    return train_idx, val_idx


def split_train_test_path(data_path, classes, train_list, test_list):
    train_folder = os.path.join(data_path, 'train')
    test_folder = os.path.join(data_path, 'test')

    if os.path.exists(train_folder):
        rmtree(train_folder)
    if os.path.exists(test_folder):
        rmtree(test_folder)
    os.mkdir(train_folder)
    os.mkdir(test_folder)

    for c in classes:
        if not os.path.exists(os.path.join(train_folder, c)):
            os.mkdir(os.path.join(os.path.join(train_folder, c)))
        if not os.path.exists(os.path.join(test_folder, c)):
            os.mkdir(os.path.join(os.path.join(test_folder, c)))
    
    for path in train_list:
        if '\\' in path:
            path = path.replace('\\', '/')
        src = path
        dst = os.path.join(train_folder, '/'.join(path.split('/')[-2:]))
        move(src, dst)

    for path in test_list:
        if '\\' in path:
            path = path.replace('\\', '/')
        src = path
        dst = os.path.join(test_folder, '/'.join(path.split('/')[-2:]))
        move(src, dst)
    
    for c in classes:
        path = os.path.join(data_path, c)
        rmtree(path)
