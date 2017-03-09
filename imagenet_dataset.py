import numpy as np
import os
from PIL import Image
from scipy.io import loadmat
import tarfile

from chainer.dataset import download
from chainer.datasets import LabeledImageDataset

from chainercv import utils


root = 'pfnet/chainercv/imagenet'


def _parse_meta_mat(developers_kit_dir):
    meta_mat_fn = os.path.join(
        developers_kit_dir, 'ILSVRC2012_devkit_t12/data',
        'meta.mat')

    synsets = {}
    mat = loadmat(meta_mat_fn)
    mat_synsets = mat['synsets']
    for mat_synset in mat_synsets:
        ilsvrc2012_id = mat_synset[0][0][0][0] - 1  # starting from 0
        synsets[ilsvrc2012_id] = {
            'WNID': mat_synset[0][1][0],
            'words': mat_synset[0][2][0],
            'gloss': mat_synset[0][3][0],
            'num_children': mat_synset[0][4][0][0],
            'children': mat_synset[0][5][0],
            'wordnet_height': mat_synset[0][6][0][0],
            'num_train_images': mat_synset[0][7][0][0]
        }

    return synsets


def _get_imagenet(urls):
    data_root = download.get_dataset_directory(root)
    # this is error prone
    if os.path.exists(os.path.join(data_root, 'train')):
        return data_root

    for key, url in urls.items():
        download_file_path = utils.cached_download(url)

        d = os.path.join(data_root, key)
        if not os.path.exists(d):
            os.makedirs(d)
        ext = os.path.splitext(url)[1]
        utils.extractall(download_file_path, d, ext)

    # train dataset contains tar inside of tar
    train_dir = os.path.join(data_root, 'train')
    for tar_fn in os.listdir(train_dir):
        if tar_fn[-3:] == 'tar':
            with tarfile.TarFile(os.path.join(train_dir, tar_fn), 'r') as t:
                t.extractall(train_dir)

    # parse developers kit
    developers_kit_dir = os.path.join(data_root, 'developers_kit')
    synsets = _parse_meta_mat(developers_kit_dir)
    wnid_to_ilsvrc_id = {
        val['WNID']: key for key, val in synsets.items()}

    # prepare train_pairs.txt
    train_pairs_fn = os.path.join(data_root, 'train_pairs.txt')
    with open(train_pairs_fn, 'w') as f:
        for fn in os.listdir(train_dir):
            synset = fn[:9]
            if synset in wnid_to_ilsvrc_id and fn[-4:] == 'JPEG':
                int_key = wnid_to_ilsvrc_id[synset]  # starting from 0
                f.write('{} {}\n'.format(fn, int_key))

    # prepare val_pairs.txt
    val_pairs_fn = os.path.join(data_root, 'val_pairs.txt')
    val_gt_fn = os.path.join(
        developers_kit_dir, 'ILSVRC2012_devkit_t12/data',
        'ILSVRC2012_validation_ground_truth.txt')
    with open(val_pairs_fn, 'w') as f:
        for i, l in enumerate(open(val_gt_fn)):
            key = int(l)  # starting from 0
            index = i + 1
            fn = 'ILSVRC2012_val_{0:08}.JPEG'.format(index)
            f.write('{} {}\n'.format(fn, key))
    return data_root


class LabeledImageImagenetDataset(LabeledImageDataset):

    def get_example(self, i):
        path, int_label = self._pairs[i]
        full_path = os.path.join(self._root, path)
        img = np.asarray(Image.open(full_path).convert('RGB'))
        img = img[:, :, ::-1]
        img = img.transpose(2, 0, 1)

        label = np.array(int_label, dtype=self._label_dtype)
        return img, label


def get_imagenet(data_dir='auto',
                 urls={'train': '', 'val': '', 'developers_kit': ''}):
    """ImageNet dataset used for `ILSVRC2012`_.

    .. _ILSVRC2012: http://www.image-net.org/challenges/LSVRC/2012/

    If you pass `\'auto\'` as an argument for `base_dir`, this directory
    tries to download from `urls`. If `urls` is `None` in that case, it will
    look up for dataset in the filesystem, but do not download anything.

    Args:
        data_dir (string): Path to the root of the training data. If this is
            :obj:`auto`, this class will automatically download data for you
            under :obj:`$CHAINER_DATASET_ROOT/pfnet/chainer_cv/imagenet`.
        urls (dict): Dict of urls. Keys correspond to type of dataset to
            download and values correspond to urls. Keys should be
            :obj:`(train, val, developers_kit)`.

    """
    if data_dir == 'auto':
        data_dir = _get_imagenet(urls)
    train_dataset = LabeledImageImagenetDataset(
        os.path.join(data_dir, 'train_pairs.txt'),
        os.path.join(data_dir, 'train'))
    val_dataset = LabeledImageImagenetDataset(
        os.path.join(data_dir, 'val_pairs.txt'),
        os.path.join(data_dir, 'val'))
    return train_dataset, val_dataset


if __name__ == '__main__':
    urls = {
        'train': '',
        'val': '',
        'developers_kit': ''
    }
    train, val = get_imagenet()

    import matplotlib.pyplot as plt

    # this part is for converting label to synset
    data_root = _get_imagenet(None)
    developers_kit_dir = os.path.join(data_root, 'developers_kit')
    synsets = _parse_meta_mat(developers_kit_dir)

    for i in range(80, 100):
        img, label = val[i]
        print synsets[np.asscalar(label)]
        plt.imshow(img.transpose(1, 2, 0)[:, :, ::-1])
        plt.show()
