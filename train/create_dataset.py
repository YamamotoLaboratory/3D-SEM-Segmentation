import sys, argparse, os, glob, random, gzip
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
import cv2
import numpy as np

from keras.preprocessing.image import img_to_array

sys.dont_write_bytecode = True
sys.path.append('{}'.format(os.getcwd()))

from config import get_config, gpu_setting
from log import load_logging_config

class ImageAugmenter(object):

    def __init__(self, args, config, logger):

        self.args = args
        self.config = config
        self.logger = logger

        self.k_size = int(config['size'])
        self.mode = 'test' if args.test else 'train'
        self.channels = 1
        self.num = int(config['num'])

        self.logger.info('Mode: {0}, Image Size: {1}x{1}, Number of Images: {2}'.format(self.mode, self.k_size, self.num))

        self.default_data, self.default_seg = self.directory_checker()

        self.save_train_path = config['train_dataset']
        os.makedirs('./dataset/train_dataset', exist_ok=True)
        self.logger.debug('Make directory, {}'.format('./dataset/train_dataset'))
        self.save_test_path = config['test_dataset']
        os.makedirs('./dataset/test_dataset', exist_ok=True)
        self.logger.debug('Make directory, {}'.format('./dataset/test_dataset'))

        self.file_checker()

    def create_dataset(self,):
        
        images_dataset = []
        masks_dataset = []

        images, masks = self.image_formatter()
        images_dataset.append(images)
        masks_dataset.append(masks)
        images_dataset = np.concatenate(images_dataset, 0)
        masks_dataset = np.concatenate(masks_dataset, 0)

        # 読み込み速度の関係よりcompresslevelは6に設定、データ容量としては約1/10に圧縮している。
        with gzip.GzipFile(self.save_test_path if self.args.test else self.save_train_path, "wb", compresslevel=6) as f:
            self.logger.debug('圧縮されたgzipファイルで保存')
            np.save(f, np.array([images_dataset, masks_dataset])/255)

        self.logger.info('データセット作成完了。{}に保存しました・'.format(self.save_test_path if self.args.test else self.save_train_path))
        self.logger.info('サイズは{}になります。'.format(images_dataset.shape))

        filesize = os.path.getsize(self.save_test_path if self.args.test else self.save_train_path)

        self.logger.info('データ容量は約{}MiBになります'.format(round(filesize/1024/1024)))

    def image_transform(self, img, mask):
        seed = tf.random.uniform(shape=[1], minval=0, maxval=2 ** 32)

        tf.random.set_seed(int(seed))
        random_num = tf.random.uniform(shape=[5], minval=0, maxval=2 ** 32)
        img = tf.image.random_crop(
            img, [self.k_size, self.k_size, 1], seed=int(random_num[0])
        )
        img = tf.image.random_flip_left_right(img, seed=int(random_num[1]))
        
        tf.random.set_seed(int(seed))
        random_num = tf.random.uniform(shape=[4], minval=0, maxval=2 ** 32)
        mask = tf.image.random_crop(
            mask, [self.k_size, self.k_size, 1], seed=int(random_num[0])
        )
        mask = tf.image.random_flip_left_right(mask, seed=int(random_num[1]))
        return img, mask
    
    def image_formatter(self, ):
        images = {}
        image_dataset = []
        mask_dataset = []

        for file in glob.glob(self.default_data + "*"):

            self.logger.debug('{}の画像のデータセット対象としてセット'.format(file))

            basename_without_ext = os.path.splitext(os.path.basename(file))[0]
            self.pair_image_checker(basename_without_ext)
            if os.path.isfile(self.default_seg + basename_without_ext + "_2col{}".format(os.path.splitext(os.path.basename(file))[1])):
                images.setdefault(
                    basename_without_ext,
                    tf.stack(
                        [
                            cv2.imread(file, 0),
                            cv2.imread(
                                self.default_seg + basename_without_ext + "_2col.{}".format(self.config['extension']), 0
                            ),
                        ],
                        0,
                    ),
                )

        for basename_without_ext, src in images.items():
            self.logger.debug('{}の画像のデータセット作成開始'.format(basename_without_ext))
            num = self.num
            img = np.array(src[0])
            img = img.reshape(img.shape[0], img.shape[1], 1)
            mask = np.array(src[1])
            mask = mask.reshape(mask.shape[0], mask.shape[1], 1)
            for i in range(num):
                format_img, format_mask = self.image_transform(img, mask)
                image_dataset.append(format_img)
                mask_dataset.append(format_mask)
            self.logger.debug('{}の画像のデータセット作成終了'.format(basename_without_ext))
        return image_dataset, mask_dataset
    
    def directory_checker(self, ):

        img_path = ''
        seg_path = ''

        if self.args.image == None and self.args.seg == None:
            img_path = "./dataset/{}/img/".format(self.mode)
            seg_path = "./dataset/{}/seg/".format(self.mode)
        elif(self.args.image == self.args.seg):
            logger.error('画像ファイルの入ったディレクトリとGT画像ファイルの入ったディレクトリは同じにはできません。')
            sys.exit('プログラムを終了します。')
        else:
            img_path = self.args.image
            seg_path = self.args.seg

        self.logger.info('{}の画像を読み込みます。'.format(img_path))
        self.logger.info('{}の画像を読み込みます。'.format(seg_path))

        return img_path,seg_path


    def file_checker(self,):
        dir_img = os.listdir(self.default_data)
        dir_seg = os.listdir(self.default_seg)

        if len(dir_img) == 0 or len(dir_seg) == 0 or (len(dir_img) != len(dir_seg)):
            logger.error('画像がディレクトリに保存されていません。')
            sys.exit('プログラムを終了します。')
        if os.path.isfile(self.save_train_path):
            logger.error('教師用データセット"{}"がすでにディレクトリに存在しています。'.format(self.save_train_path))
            sys.exit('プログラムを終了します。')
        if os.path.isfile(self.save_test_path):
            logger.error('テスト用データセット"{}"がすでにディレクトリに存在しています。'.format(self.save_train_path))
            sys.exit('プログラムを終了します。')

    def pair_image_checker(self, basename_without_ext):
        if not os.path.exists(self.default_seg + basename_without_ext + "_2col.{}".format(self.config['extension'])):
            logger.warn('ペアとなるGT画像がありません。')
            sys.exit('プログラムを終了します。')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='データセットを作成するためのプログラムになります。')
    parser.add_argument('--debug', action='store_true', help='debugモードでプログラムを実行します。')
    parser.add_argument('--test', action='store_true', help='テスト用のデータセットを作成します。')
    parser.add_argument('-i', '--image', help='画像ファイルの入ったディレクトリの指定が可能です。デフォルトは"./image/<train or test>/img/"')
    parser.add_argument('-s', '--seg', help='GT画像ファイルの入ったディレクトリの指定が可能です。デフォルトは"./image/<train or test>/seg/"')

    args = parser.parse_args()

    logger = load_logging_config((lambda args: 'debug_logger' if args.debug else __name__)(args))
    config = get_config(logger, '{}/config/train/config.ini'.format(os.getcwd()))
    gpu_setting(logger, int(config.get('DEFAULT', 'gpu_memory_limit')))

    dataset_creater = ImageAugmenter(args, config['DATASET'], logger)
    dataset_creater.create_dataset()