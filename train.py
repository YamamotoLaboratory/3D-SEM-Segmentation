import os, sys, argparse, json, datetime, psutil, re
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

import tensorflow as tf
import numpy as np

from rich.progress import track

sys.dont_write_bytecode = True

from config import get_config, gpu_setting
from log import load_logging_config
from models import U_Net, FCN
from loss import LossValue
from metrics import MetricsValue
from optimizers import Optimizer, StepLR 

@tf.function
def train_step(images, masks):
    
    with tf.GradientTape() as unet_type:
        logits = model(images, training = True)
        for k, v in train_loss_values.items():v.calc(masks, logits)
        for k, v in train_metrics_values.items():v.calc(masks, logits)

    gradients_of_model = unet_type.gradient(train_loss_values[config.get('PARAMETERS', 'loss')].value, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients_of_model, model.trainable_variables))

    for k, v in train_loss_values.items():v.update()
    for k, v in train_metrics_values.items():v.update()

def train(epochs, img_dataset, seg_dataset, opt_lr):

    min_loss = 100

    for epoch in track(range(epochs), description="Processing..."):
        
        opti_lr.set_learning_rate(optimizer, epoch)

        for img_batch, seg_batch in zip(img_dataset, seg_dataset):
            train_step(img_batch, seg_batch)

        basic_data = [
            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            str(epoch+1),
            str(psutil.cpu_percent(interval=0.5)),
            '',
            str(psutil.virtual_memory().percent),
            str(opt_lr.get_lr(epoch))
        ]

        loss_data = [str(v.data.result().numpy()) for k, v in train_loss_values.items()]
        metrics_data = [str(v.data.result().numpy()) for k, v in train_metrics_values.items()]

        write_file(config.get('RESULT', 'train_filepath'), ','.join(basic_data + loss_data + metrics_data))

        if train_loss_values[config.get('PARAMETERS', 'loss')].data.result() < min_loss:
            min_loss = train_loss_values[config.get('PARAMETERS', 'loss')].data.result()
            logger.info('Epoch {0}: loss improved from inf to {1}, saving model to {2}'.format(epoch + 1, min_loss, config.get('RESULT', 'model_filepath') ))
            model.save(config.get('RESULT', 'model_filepath'))
        else:
            logger.info('Epoch {0}: loss did not improve from {1}'.format(epoch +1 , min_loss))

        for k, v in train_loss_values.items():v.data.reset_states()
        for k, v in train_metrics_values.items():v.data.reset_states()

        logger.info('{}回目の学習結果 LOSS: IoU: CPU: MEMORY:'.format(epoch+1))


def write_file(filename, row, mode='a'):
    with open(filename, mode) as f:
        f.write(row+'\n')

def create_detaset(dataset, batch_size, k_size, seed ):
    dataset = dataset.reshape(dataset.shape[0], k_size, k_size, 1).astype('float32')
    train_dataset = tf.data.Dataset.from_tensor_slices(dataset).shuffle(dataset.shape[0], seed).batch(batch_size)
    return train_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='モデルを学習するためのプログラムになります。')
    parser.add_argument('--debug', action='store_true', help='debugモードでプログラムを実行します。')
    parser.add_argument('-m', '--model', help='学習するモデルを選択可能です。', default='unet', choices=['unet', 'fcn32s', 'fcn16s', 'fcn8s'])
    parser.add_argument('--save_png', action='store_true', help='モデル図を画像として保存します。')

    args = parser.parse_args()

    logger = load_logging_config((lambda args: 'debug_logger' if args.debug else __name__)(args))
    config = get_config(logger)
    gpu_setting(logger, int(config.get('DEFAULT', 'gpu_memory_limit')))

    # Optmizer Settings
    opti_config = config['OPTIMIZER']

    opti = Optimizer(opti_config['opt'], float(opti_config['lr']))
    optimizer = opti.get_optimizer()

    opti_lr = StepLR(float(opti_config['lr']), int(opti_config['step_size']), float(opti_config['gamma']))

    # DataSet Loading
    dadtaset_config = config['DATASET']
    img_train, seg_train = np.load(dadtaset_config['train_dataset'])
    img_train = np.array(img_train)
    seg_train = np.array(seg_train)

    train_img_dataset = create_detaset(
        img_train, 
        int(dadtaset_config['batchsize']), 
        int(dadtaset_config['size']), 
        int(dadtaset_config['seed'])
    )

    train_seg_dataset = create_detaset(
        seg_train, 
        int(dadtaset_config['batchsize']), 
        int(dadtaset_config['size']), 
        int(dadtaset_config['seed'])
    )

    del img_train, seg_train

    # define data
    with open(config.get('RESULT', 'csv_file_columns_path')) as f:
        train_data = json.load(f)
    
    train_basic_column = [v['column'] for k, v in train_data.items() if 'column' in v.keys()]
    train_loss_column = [v['column'] for k, v in train_data['losses'].items() if 'column' in v.keys()]
    train_metrics_column = [v['column'] for k, v in train_data['metrics'].items() if 'column' in v.keys()]

    train_loss_values = {v['column']:LossValue(v['function'], v['metrics']) for k, v in train_data['losses'].items() if 'column' in v.keys()}
    train_metrics_values = {v['column']:MetricsValue(v['function'], v['metrics']) for k, v in train_data['metrics'].items() if 'column' in v.keys()}
    
    write_file(config.get('RESULT', 'train_filepath'), ','.join(train_basic_column + train_loss_column + train_metrics_column), mode='w')
    logger.info('{} にデータを保存しました。'.format(config.get('RESULT', 'train_filepath')))

    del train_basic_column,train_loss_column,train_metrics_column

    # Model Settings
    
    build_model = None
    if args.model == 'unet':
        model_config = config[args.model]
        build_model = U_Net(
            [int(model_config['size']), int(model_config['size'])],
            int(model_config['filters']),
            int(model_config['depth']),
            int(model_config['input_channel']),
            int(model_config['class_num']),
            logger
        )
    else:
        model_config = config['fcn']
        build_model = FCN(
            [int(model_config['size']), int(model_config['size'])],
            int(model_config['input_channel']),
            int(model_config['class_num']),
            logger,
            up = re.findall(r'fcn(.+)', args.model)[0],
            backbone = model_config['backbone']
        )

    model = build_model.get_model()

    if args.save_png:
        tf.keras.utils.plot_model(model, show_shapes = True, show_layer_names = True, to_file='./results/{}_model.png'.format(args.model))
        logger.info('./results/{}_model.png に保存しました。'.format(args.model))

    logger.info('Modelの学習を開始します。')

    train(int(config.get('PARAMETERS', 'epochs')), train_img_dataset, train_seg_dataset, opti_lr)

    logger.info('Modelの学習が終了しました。')