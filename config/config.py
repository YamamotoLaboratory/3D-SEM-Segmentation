import os, sys, configparser, errno
import tensorflow as tf

def get_config(logger, path):

    config = configparser.ConfigParser()

    config_path = path
    
    if not os.path.exists(config_path):
        logger.error('ファイルの読み込みにエラーが発生しました。{} のファイルが存在しません。'.format(config_path))
        sys.exit('プログラムを終了します。')

    config.read(config_path, encoding='utf-8')
    logger.info('{} の読み込みが完了しました。'.format(config_path))

    return config

def gpu_setting(logger, limit=3):
    g = len(tf.config.experimental.list_physical_devices('GPU'))
    gpus = tf.config.experimental.list_physical_devices('GPU')
    message = ""
    if gpus:
        try:
            tf.config.experimental.set_virtual_device_configuration(
                gpus[0],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*limit)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            message = 'Physical GPUs {}, Logical GPUs {}'.format(len(gpus), len(logical_gpus))
            # logger.info(message)
        except RuntimeError as e:
            logger.exception('Error')
    
        logger.info("Num GPUs Available: {}, MEMORY LIMIT: {}MiB".format(g, str(1024*limit)))
        logger.info(message)
