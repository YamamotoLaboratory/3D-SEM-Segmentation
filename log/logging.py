from logging import getLogger,config
import yaml

def load_logging_config(name):
    config.dictConfig(yaml.load(open('./log/logging_setting.yaml').read(), Loader=yaml.SafeLoader))
    logger = getLogger(name)
    logger.debug('デバッグモードでプログラムが実行されました。')
    logger.info('ログ設定の読み込みが完了しました。')
    return logger