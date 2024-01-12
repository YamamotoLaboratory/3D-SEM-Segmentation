from logging import DEBUG, ERROR, INFO, WARNING
import warnings
import pytest
warnings.filterwarnings('ignore')

import sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

sys.dont_write_bytecode = True
sys.path.append('{}'.format(os.getcwd()))

from config import get_config, gpu_setting
from log import load_logging_config

# 
class TestConfig:
    
    def setup_method(self,):
        self.logger = load_logging_config('debug_logger')

    def test_config_load(self, caplog):
        config = get_config(self.logger, '{}/config/train/config.ini'.format(os.getcwd()))

        assert len(caplog.records) == 1
        assert ("debug_logger", INFO, "config.ini の読み込みが完了しました。") in caplog.record_tuples
    
    def test_config_load_no_file(self, caplog):

        try:
            config = get_config(self.logger, '{}/config/train/no-file'.format(os.getcwd()))
        except SystemExit as e:
            assert e.code == 'プログラムを終了します。'

        assert len(caplog.records) == 1
        assert ("debug_logger", ERROR, "ファイルの読み込みにエラーが発生しました。no-file のファイルが存在しません。") in caplog.record_tuples

    def test_config_train_content(self, caplog):

        config = get_config(self.logger, '{}/config/train/config.ini'.format(os.getcwd()))

        assert config['DEFAULT']['gpu_memory_limit'] == '3'
        assert config['DATASET']['size'] == '256'
        assert config['PARAMETERS']['loss'] == 'BCE DICE LOSS'
        assert config['OPTIMIZER']['opt'] == 'Adam'
        assert config['unet']['size'] == '256'
        assert config['fcn']['size'] == '256'
        assert config['RESULT']['train_filepath'] == './results/train.csv'

