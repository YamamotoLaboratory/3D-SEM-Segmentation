import warnings
from logging import INFO

warnings.filterwarnings('ignore')

import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

sys.dont_write_bytecode = True
sys.path.append('{}'.format(os.getcwd()))

from config import get_config
from log import load_logging_config
from train.models import Unet


class TestUNet:

    def setup_method(self,):
        self.logger = load_logging_config('debug_logger')
        self.config = get_config(self.logger, '{}/config/train/config.ini'.format(os.getcwd()))

    def test_unet(self, caplog):
        model_config = self.config['unet']

        build_model = Unet(
            [int(model_config['size']), int(model_config['size'])],
            int(model_config['filters']),
            int(model_config['depth']),
            int(model_config['input_channel']),
            int(model_config['class_num']),
            self.logger
        )

        model = build_model.get_model()
        config = model.get_config()

        assert config['layers'][0]['config']['batch_input_shape'] == (None, int(model_config['size']), int(model_config['size']), int(model_config['input_channel']))
        assert ("debug_logger", INFO, "Model: U-Net, Input Size: 256x256") in caplog.record_tuples
        assert ("debug_logger", INFO, "U-Net作成完了") in caplog.record_tuples