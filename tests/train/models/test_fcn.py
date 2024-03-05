import warnings
from logging import INFO

warnings.filterwarnings('ignore')

import os
import re
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

sys.dont_write_bytecode = True
sys.path.append('{}'.format(os.getcwd()))

from config import get_config
from log import load_logging_config
from train.models import FCN


class TestFCN:

    def setup_method(self,):
        self.logger = load_logging_config('debug_logger')
        self.config = get_config(self.logger, '{}/config/train/config.ini'.format(os.getcwd()))

    def test_fcn_32s_vgg(self, caplog):
        model_config = self.config['fcn']

        build_model = FCN(
            [int(model_config['size']), int(model_config['size'])],
            int(model_config['input_channel']),
            int(model_config['class_num']),
            self.logger,
            up = re.findall(r'fcn(.+)', 'fcn32s')[0],
            backbone = model_config['backbone']
        )

        model = build_model.get_model()
        config = model.get_config()

        assert config['layers'][0]['config']['batch_input_shape'] == (None, int(model_config['size']), int(model_config['size']), int(model_config['input_channel']))
        assert ("debug_logger", INFO, "Model: FCN-32s, Input Size: 256x256, Backbone: VGG-16") in caplog.record_tuples
        assert ("debug_logger", INFO, "FCN-32s作成完了") in caplog.record_tuples
    
    def test_fcn_16s_vgg(self, caplog):
        model_config = self.config['fcn']

        build_model = FCN(
            [int(model_config['size']), int(model_config['size'])],
            int(model_config['input_channel']),
            int(model_config['class_num']),
            self.logger,
            up = re.findall(r'fcn(.+)', 'fcn16s')[0],
            backbone = model_config['backbone']
        )

        model = build_model.get_model()
        config = model.get_config()

        assert config['layers'][0]['config']['batch_input_shape'] == (None, int(model_config['size']), int(model_config['size']), int(model_config['input_channel']))
        assert ("debug_logger", INFO, "Model: FCN-16s, Input Size: 256x256, Backbone: VGG-16") in caplog.record_tuples
        assert ("debug_logger", INFO, "FCN-16s作成完了") in caplog.record_tuples

    def test_fcn_8s_vgg(self, caplog):
        model_config = self.config['fcn']

        build_model = FCN(
            [int(model_config['size']), int(model_config['size'])],
            int(model_config['input_channel']),
            int(model_config['class_num']),
            self.logger,
            up = re.findall(r'fcn(.+)', 'fcn8s')[0],
            backbone = model_config['backbone']
        )

        model = build_model.get_model()
        config = model.get_config()

        assert config['layers'][0]['config']['batch_input_shape'] == (None, int(model_config['size']), int(model_config['size']), int(model_config['input_channel']))
        assert ("debug_logger", INFO, "Model: FCN-8s, Input Size: 256x256, Backbone: VGG-16") in caplog.record_tuples
        assert ("debug_logger", INFO, "FCN-8s作成完了") in caplog.record_tuples

    def test_fcn_8s_alexnet(self, caplog):
        model_config = self.config['fcn']

        build_model = FCN(
            [int(model_config['size']), int(model_config['size'])],
            int(model_config['input_channel']),
            int(model_config['class_num']),
            self.logger,
            up = re.findall(r'fcn(.+)', 'fcn8s')[0],
            backbone = 'AlexNet'
        )

        model = build_model.get_model()
        config = model.get_config()

        assert config['layers'][0]['config']['batch_input_shape'] == (None, int(model_config['size']), int(model_config['size']), int(model_config['input_channel']))
        assert ("debug_logger", INFO, "Model: FCN-8s, Input Size: 256x256, Backbone: AlexNet") in caplog.record_tuples
        assert ("debug_logger", INFO, "FCN-8s作成完了") in caplog.record_tuples

    def test_fcn_8s_resnet50(self, caplog):
        model_config = self.config['fcn']

        build_model = FCN(
            [int(model_config['size']), int(model_config['size'])],
            int(model_config['input_channel']),
            int(model_config['class_num']),
            self.logger,
            up = re.findall(r'fcn(.+)', 'fcn8s')[0],
            backbone = 'Resnet50'
        )

        model = build_model.get_model()
        config = model.get_config()

        assert config['layers'][0]['config']['batch_input_shape'] == (None, int(model_config['size']), int(model_config['size']), int(model_config['input_channel']))
        assert ("debug_logger", INFO, "Model: FCN-8s, Input Size: 256x256, Backbone: Resnet50") in caplog.record_tuples
        assert ("debug_logger", INFO, "FCN-8s作成完了") in caplog.record_tuples