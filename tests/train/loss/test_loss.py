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

from train.loss import dice_loss, bce_dice_loss, cross_entropy

class TestLoss:

    def setup_method(self,):
        self.logger = load_logging_config('debug_logger')

    def test_dice_loss(self):
        y_true = []
        y_pred = []
        dice_loss(y_true, y_pred, smooth = 1)

    def test_bce_dice_loss(self):
        y_true = []
        y_pred = []
        bce_dice_loss(y_true, y_pred)

    def test_cross_entropy(self):
        y_true = []
        y_pred = []
        cross_entropy(y_true, y_pred)