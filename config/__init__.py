from .config import get_config, gpu_setting

import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

