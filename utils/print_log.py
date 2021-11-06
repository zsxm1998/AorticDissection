import os
import sys
import time
import logging
from logging import Logger, Formatter, StreamHandler, FileHandler, Filter, handlers

log_dir = 'details/logs/'
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
    print(f'Make log dir: {os.path.join(os.getcwd(), log_dir)}')

class LevelFilter(Filter):
    def __init__(self, name: str='', level: int=logging.INFO) -> None:
        super().__init__(name=name)
        self.level = level

    def filter(self, record):
        if record.levelno < self.level:
            return False
        return True


train_log = logging.getLogger('train_log')  # 这里用getLogger而不用Logger类的构造函数，可以让在其他地方getLogger同名的logger时获取到这个logger，不用重新配置
train_log.setLevel(logging.DEBUG)
train_log.propagate = False # 防止向上传播导致root logger也打印log

stdf = StreamHandler(sys.stdout)
stdf.addFilter(LevelFilter('std_filter', logging.INFO))
stdf.setFormatter(Formatter('[%(levelname)s]: %(message)s'))
train_log.addHandler(stdf)

filef = FileHandler(f'{log_dir}/log_train_{time.strftime("%m-%d_%H:%M:%S", time.localtime())}.txt', 'w')
filef.addFilter(LevelFilter('file_filter', logging.INFO))
filef.setFormatter(Formatter('[%(levelname)s %(asctime)s] %(message)s', "%Y%m%d-%H:%M:%S"))
train_log.addHandler(filef)
