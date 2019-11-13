import logging
import time
import os

def config_logger(log_prefix):
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO)
    logger = logging.getLogger(__name__)
    rq = time.strftime('%Y%m%d%H%M', time.localtime(time.time()))
    log_path = os.getcwd() + '/logs/' + log_prefix + '/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_name = log_path + rq + '.log'
    file_handler = logging.FileHandler(log_name, mode='w')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'))
    logger.addHandler(file_handler)
    return logger