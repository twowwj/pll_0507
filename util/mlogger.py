import logging
import time, os
import shutil

def create_logger(final_output_dir):

    time_str = time.strftime('%Y-%m-%d-%H:%M')
    log_file = '{}.log'.format(time_str)
    final_log_file = os.path.join(final_output_dir, log_file)
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger

if __name__ == '__main__':
    logger = create_logger("/datadrive/wjwang/FER_0108/")
