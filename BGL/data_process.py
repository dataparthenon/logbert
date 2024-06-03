import sys
sys.path.append('../')
import os
import gc
import pandas as pd
import numpy as np
from logparser import Spell, Drain
from tqdm import tqdm
from logdeep.dataset.session import sliding_window

tqdm.pandas()
pd.options.mode.chained_assignment = None

PAD = 0
UNK = 1
START = 2

data_dir = os.path.expanduser("")
output_dir = "../output/bgl/"
log_file = "BGL.log"


def count_anomaly():
    total_size = 0
    normal_size = 0
    with open(data_dir + log_file, encoding="utf8") as f:
        for line in f:
            total_size += 1
            if line.split(' ',1)[0] == '-':
                normal_size += 1
    print("total size {}, abnormal size {}".format(total_size, total_size - normal_size))


def deeplog_file_generator(filename, df, features):
    with open(filename, 'w') as f:
        for _, row in df.iterrows():
            for val in zip(*row[features]):
                f.write(','.join([str(v) for v in val]) + ' ')
            f.write('\n')


def parse_log(input_dir, output_dir, log_file, parser_type):
    log_format = '<Label> <Id> <Date> <Code1> <Time> <Code2> <Component1> <Component2> <Level> <Content>'
    regex = [
        r'(0x)[0-9a-fA-F]+', #hexadecimal
        r'\d+.\d+.\d+.\d+',
        # r'/\w+( )$'
        r'\d+'
    ]
    keep_para = False
    if parser_type == "drain":
        # the hyper parameter is set according to http://jmzhu.logpai.com/pub/pjhe_icws2017.pdf
        st = 0.3  # Similarity threshold
        depth = 3  # Depth of all leaf nodes
        parser = Drain.LogParser(log_format, indir=input_dir, outdir=output_dir, depth=depth, st=st, rex=regex, keep_para=keep_para)
        parser.parse(log_file)
    elif parser_type == "spell":
        tau = 0.55
        parser = Spell.LogParser(indir=data_dir, outdir=output_dir, log_format=log_format, tau=tau, rex=regex, keep_para=keep_para)
        parser.parse(log_file)


if __name__ == "__main__":

    ##########
    # Parser #
    #########

    parse_log(data_dir, output_dir, log_file, 'drain')

    #########
    # Count #
    #########
    # count_anomaly()

    ##################
    # Transformation #
    ##################
    # mins
    window_size = 5
    step_size = 1
    train_ratio = 0.4

    df = pd.read_csv(f'{output_dir}{log_file}_structured.csv')

    # data preprocess
    df['datetime'] = pd.to_datetime(df['Time'], format='%Y-%m-%d-%H.%M.%S.%f')
    df["Label"] = df["Label"].apply(lambda x: int(x != "-"))
    df['timestamp'] = df["datetime"].values.astype(np.int64) // 10 ** 9
    df['deltaT'] = df['datetime'].diff() / np.timedelta64(1, 's')
    df['deltaT'].fillna(0)

    # sampling with sliding window
    deeplog_df = sliding_window(df[["timestamp", "Label", "EventId", "deltaT"]],
                                para={"window_size": int(window_size)*60, "step_size": int(step_size) * 60}
                                )

    #########
    # Train #
    #########
    df_normal =deeplog_df[deeplog_df["Label"] == 0]
    df_normal = df_normal.sample(frac=1, random_state=12).reset_index(drop=True) #shuffle
    normal_len = len(df_normal)
    train_len = int(normal_len * train_ratio)

    train = df_normal[:train_len]
    # deeplog_file_generator(os.path.join(output_dir,'train'), train, ["EventId", "deltaT"])
    deeplog_file_generator(os.path.join(output_dir,'train'), train, ["EventId"])

    print("training size {}".format(train_len))

    ###############
    # Test Normal #
    ###############
    test_normal = df_normal[train_len:]
    deeplog_file_generator(os.path.join(output_dir, 'test_normal'), test_normal, ["EventId"])
    print("test normal size {}".format(normal_len - train_len))

    del df_normal
    del train
    del test_normal
    gc.collect()

    #################
    # Test Abnormal #
    #################
    df_abnormal = deeplog_df[deeplog_df["Label"] == 1]
    deeplog_file_generator(os.path.join(output_dir,'test_abnormal'), df_abnormal, ["EventId"])
    print('test abnormal size {}'.format(len(df_abnormal)))
