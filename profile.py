import os, csv, argparse
import numpy as np

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    return parser.parse_args()

args = _parse_args()

with open('output/submission_{}.csv'.format(args.epoch), 'r', newline='') as infile:
    data = list(csv.DictReader(infile, delimiter=','))
    with open('resources/valid_textless.csv', 'r', newline='') as outfile:
        answers = list(csv.DictReader(outfile, delimiter=','))
        for threshold in np.arange(0.1,1.0,0.02):
            hit = 0
            miss = 0
            for idx, row in enumerate(data):
                target = '1' if float(row['Gold']) >= float(threshold) else '0'
                if target == answers[idx]['Gold']:
                    hit += 1
                else:
                    miss += 1
            print("Threshold:{:.3f}, Hit:{}, Miss:{}, Ratio:{:.5f}".format(threshold, hit, miss, hit/(hit+miss)))
