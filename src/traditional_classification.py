import os
import cv2
import sys
import glob 
import copy
import skimage
import operator
import numpy as np
import scipy as scp
import pandas as pd
from collections import Counter

from trend_detector import LineChart

        
def get_tags(run_mode):
    folder = '../data/'
    labels = open(folder+'{}_labels_linear.txt'.format(run_mode), 'r').readlines()
    graph_to_labels = {}
    for r in labels:
        rlabels = r.rstrip().split(' ')
        graph_to_labels[rlabels[0]] = rlabels[1:]
    return graph_to_labels

def main(run_mode, pct, single_line=True, shuffle=True):
    folder='../data/{}/'.format(run_mode)
    if run_mode=='test':
        files = [line.strip().split(' ') for line in open(folder+'concensus_labels.txt', 'r').readlines()]
    else:
        files = list(glob.glob(folder+"*.png"))
        graph_to_labels = get_tags(run_mode)
    if shuffle:
        np.random.shuffle(files)
    files = files[:int(len(files)*pct)] 
    num_files = len(files)
    correct = 0
    output_file = open('traditional_output.txt', 'w')

    print("Processing {} images from {} set...".format(num_files, run_mode))
    print("FILE                 |   GUESS   |   ACTUAL")
    print('-'*50)

    for i, chartpath  in enumerate(files):
        if run_mode=='test':
            chartpath, actual = chartpath 
            chartpath = folder+chartpath
            actual = [actual]
        print(chartpath, end="")
        chart = LineChart(chartpath)
        if single_line:
            trends = chart.separate_one_line()
        else:
            imgs, trends = chart.get_trends()
        file_num = chartpath.split('/')[-1]
        print(" |  {}  |  {}\n".format( " ".join(trends), actual[0]))
        output_file.write("{} {}\n".format(file_num, " ".join(trends)))
        trends.sort(key=lambda x: x[0])
        actual.sort(key=lambda x: x[0])
        if trends==actual:
            correct += 1

        if i*100 % num_files == 0 and i != 0:
            print("*"*50)
            print(" "*10,"{}% processed...".format(100*i/num_files))
            print("*"*50)

    output_file.close()
    print("Process finished. Accuracy: {}/{}".format(correct, len(files)))

if __name__=='__main__':

    if len(sys.argv)<2:
        run_mode = input(str("Do you want to run the 'train' set, 'val' set, or 'test'set?\n "))
    else:
        run_mode = sys.argv[1] 

    if len(sys.argv)<3:
        folder='../data/{}/'.format(run_mode)
        files = list(glob.glob(folder+"*.png"))
        pct = int(input(str("What percentage of the {} set do you want to run, out of {} files  (enter integer number, not decimal)? \n".format(run_mode, len(files)))))/100
        print(pct)
    else:
        pct = sys.argv[2]
    main(run_mode, pct) #second argument implies percentage of folder to run 
    






