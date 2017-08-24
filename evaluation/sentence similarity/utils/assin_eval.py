# -*- coding: utf-8 -*-

'''
Script to evaluate system performance on the ASSIN shared task data.

Author: Erick Fonseca
'''

from __future__ import division, print_function

import argparse
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import pearsonr
from utils.commons import read_xml


def eval_rte(pairs_gold, pairs_sys):
    '''
    Evaluate the RTE output of the system against a gold score.
    Results are printed to stdout.
    '''
    # check if there is an entailment value
    if pairs_sys[0].entailment is None:
        print()
        print('No RTE output to evaluate')
        return

    gold_values = np.array([p.entailment for p in pairs_gold])
    sys_values = np.array([p.entailment for p in pairs_sys])
    macro_f1 = f1_score(gold_values, sys_values, average='macro')
    accuracy = (gold_values == sys_values).sum() / len(gold_values)

    print()
    print('RTE evaluation')
    print('Accuracy\tMacro F1')
    print('--------\t--------')
    print('{:8.2%}\t{:8.2f}'.format(accuracy, macro_f1))

def eval_similarity(pairs_gold, pairs_sys):
    '''
    Evaluate the semantic similarity output of the system against a gold score.
    Results are printed to stdout.
    '''
    # check if there is an entailment value
    if pairs_sys[0].similarity is None:
        print()
        print('No similarity output to evaluate')
        return

    gold_values = np.array([p.similarity for p in pairs_gold])
    sys_values = np.array([p.similarity for p in pairs_sys])
    pearson = pearsonr(gold_values, sys_values)[0]
    absolute_diff = gold_values - sys_values
    mse = (absolute_diff ** 2).mean()

    print()
    print('Similarity evaluation')
    print('Pearson\t\tMean Squared Error')
    print('-------\t\t------------------')
    print('{:7.2f}\t\t{:18.2f}'.format(pearson, mse))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gold_file', help='Gold file')
    parser.add_argument('system_file', help='File produced by a system')
    args = parser.parse_args()

    pairs_gold = read_xml(args.gold_file, True)
    pairs_sys = read_xml(args.system_file, True)

    eval_rte(pairs_gold, pairs_sys)
    eval_similarity(pairs_gold, pairs_sys)
