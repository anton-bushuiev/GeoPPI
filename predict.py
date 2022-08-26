import time

import numpy as np
import sys
import os
import torch
import pickle

from models import *
from encode import encode


# TODO Rewrite to XGBoost
def predict(
    fea,
    gbtfile,
    idxfile,
):
    # Load GBT model
    try:
        with open(gbtfile, 'rb') as pickle_file:
            forest = pickle.load(pickle_file)
    except:
        raise Exception('File reading error: Please redownload the file {} via the following command: \
                wget https://media.githubusercontent.com/media/Liuxg16/largefiles/8167d5c365c92d08a81dffceff364f72d765805c/gbt-s4169.pkl -P trainedmodels/'.format(gbtfile))

    # Load (feature importance of node embeddings produced with GNN)?
    try:
        sorted_idx = np.load(idxfile)
    except:
        raise Exception('File reading error: Please redownload the file {} from the GitHub website again!'.format(idxfile))

    # Predict
    ddg = GeoPPIpredict(fea, forest, sorted_idx)
    return ddg


def main():
    t_begin = time.time()

    # Read input parameters
    pdbfile = sys.argv[1]
    mutationinfo = sys.argv[2]
    if_info = sys.argv[3]

    # Read path to FoldX cache with optimized structures
    foldxsavedir = None
    if len(sys.argv) > 4:
        foldxsavedir = sys.argv[4]

    # Set FoldX executable with specified version
    foldx_exec = './foldx'
    if len(sys.argv) > 5:
        foldx_exec = sys.argv[5]

    # Set forest parameters
    gbtfile = 'trainedmodels/gbt-s4169.pkl'
    idxfile = 'trainedmodels/sortidx.npy'

    # Encode
    fea = encode(
        pdbfile, mutationinfo, if_info,
        gnnfile='trainedmodels/GeoEnc.tor',
        foldx_exec=foldx_exec,
        foldxsavedir=None
    )

    # Predict
    ddg = predict(fea, gbtfile, idxfile)

    # Print prediction and running time
    runtime = time.time() - t_begin
    print(f'{ddg};{runtime}', end=';')
    # print('='*40+'Results'+'='*40)
    # if ddg<0:
    #     mutationeffects = 'destabilizing'
    #     print('The predicted binding affinity change (wildtype-mutant) is {} kcal/mol ({} mutation).'.format(ddg,mutationeffects))
    # elif ddg>0:
    #     mutationeffects = 'stabilizing'
    #     print('The predicted binding affinity change (wildtype-mutant) is {} kcal/mol ({} mutation).'.format(ddg,mutationeffects))
    # else:
    #     print('The predicted binding affinity change (wildtype-mutant) is 0.0 kcal/mol.')


if __name__ == '__main__':
    main()
