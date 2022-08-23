import time

import numpy as np
import sys, os
import torch, pickle
from models import *
from encode import encode


def main():
    t_begin = time.time()

    # Read and set parameters
    gnnfile = 'trainedmodels/GeoEnc.tor'
    gbtfile = 'trainedmodels/gbt-s4169.pkl'
    idxfile = 'trainedmodels/sortidx.npy'
    pdbfile = sys.argv[1]
    mutationinfo = sys.argv[2]
    if_info = sys.argv[3]
    foldxsavedir = '${DATA_DIR}/structures/FoldX'

    # Set flag whether a WT structure is already ready or not and has to be
    # optimized with FoldX
    wt_structure_ready = False
    if len(sys.argv) > 4:
        wt_structure_ready = bool(int(sys.argv[4]))

    # Set FoldX executable with specified version
    foldx_exec = './foldx'
    if len(sys.argv) > 5:
        foldx_version = int(sys.argv[5])
        if foldx_version == 4:
            foldx_exec = './foldx'
        elif foldx_version == 5:
            foldx_exec = '${SOFTWARE_DIR}/foldx5/foldx'
        else:
            raise ValueError('Wrong FoldX version specified')

    # Load (feature importance of node embeddings produced with GNN)?
    try:
        sorted_idx = np.load(idxfile)
    except:
        print('File reading error: Please redownload the file {} from the GitHub website again!'.format(idxfile))

    # Prepare working directory
    pdbfile, pdb, workdir = prepare_workdir(pdbfile, mutationinfo)

    # Generate interface residues
    interfacefile = prepare_interface(workdir, pdbfile, if_info)

    # Prepare wild-type and mutated structures with FoldX
    savedir = '${DATA_DIR}/foldx-structures'
    wildtypefile, mutantfile = prepare_structures(
        workdir, pdbfile, pdb, mutationinfo, foldx_exec, wt_structure_ready,
        foldxsavedir
    )

    # Convert built .pdb files to graph representation
    A, E, A_m, E_m = pdbs_to_graphs(
        wildtypefile, mutantfile, mutationinfo, interfacefile, if_info
    )

    # Load GBT model
    try:
        with open(gbtfile, 'rb') as pickle_file:
            forest = pickle.load(pickle_file)
    except:
        print('File reading error: Please redownload the file {} via the following command: \
                wget https://media.githubusercontent.com/media/Liuxg16/largefiles/8167d5c365c92d08a81dffceff364f72d765805c/gbt-s4169.pkl -P trainedmodels/'.format(gbtfile))

    # Load GNN encoder
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = GeometricEncoder(256)
    try:
        model.load_state_dict(torch.load(gnnfile,map_location=device))
    except:
        print('File reading error: Please redownload the file {} from the GitHub website again!'.format(gnnfile))

    # Move graph tensors to device
    model.to(device)
    model.eval()
    A = A.to(device)
    E = E.to(device)
    A_m = A_m.to(device)
    E_m = E_m.to(device)

    # Predict
    ddg = GeoPPIpredict(A,E,A_m,E_m, model, forest, sorted_idx)

    # Clean
    os.system('rm ./{}'.format(pdbfile))
    os.system(f'rm -rf ./{workdir}')

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
