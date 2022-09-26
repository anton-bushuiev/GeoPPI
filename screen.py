import os
import sys
import csv
import datetime
import traceback
from pathlib import Path
from time import perf_counter
from collections.abc import Iterable
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import pickle
import numpy as np
import torch

from models import GeometricEncoder
from encode import encode
from predict import predict


KILL_WRITE_PROCESS_MSG = 'KILL_WRITE_PROCESS_MSG'
ROOT_DIR = os.environ.get('ROOT_DIR')


# TODO move to utils.misc
def list_to_chunks(lst, n):
    """
    Yield successive n-sized chunks from lst.
    https://stackoverflow.com/a/312464/17572887
    """
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


# TODO import from utils.pdb
def path_to_id(path):
    return Path(path).stem


# TODO move to utils.io
def is_iterable_and_not_str(obj):
    return isinstance(obj, Iterable) and not isinstance(obj, str)


# TODO move to utils.io
def write_to_csv(data, outfile, header=None):
    """
    :param data: list or 2D list
    :param outfile: path to output .csv file
    :param header: list specifying .csv header
    """
    # Check if file does not exists or empty to write header
    write_header = False
    if header is not None:
        if not outfile.exists() or outfile.stat().st_size == 0:
            write_header = True

    with open(outfile, 'a', newline='') as file:
        writer = csv.writer(file, delimiter=';')

        # Write header
        if write_header:
            writer.writerow(header)

        # Write rows
        if is_iterable_and_not_str(data):
            if is_iterable_and_not_str(data[0]):
                for row in data:
                    writer.writerow(row)
            else:
                writer.writerow(data)
        else:
            raise TypeError('Wrong output type')


def get_results_csv_header():
    return 'Mutation', 'ddG', 'Elapsed time'


def screen_chunk(chunk, encode_params, predict_params, write_queue):
    # Start timer
    start = perf_counter()

    # Encode
    chunk_features = []
    for input in chunk:
        features = encode(*input, **encode_params)
        chunk_features.append(features)
    chunk_features = np.array(chunk_features, dtype=object)

    # Predict
    chunk_prediction = predict(chunk_features, **predict_params)

    # Finish timer
    finish = perf_counter()
    elapsed = finish - start

    # Format and store output
    chunk_mutations = list(map(lambda input: input[1], chunk))
    chunk_elapsed = len(chunk) * [elapsed / len(chunk)]
    chunk_output = list(zip(chunk_mutations, chunk_prediction, chunk_elapsed))
    write_queue.put(chunk_output)


def write_process(write_queue, outfile, header):
    while True:
        data = write_queue.get()
        if data == KILL_WRITE_PROCESS_MSG:
            break
        write_to_csv(data, outfile, header)


def main():
    pdbfile = Path(sys.argv[1])
    partners = sys.argv[2]
    mutfile = Path(sys.argv[3])
    outdir = Path(sys.argv[4])
    runid = sys.argv[5]
    # TODO other params from command line
    wt_kind = 'foldx_byproduct'
    if len(sys.argv) > 6:
        wt_kind = sys.argv[6]

    # Set parameters for parallelization
    max_workers = os.cpu_count() - 2
    chunk_size = 5

    # Set output parameters
    os.makedirs(outdir, exist_ok=True)
    outfile = outdir / f'results-{runid}.csv'
    runstatsfile = outdir / 'runstats.csv'

    # Set screening parameters
    gnnfile = 'trainedmodels/GeoEnc.tor'
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # gnnmodel = GeometricEncoder(256)
    # try:
    #     gnnmodel.load_state_dict(torch.load(gnnfile, map_location=device))
    # except Exception as exc:
    #     print('File reading error: Please redownload the file {}'
    #           ' from the GitHub website again!'.format(gnnfile))
    #     print(exc)
    # gnnmodel.to(device)
    # gnnmodel.eval()
    encode_params = dict(
        # gnnmodel=gnnmodel,
        gnnfile=gnnfile,
        foldx_exec='./foldx',
        foldxsavedir=None,
        wt_kind=wt_kind
    )

    # Set prediction parameters
    xgbfile = 'trainedmodels/SKEMPI2-ML/predictor.pkl'
    with open(xgbfile, 'rb') as pickle_file:
        xgb_model = pickle.load(pickle_file)
    predict_params = dict(
        predictor_sklearn=xgb_model
        # xgbfile='trainedmodels/SKEMPI2-ML/predictor.pkl'
        # gbtfile='trainedmodels/gbt-s4169.pkl',
        # idxfile='trainedmodels/sortidx.npy'
    )

    # Construct input chunks
    with open(mutfile) as file:
        lines = file.readlines()
        mutations_list = [line.rstrip() for line in lines]
    pdbfiles_list = len(mutations_list) * [str(pdbfile)]
    partners_list = len(mutations_list) * [partners]
    inputs = list(zip(pdbfiles_list, mutations_list, partners_list))
    chunks = list_to_chunks(inputs, chunk_size)

    # Prepare multi-processing
    manager = mp.Manager()
    write_queue = manager.Queue()

    # Run screening in parallel
    start = perf_counter()
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Start writer
        header = get_results_csv_header()
        executor.submit(write_process, write_queue, outfile, header)

        # Submit chunks
        future_to_chunk = {
            executor.submit(
                screen_chunk, chunk, encode_params, predict_params,
                write_queue
            ): chunk for chunk in chunks
        }
        for future in as_completed(future_to_chunk):
            chunk = future_to_chunk[future]
            try:
                future.result()
            except Exception as exc:
                print(f'{chunk} generated an exception: {exc}')
                print(traceback.format_exc(), end='\n\n')

        # Stop writer
        write_queue.put(KILL_WRITE_PROCESS_MSG)

    # Write run statistics
    finish = perf_counter()
    elapsed = finish - start
    now = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    pdbfile_rel = pdbfile.relative_to(ROOT_DIR) if ROOT_DIR else pdbfile
    mutfile_rel = mutfile.relative_to(ROOT_DIR) if ROOT_DIR else mutfile
    runstats = {
        'Datetime': now,
        'PDB file': pdbfile_rel,
        'Partners': partners,
        'Screened space file': mutfile_rel,
        'Max workers': max_workers,
        'Chunk size': chunk_size,
        'Elapsed time': elapsed,
        'Run id': runid,
        'Wild type kind': wt_kind
    }
    output = list(runstats.values())
    header = list(runstats.keys())
    write_to_csv(output, runstatsfile, header)


if __name__ == '__main__':
    main()
