from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import sys
import csv
import datetime
from pathlib import Path
from time import perf_counter
from collections.abc import Iterable

from encode import encode
from predict import predict


KILL_WRITE_PROCESS_MSG = 'KILL_WRITE_PROCESS_MSG'


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
    # Compute outputs for chunk
    chunk_output = []
    for input in chunk:
        # Compute
        start = perf_counter()
        features = encode(*input, **encode_params)
        prediction = predict(features, **predict_params)
        finish = perf_counter()

        # Format and store output
        elapsed = finish - start
        mutation = input[1]
        output = (mutation, prediction, elapsed)
        chunk_output.append(output)

    # Write
    write_queue.put(chunk_output)


def write_process(write_queue, outfile, header):
    while True:
        data = write_queue.get()
        if data == KILL_WRITE_PROCESS_MSG:
            break
        write_to_csv(data, outfile, header)


def main():
    pdbfile = sys.argv[1]
    partners = sys.argv[2]
    mutfile = Path(sys.argv[3])
    outdir = Path(sys.argv[4])
    runid = sys.argv[5]
    # TODO other params from command line
    wt_kind = 'foldx_byproduct'
    if len(sys.argv) > 6:
        wt_kind = sys.argv[6]

    # Set parameters for parallelization
    max_workers = os.cpu_count()
    chunk_size = 5

    # Set output parameters
    os.makedirs(outdir, exist_ok=True)
    outfile = outdir / f'results-{runid}.csv'
    runstatsfile = outdir / 'runstats.csv'

    # Set screening parameters
    encode_params = dict(
        gnnfile='trainedmodels/GeoEnc.tor',
        foldx_exec='./foldx',
        foldxsavedir=None,
        wt_kind=wt_kind
    )
    predict_params = dict(
        gbtfile='trainedmodels/gbt-s4169.pkl',
        idxfile='trainedmodels/sortidx.npy'
    )

    # Construct input data
    with open(mutfile) as file:
        lines = file.readlines()
        mutations_list = [line.rstrip() for line in lines]
    pdbfiles_list = len(mutations_list) * [pdbfile]
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

        # Stop writer
        write_queue.put(KILL_WRITE_PROCESS_MSG)

    # Write run statistics
    # TODO cut paths to project home dir
    finish = perf_counter()
    elapsed = finish - start
    now = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    runstats = {
        'Datetime': now,
        'PDB file': pdbfile,
        'Partners': partners,
        'Screened space file': mutfile,
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
