import sys
import os
import os.path as path
import warnings
import argparse
import torch
import pickle
import string
import random
from itertools import groupby
from pathlib import Path
import Bio.PDB
from models import *


def gen_graph_data(pdbfile, mutinfo, interfile,  cutoff, if_info=None):
    max_dis = 12
    pdbfile = open(pdbfile)
    lines = pdbfile.read().splitlines()
    chainid = [x.split('_')[0] for x in mutinfo]
    interface_res = read_inter_result(interfile,if_info, chainid)
    if len(interface_res)==0: print('Warning: We do not find any interface residues between the two parts: {}. Please double check your inputs. Thank you!'.format(if_info))
    sample = build_graph(lines, interface_res,mutinfo, cutoff,max_dis)
    return sample


def read_inter_result(path, if_info=None, chainid=None, old2new=None):
    if if_info is not None:
        info1 = if_info.split('_')
        pA = info1[0]
        pB = info1[1]
        mappings = {}
        for a in pA:
            for b in pB:
                if a not in mappings:
                    mappings[a] = [b]
                else:
                    mappings[a] += [b]
                if b not in mappings:
                    mappings[b] = [a]
                else:
                    mappings[b] += [a]


        target_chains = []
        for chainidx in chainid:
            if chainidx in mappings:
                target_chains += mappings[chainidx]

        target_inters = []
        for chainidx in chainid:
            target_inters += ['{}_{}'.format(chainidx,y) for y in target_chains]+ \
                             ['{}_{}'.format(y,chainidx) for y in target_chains]

        target_inters =list(set(target_inters))
    else:
        target_inters = None

    inter = open(path)
    interlines = inter.read().splitlines()
    interface_res = []
    for line in interlines:
        iden = line[:3]
        if target_inters is None:
            if iden.split('_')[0] not in chainid and iden.split('_')[1] not in chainid:
                continue
        else:
            if iden not in target_inters:
                continue
        infor = line[4:].strip().split('_')#  chainid, resid
        assert len(infor)==2
        interface_res.append('_'.join(infor))

    if old2new is not None:
        mapps = {x[:-4] :y[:-4]  for x,y in old2new.items()}
        interface_res = [mapps[x] for x in interface_res if x in mapps]

    return interface_res


def build_graph(lines, interface_res, mutinfo, cutoff=3, max_dis=12, noisedict = None):
    atomnames = ['C','N','O','S']
    residues = ['ARG','MET','VAL','ASN','PRO','THR','PHE','ASP','ILE', \
                'ALA','GLY','GLU','LEU','SER','LYS','TYR','CYS','HIS','GLN','TRP']
    res_code = ['R','M','V','N','P','T','F','D','I', \
                'A','G','E','L','S','K','Y','C','H','Q','W']
    res2code ={x:idxx for x,idxx in zip(residues, res_code)}

    atomdict = {x:i for i,x in enumerate(atomnames)}
    resdict = {x:i for i,x in enumerate(residues)}
    V_atom = len(atomnames)
    V_res = len(residues)

    # build chain2id
    chain2id = []
    interface_coordinates = []
    line_list= []
    mutant_coords = []
    for line in lines:
        if line[0:4] == 'ATOM':
            atomname = line[12:16].strip()
            elemname = list(filter(lambda x: x.isalpha(), atomname))[0]
            resname  = line[16:21].strip()
            chainid =  line[21]
            res_idx = line[22:28].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            if elemname not in atomdict:
                continue

            coords = torch.tensor([x,y,z])
            atomid = atomdict[elemname]
            if resname not in resdict:
                resname = resname[1:]
            if resname not in resdict:
                continue

            if chainid not in chain2id:
                chain2id.append(chainid)

            line_token = '{}_{}_{}_{}'.format(atomname,resname,chainid, res_idx)
            if line_token not in line_list:
                line_list.append(line_token)
            else:
                continue

            resid  = resdict[resname]
            cr_token = '{}_{}'.format(chainid, res_idx)
            float_cd  = [float(x) for x in coords]
            cd_tensor = torch.tensor(float_cd)
            if cr_token in interface_res:
                interface_coordinates.append(cd_tensor)
            if mutinfo is not None and cr_token in mutinfo:
                interface_coordinates.append(cd_tensor)
                interface_res.append(cr_token)
                mutant_coords.append(cd_tensor)

    inter_coors_matrix = torch.stack(interface_coordinates)
    chain2id = {x:i for i,x in enumerate(chain2id)}
    global_resid2noise = {}

    n_features = V_atom+V_res+1+1+3 +1 +1+1+1
    line_list= []
    atoms = []
    flag_mut = False
    res_index_set = {}
    for line in lines:
        if line[0:4] == 'ATOM':
            features = [0]*n_features
            atomname = line[12:16].strip()
            elemname = list(filter(lambda x: x.isalpha(), atomname))[0]
            resname  = line[16:21].strip()
            chainid =  line[21]
            res_idx = line[22:28].strip()
            x = float(line[30:38].strip())
            y = float(line[38:46].strip())
            z = float(line[46:54].strip())
            if elemname not in atomdict:
                continue

            coords = torch.tensor([x,y,z])
            atomid = atomdict[elemname]
            if resname not in resdict:
                resname = resname[1:]
            if resname not in resdict:
                continue
            line_token = '{}_{}_{}_{}'.format(atomname,resname,chainid, res_idx)
            if line_token not in line_list:
                line_list.append(line_token)
            else:
                continue

            resid  = resdict[resname]
            features[atomid] = 1
            features[V_atom+resid] = 1

            cr_token = '{}_{}'.format(chainid, res_idx)
            float_cd  = [float(x) for x in coords]
            cd_tensor = torch.tensor(float_cd)
            #24
            if cr_token in interface_res:
                features[V_atom+V_res] = 1

            if mutinfo is not None:
                for inforrr in mutinfo:
                    mut_chainid = inforrr.split('_')[0]
                    if chainid==mut_chainid:
                        #25
                        features[V_atom+V_res+1] = 1
            #26
            features[V_atom+V_res+2] = chain2id[chainid]

            #27
            if cr_token not in res_index_set:
                res_index_set[cr_token] = len(res_index_set)+1

            features[V_atom+V_res+3] = res_index_set[cr_token]

            #28
            if atomname=='CA':
                features[V_atom+V_res+4] = res_index_set[cr_token]
                if noisedict is not None and cr_token in noisedict:
                    global_resid2noise[res_index_set[cr_token]] = noisedict[cr_token]

            flag = False
            dissss = torch.norm(cd_tensor-inter_coors_matrix,dim=1)
            flag = (dissss<max_dis).any()


            #29-31
            features[V_atom+V_res+5:V_atom+V_res+8] = float_cd

            res_iden_token = '{}_{}_{}'.format(chainid, res_idx, resname).upper()
            if  mutinfo is not None and cr_token in mutinfo:
                #32
                features[V_atom+V_res+8]=1
                flag_mut = True
                flag = True

            if flag:
                atoms.append(features)

    if mutinfo is not None and len(interface_res)>0:
        assert flag_mut==True

    if len(atoms)<5:
        return None
    atoms = torch.tensor(atoms, dtype=torch.float)
    N = atoms.size(0)
    atoms_type = torch.argmax(atoms[:,:4],1)
    atoms_type = atoms_type.unsqueeze(1).repeat(1,N)
    edge_type = atoms_type*4+atoms_type.t()

    pos  = atoms[:,-4:-1] #N,3
    row = pos[:,None,:].repeat(1,N,1)
    col = pos[None,:,:].repeat(N,1,1)
    direction = row-col
    del row, col
    distance = torch.sqrt(torch.sum(direction**2,2))+1e-10
    distance1 = (1.0/distance)*(distance<float(cutoff)).float()
    del distance
    diag = torch.diag(torch.ones(N))
    dist = diag+ (1-diag)*distance1
    del distance1, diag
    flag = (dist>0).float()
    direction = direction*flag.unsqueeze(2)
    del direction, dist
    edge_sparse = torch.nonzero(flag) #K,2
    edge_attr_sp = edge_type[edge_sparse[:,0],edge_sparse[:,1]] #K,4
    if noisedict is None:
        savefilecont = [ atoms, edge_sparse, edge_attr_sp]
    else:
        savefilecont = [ atoms, edge_sparse, edge_attr_sp, global_resid2noise]
    return savefilecont


def prepare_workdir(pdbfile, mutationinfo):
    # Generate random id
    # Note: Needed to avoid collision if the same copied mutation is run
    # in parallel in two or more processes (which is not a very practical
    # scenario though)
    random_id = ''.join(random.choices(string.ascii_letters, k=20))

    # Rename original .pdb file to incorporate mutation info
    mutationinfo_osname = mutationinfo.replace(',', '_')
    os.system(f'cp {pdbfile} {pdbfile[:-4]}-{mutationinfo_osname}-{random_id}-{pdbfile[-4:]}')

    # Move renamed file to local directory
    pdbfile = f'{pdbfile[:-4]}-{mutationinfo_osname}-{random_id}-{pdbfile[-4:]}'
    os.system('mv {} ./'.format(pdbfile))

    # Set up working directory
    pdbfile = pdbfile.split('/')[-1]
    pdb = pdbfile.split('.')[0]
    workdir = f'workdir-{pdb}'
    if path.exists('./{}'.format(workdir)):
        os.system('rm -r {}'.format(workdir))
    os.system('mkdir {}'.format(workdir))

    return pdbfile, pdb, workdir


def prepare_interface(workdir, pdbfile, if_info):
    os.system('python gen_interface.py {} {} {} > {}/pymol.log'.format(pdbfile, if_info,workdir,workdir))
    interfacefile = '{}/interface.txt'.format(workdir)
    return interfacefile


def run_foldx(foldx_exec, pdbfile, individual_list, outdir):
    command = (
        f'{foldx_exec}'
        f' --command=BuildModel'
        f' --pdb={pdbfile}'
        f' --mutant-file={individual_list}'
        f' --output-dir={outdir}'
        f' > {outdir}/foldx.log'
    )
    return os.system(command)


# TODO move to utils.mutations
def is_mutation_reverse(pdbfile, mutationinfo, model_id=0):
    # Select first mutation
    # (Relaxed to check only the first one for better performance)
    mut = mutationinfo.split(',')[0]

    # Parse mutation info
    wtres, chainid, pos, mutres = mut[0], mut[1], int(mut[2:-1]), mut[-1]

    # Read structure
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = Bio.PDB.PDBParser().get_structure(None, pdbfile)[model_id]

    # Check mutated amino acid
    resname = model[chainid][pos].get_resname()
    resname = Bio.PDB.Polypeptide.protein_letters_3to1[resname]

    if resname == wtres:
        return False
    elif resname == mutres:
        return True
    else:
        raise ValueError(f'{mut} is neither forward nor reverse.')


# TODO import from utils.mutations
def revert_mutation(mut):
    """
    :param mut: str representing single- or multi-point mutation. Examples:
        YC17T
        YC17T,TA20A
    :return: str representing single- or multi-point mutation. Examples:
        TC17Y
        TC17Y,AA20T
    """
    # Check if single- or multi-point
    if ',' not in mut:
        # If single-, revert
        return mut[-1] + mut[1:-1] + mut[0]
    else:
        # If multiple-, call recursively for each one
        mut_list = mut.replace(' ', '').split(',')
        return ','.join(list(map(revert_mutation, mut_list)))


def all_equal(iterable):
    g = groupby(iterable)
    return next(g, True) and not next(g, False)


# TODO Read structures from cache
def prepare_structures(
        workdirs, pdbfiles, pdbs, mutationinfos,
        foldx_exec, foldxsavedir, wt_kind='foldx_byproduct'
):
    # Check input
    if wt_kind not in ['original', 'foldx_itself', 'foldx_byproduct']:
        raise ValueError('Wrong wt-kind value')

    # TODO Remove redundant workdirs
    # pdbs_pref = list(map(lambda: x.split('-')[0], pdbfiles))
    # pdbfiles_pref = list(map(lambda: x.split('-')[0], pdbfiles))
    # if len(pdbs) > 1 and (not all_equal(pdbs_pref) or not all_equal(pdbfiles_pref)):
    #     raise ValueError('Batched FoldX execution is only available on same'
    #                      ' structure')
    pdbfile = pdbfiles[0]
    pdb = pdbs[0]

    # Define necessary paths
    workdir_common = f'{workdirs[0]}/common'
    os.mkdir(workdir_common)
    individual_list_path = f'{workdir_common}/individual_list.txt'

    wt_paths = [
        f'{workdir_common}/WT_{pdb}_{i+1}.pdb' for i in range(len(pdbs))
    ]
    mut_paths = [
        f'{workdir_common}/{pdb}_{i+1}.pdb' for i in range(len(pdbs))
    ]

    # If mutations is reverse, revert back and swap roles
    mutationinfos = list(mutationinfos)
    for i in range(len(mutationinfos)):
        mutation_is_reverse = is_mutation_reverse(pdbfile, mutationinfos[i])
        if mutation_is_reverse:
            mutationinfos[i] = revert_mutation(mutationinfos[i])
            wt_paths[i], mut_path[i] = mut_path[i], wt_path[i]

    # Run FoldX to mutate structure
    # This run also automatically produces a wild-type structure for
    # wt_kind == 'foldx_byproduct' scenario
    with open(individual_list_path, 'w') as handle:
        for mutationinfo in mutationinfos:
            handle.write(mutationinfo + ';\n')
    run_foldx(foldx_exec, pdbfile, individual_list_path, workdir_common)

    # Save structures
    # if foldxsavedir is not None:
    #     wt_path_cache = f'{foldxsavedir}/{pdb}-wt.pdb'
    #     mut_path_cache = f'{foldxsavedir}/{pdb}-mut.pdb'
    #     os.system(f'cp {wt_path} {wt_path_cache}')
    #     os.system(f'cp {mut_path} {mut_path_cache}')

    # Clean
    # os.system(f'rm -f {individual_list_path}')

    return wt_paths, mut_paths


def pdbs_to_graphs(
        wildtypefile, mutantfile, mutationinfo, interfacefile, if_info, workdir
):
    # Construct graphs from built .pdb files
    graph_mutinfo = []
    for mut in mutationinfo.split(','):
        graph_mutinfo.append(f'{mut[1]}_{mut[2:-1]}')
    cutoff = 3

    try:
        A, E, _ = gen_graph_data(wildtypefile, graph_mutinfo, interfacefile,
                                 cutoff, if_info)
        A_m, E_m, _= gen_graph_data(mutantfile, graph_mutinfo, interfacefile,
                                    cutoff, if_info)
    except:
        print('Data processing error: Please double check your inputs is'
              ' correct! Such as the pdb file path, mutation information'
              ' and binding partners. You might find more error details'
              ' at {}/foldx.log'.format(workdir))

    return A, E, A_m, E_m


def encode(
        pdbfiles, mutationinfos, if_infos,
        gnnmodel=None,
        gnnfile='trainedmodels/GeoEnc.tor',
        foldx_exec='./foldx',
        foldxsavedir=None,
        wt_kind='foldx_byproduct',
        foldx_chunked=False
):
    # Load GNN encoder
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if gnnmodel is None:
        gnnmodel = GeometricEncoder(256)
        try:
            gnnmodel.load_state_dict(torch.load(gnnfile, map_location=device))
        except:
            print('File reading error: Please redownload the file {}'
                  ' from the GitHub website again!'.format(gnnfile))
        gnnmodel.to(device)
        gnnmodel.eval()

    # 1. Sequential step
    pdbfiles_new, pdbs, workdirs, interfacefiles = [], [], [], []
    inputs = zip(pdbfiles, mutationinfos, if_infos)
    for pdbfile, mutationinfo, if_info in inputs:
        # Prepare working directory
        pdbfile, pdb, workdir = prepare_workdir(pdbfile, mutationinfo)

        # Generate interface residues
        interfacefile = prepare_interface(workdir, pdbfile, if_info)

        pdbfiles_new.append(pdbfile)
        pdbs.append(pdb)
        workdirs.append(workdir)
        interfacefiles.append(interfacefile)
    pdbfiles = pdbfiles_new

    # 2. Sequential/Parallel step
    if foldx_chunked:
        wildtypefiles, mutantfiles = prepare_structures(
            workdirs, pdbfiles, pdbs, mutationinfos,
            foldx_exec, foldxsavedir, wt_kind
        )
    else:
        wildtypefiles, mutantfiles = [], []
        inputs = zip(workdirs, pdbfiles, pdbs, mutationinfos)
        for workdir, pdbfile, pdb, mutationinfo in inputs:
            wildtypefiles_single, mutantfiles_single = prepare_structures(
                [workdir], [pdbfile], [pdb], [mutationinfo],
                foldx_exec, foldxsavedir, wt_kind
            )
            wildtypefiles.extend(wildtypefiles_single)
            mutantfiles.extend(mutantfiles_single)

    # 3. Sequential step
    features = []
    inputs = zip(wildtypefiles, mutantfiles, mutationinfos,
                 interfacefiles, if_infos, workdirs, pdbfiles)
    for wildtypefile, mutantfile, mutationinfo,\
        interfacefile, if_info, workdir, pdbfile in inputs:
        # Convert built .pdb files to graph representation
        A, E, A_m, E_m = pdbs_to_graphs(
            wildtypefile, mutantfile, mutationinfo,
            interfacefile, if_info, workdir
        )
        # Move graph tensors to device
        A = A.to(device)
        E = E.to(device)
        A_m = A_m.to(device)
        E_m = E_m.to(device)

        # Encode
        fea = GeoPPIencode(A, E, A_m, E_m, gnnmodel)
        fea = fea.cpu().detach().numpy()

        # Clean
        os.system('rm ./{}'.format(pdbfile))
        os.system(f'rm -rf ./{workdir}')

        features.append(fea)

    features = np.array(features, dtype=object)
    return features


def main():
    """
    This main function is designed for encoding of several mutations of the
    same structure and chains.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('posargs', nargs=2)
    parser.add_argument('--outdir', default=None)
    parser.add_argument('--foldx_savedir', default=None)
    parser.add_argument('--foldx_exec', default='./foldx')
    parser.add_argument('--mutationinfo', default=None)
    parser.add_argument('--mutfile', default=None)
    args = parser.parse_args()
    pdbfile, if_info = args.posargs

    # Initialize mutations
    if args.mutationinfo is not None and args.mutfile is not None:
        raise ValueError('Both `mutationinfo` and `mutfile` specified.')
    elif args.mutationinfo is not None:
        mutationinfos = (args.mutationinfo,)
    elif args.mutfile is not None:
        with open(args.mutfile, 'r') as handle:
            mutationinfos = tuple(handle.read().splitlines())
    else:
        raise ValueError('Mutation(s) are not specified.')

    # Clone structure and chain info
    pdbfiles = len(mutationinfos) * (pdbfile,)
    if_infos = len(mutationinfos) * (if_info,)

    # Encode
    features = encode(
        pdbfiles, mutationinfos, if_infos,
        gnnfile='trainedmodels/GeoEnc.tor',
        foldx_exec=args.foldx_exec,
        foldxsavedir=args.foldx_savedir
    )

    # TODO
    # Store output
    # outdir = args.outdir
    # if outdir:
    #     # Write to file
    #     pdb = Path(pdbfile).stem
    #     outfile = Path(outdir) / f'{pdb}-{mutationinfo}-{if_info}'
    #     with open(outfile, 'wb') as file:
    #         pickle.dump(fea, file)
    # else:
    #     # Print features
    #     fea = map(str, fea.tolist())
    #     print(';'.join(fea))


if __name__ == '__main__':
    main()
