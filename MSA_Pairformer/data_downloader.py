import os
import msa_model.utils
import multiprocessing as mp
from tqdm import tqdm

def download_AF_pdb(prot_id, out_dir):
    url = f"https://alphafold.ebi.ac.uk/files/AF-{prot_id}-F1-model_v4.pdb"
    cmd = f"wget -P {out_dir} {url}"
    res = msa_model.utils._run(cmd)
    log_file_prefix = os.path.join(out_dir, "download.log")
    msa_model.utils.write_log(log_file_prefix, res)

def download_rcsb_pdb(pdb_id, out_dir, log=False, cif=False):
    if not cif:
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        cmd = f"wget {url} -O {out_dir}/{pdb_id}.pdb"
    else:
        url = f"https://files.rcsb.org/download/{pdb_id}.cif"
        cmd = f"wget {url} -O {out_dir}/{pdb_id}.cif"
    res = msa_model.utils._run(cmd)
    if log:
        log_file_prefix = os.path.join(out_dir, "download.log")
        msa_model.utils.write_log(log_file_prefix, res)

def download_rcsb_pdb_mp(param_d: dict):
    pdb_id = param_d["pdb_id"]
    out_dir = param_d["out_dir"]
    log = param_d["log"]
    cif = param_d["cif"]
    download_rcsb_pdb(pdb_id, out_dir, log, cif)

def download_rcsb_pdbs(pdb_ids_l: list, out_dir: str, log: bool = False, cpu_buffer: int = 4, cif: bool = False):
    with mp.Pool(processes=mp.cpu_count() - cpu_buffer) as pool:
        list(tqdm(pool.imap(download_rcsb_pdb_mp, [{'pdb_id': pdb_id, 'out_dir': out_dir, 'log': log, 'cif': cif} for pdb_id in pdb_ids_l]), total=len(pdb_ids_l), leave=True))
        pool.close()
