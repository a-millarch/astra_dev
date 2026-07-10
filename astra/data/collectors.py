import os
import pandas as pd
import pyarrow.parquet as pq
import mltable
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azureml.core import Workspace, Datastore, Dataset, Environment

import logging 

from astra.utils import  is_file_present, are_files_present, ensure_parent_dir
from astra.utils import cfg, get_base_df

logger = logging.getLogger(__name__)


def download_to_local(FILES: list, LOCAL_DIR="data/dl"):
    #### Step 1: Download
    WS = Workspace.from_config()

    ## Get datastores
    datastore_sp = Datastore.get(WS, "sp_data")

    total_failed = 0
    os.makedirs(LOCAL_DIR, exist_ok=True)
    f_status = dict()
    for fn in FILES:
        if is_file_present(f"data/dl/CPMI_{fn}.parquet"):
            pass
        else:
            try:
                logger.info(f"> {fn}")
                ds = Dataset.File.from_files((datastore_sp, "CPMI_" + fn + ".parquet"))
                logger.info(">> Downloading...")
                ds.download(LOCAL_DIR)
                logger.info(">> Done!")
                f_status[fn] = True
            except Exception:
                f_status[fn] = False
                total_failed += 1
                logger.warning(f"Could not load {fn}!")
                
def collect_subsets(cfg, base=None):
    # First, load small files using population filter function
    for filename in cfg['default_load_filenames']:
        if is_file_present(f"data/raw/{filename}.csv"):
            logger.info(f'{filename} found in raw')
        else:
            population_filter_parquet(filename, base=base)
        
    # For larger files, we need to first download parquet to local first
    if are_files_present('dl', 
                         ['CPMI_'+i for i in cfg['large_load_filenames']], 
                         extension='.parquet'):
        logger.info('parquet files found locally, continue')
    else:
        logger.info('missing local parquet files, downloading')
        download_to_local(cfg['large_load_filenames'])
        
    # Now chunk filter to only population    
    for filename in cfg['large_load_filenames']:
        if is_file_present(f"data/raw/{filename}.csv"):
            logger.info(f'{filename} found in raw')
        else:
            logger.info(f'Processing {filename}')
            chunk_filter_parquet(filename, base=base)
        

def collect_procedures(cfg=cfg):
    path= f'{cfg["raw_file_path"]}CPMI_Procedurer.parquet'
    df_procedure = Dataset.Tabular.from_parquet_files(path=path)
    dtr_procedure = df_procedure.to_pandas_dataframe()
    traumepatienter = dtr_procedure[dtr_procedure['ProcedureCode'] == "BWST1F"][["CPR_hash", "ServiceDate"]]
    ensure_parent_dir("data/raw/Procedurer_population.csv")
    traumepatienter.to_csv("data/raw/Procedurer_population.csv")


def chunk_filter_parquet(filename, base = None, chunk_size = 4000000):
    if base is None:
        base = get_base_df()
        logger.info("Loaded base df")

    poplist= base['CPR_hash'].unique()
    # Specify the path to your Parquet file
    file_path = f'data/dl/CPMI_{filename}.parquet'
    output_path=f'data/raw/{filename}.csv'

    # Open the Parquet file
    parquet_file = pq.ParquetFile(file_path)

    # Define the chunk size
      # Adjust this as needed
    chunk_n = 0
    num_chunks = (parquet_file.metadata.num_rows / chunk_size)#.round(0)
    logger.info(f'>Initiating {num_chunks} chunks')
    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        chunk_n = chunk_n +1
        logger.debug("chunk %d of %d", chunk_n, num_chunks)
        chunk_df = batch.to_pandas() 
        chunk_df = chunk_df[chunk_df.CPR_hash.isin(poplist)]
        ensure_parent_dir(output_path)
        chunk_df.to_csv(output_path, mode='a', header= not os.path.exists(output_path))
    logger.info(f'Finished, saved file at: {output_path}')
        
    
def population_filter_parquet(filename, base =None, blobstore_uri=None):
    if base is None:
        base = get_base_df()
    if blobstore_uri is None:
        blobstore_uri = cfg["raw_file_path"]
    
    logger.info(f'Collecting and filtering {filename}')
    path = f'{blobstore_uri}CPMI_{filename}.parquet'
    #path = f'data/raw/CPMI_{filename}.parquet'
    ds = Dataset.Tabular.from_parquet_files(path=path)
    df = ds.to_pandas_dataframe()
    # if single mode then use ds class to filter?
    df = df[df.CPR_hash.isin(base.CPR_hash)]
    logger.info(f'loaded {len(df)} rows. Saving file.')

    ensure_parent_dir(f"data/raw/{filename}.csv")
    df.to_csv(f"data/raw/{filename}.csv")


