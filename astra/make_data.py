import argparse
import logging
import os

import pandas as pd
import numpy as np

from astra.utils import ProjectManager, cfg, setup_logging, ensure_parent_dir
from astra.utils import is_file_present, are_files_present

logger = logging.getLogger(__name__)

from astra.data.collectors import collect_subsets
import astra.data.build_patient_info as bpi
from astra.data.filters import filter_subsets_inhospital, mark_traumatext
from astra.data.mapper import map_concept, map_concept_optimized

from astra.data.datasets import TSDS

def generate_base_df():
    # JUST A TEMPORARY TESTER FUNCTION, used by load_or_collect_population
    
    ensure_parent_dir('data/external/trauma_call.csv')
    pd.DataFrame.from_dict({'CPR_hash':['FFFB69AEF2D7DED6288C835FE45672455D6E68F1F725207109750F772EDC68C4'],
    'ServiceDate':[np.datetime64('2023-08-20T15:21:00.000000000')]}, orient='columns').to_csv('data/external/trauma_call.csv')
    
   
    # saved as pickle

def proces_raw_concepts(cfg, base= None, reset=False): # move to construct data_sets?
    subsets_filenames = cfg["default_load_filenames"] + cfg["large_load_filenames"]
    if (
            are_files_present("data/raw", subsets_filenames, extension=".csv")
            and reset == False
        ):
            logger.info("All subsets found, continuing")
    else:
            logger.info("Subsets missing, collecting missing")
            collect_subsets(cfg, base=base)

def proces_inhospital_concepts(cfg, reset=False):
    subsets_filenames = cfg["default_load_filenames"] + cfg["large_load_filenames"]
    if (
        are_files_present("data/interim/concepts", subsets_filenames, extension=".pkl")
        and reset == False
    ):
        logger.info("Interim subsets found, continuing")
    else:
        logger.info("Filtering subsets")
        filter_subsets_inhospital(cfg)
    
def map_data(cfg):
    logger.info("Mapping data to bins")
    map_dir = "data/interim/mapped/"
    for concept in cfg["concepts"]:
        for agg_func in cfg["agg_func"][concept]:
            if is_file_present(
                f"{map_dir}{concept}_{agg_func}.csv"
            ) and is_file_present(f"{map_dir}{concept}_{agg_func}.pkl"):
                pass
            else:
                logger.debug(f"Binning and mapping {concept} with agg_func: {agg_func}")
                if concept in cfg["dataset"]["ts_cat_names"]:
                    is_categorical = True
                    is_multi_label =True
                else:
                    is_categorical = False
                    is_multi_label =False                    
                map_concept(cfg, concept, agg_func, is_categorical, is_multi_label)

def map_data_optimized(cfg, overwrite=False):
    """Updated to use optimized mapper."""
    logger.info("Mapping data to bins")
    map_dir = "data/interim/mapped/"

    for concept in cfg["concepts"]:
        for agg_func in cfg["agg_func"][concept]:
            output_file = f"{map_dir}{concept}_{agg_func}.csv"

            if not overwrite and os.path.exists(output_file):
                logger.info(f"Skipping {concept}_{agg_func} (already exists)")
                continue
            
            logger.info(f"Processing {concept} with {agg_func}")
            
            is_categorical = concept in cfg["dataset"]["ts_cat_names"]
            is_multi_label = concept in cfg["dataset"]["ts_categorical_multi_label"]
                        
            map_concept_optimized(
                cfg, 
                concept, 
                agg_func, 
                is_categorical, 
                is_multi_label
            )            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="ASTRA data pipeline")
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite existing intermediate files instead of skipping them",
    )
    args = parser.parse_args()
    overwrite = args.overwrite

    pm = ProjectManager()
    setup_logging()

    # Cohort mode
    if not overwrite and is_file_present(cfg['base_df_path']):
        base = pd.read_pickle(cfg['base_df_path'])
    else:
        base = bpi.create_base_df(cfg)

    # Pre-hospital data extraction (when enabled)
    if cfg.get("prehospital"):
        from astra.data.prehospital import run_prehospital_pipeline
        logger.info("Pre-hospital pipeline enabled — extracting PPJ data")
        base = run_prehospital_pipeline(cfg, base=base)
        # Re-save base_df with prehospital_start + ABCD columns
        ensure_parent_dir(cfg["base_df_path"])
        base.to_pickle(cfg["base_df_path"], protocol=4)
        logger.info(f"Updated base_df saved at {cfg['base_df_path']}")

    # bin_df (now uses prehospital_start if available)
    if not overwrite and is_file_present(cfg['bin_df_path']):
        pass
    else:
        bpi.create_bin_df(cfg, base=base)

    proces_inhospital_concepts(cfg, reset=overwrite)

    # Mark trauma text keywords on base (requires Notater.pkl from step above)
    if cfg.get("traumatext_config", {}).get("enabled", False):
        base = mark_traumatext(base, cfg)
        ensure_parent_dir(cfg["base_df_path"])
        base.to_pickle(cfg["base_df_path"], protocol=4)
        logger.info(f"Updated base_df with TRAUMATEXT columns at {cfg['base_df_path']}")

    map_data_optimized(cfg, overwrite=overwrite)
    logger.info("Creating TSDS")
    tsds = TSDS(cfg, base)
    logger.info(tsds.concepts.keys())
