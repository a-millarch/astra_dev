import logging
import os

import pandas as pd
import numpy as np

from astra.utils import ProjectManager, cfg, setup_logging
from astra.utils import is_file_present, are_files_present

logger = logging.getLogger(__name__)

from astra.data.collectors import collect_subsets
import astra.data.build_patient_info as bpi
from astra.data.filters import filter_subsets_inhospital
from astra.data.mapper import map_concept, map_concept_optimized
from astra.data.notes_features import build_notes_features

from astra.data.datasets import TSDS

def generate_base_df():
    # JUST A TEMPORARY TESTER FUNCTION, used by load_or_collect_population
    
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

def map_data_optimized(cfg):
    """Updated to use optimized mapper."""
    logger.info("Mapping data to bins")
    map_dir = "data/interim/mapped/"

    for concept in cfg["concepts"]:
        for agg_func in cfg["agg_func"][concept]:
            output_file = f"{map_dir}{concept}_{agg_func}.csv"

            if os.path.exists(output_file):
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


def _forward_fill_concept(cfg: dict, concept: str) -> None:
    """Forward-fill time columns in mapped concept pickle.

    Ensures semi-static features (ISS, INTUBATED) propagate forward from
    first observation. ffill on axis=1 is inherently forward-only.
    """
    map_dir = "data/interim/mapped/"
    for agg_func in cfg["agg_func"][concept]:
        path = f"{map_dir}{concept}_{agg_func}.pkl"
        if not os.path.exists(path):
            logger.warning(f"Forward-fill skipped: {path} not found")
            continue

        df = pd.read_pickle(path)
        time_cols = [c for c in df.columns if c not in ("PID", "FEATURE")]
        df[time_cols] = df[time_cols].ffill(axis=1)
        df.to_pickle(path, protocol=4)
        logger.info(f"Forward-filled {concept}_{agg_func}")

if __name__ =='__main__':
    pm = ProjectManager()
    setup_logging()
    #Single patient loop
    #generate_base_df() #Simulates new patient drop
    #population = bpi.load_or_collect_population(cfg)
    #proces_raw_concepts(cfg, base=population)

    # Cohort mode
    if is_file_present(cfg['base_df_path']):
        pass
    else:
        base = bpi.create_base_df(cfg)
    # bin_df    
    if is_file_present(cfg['bin_df_path']):
        pass
    else: 
        bpi.create_bin_df(cfg)



    proces_inhospital_concepts(cfg, reset=False)

    # Build features from notes (GCS, ISS, Intubation)
    if not is_file_present("data/interim/concepts/TraumaAssessment.pkl"):
        logger.info("Building features from notes...")
        build_notes_features()
    else:
        logger.info("TraumaAssessment.pkl already exists, skipping")

    #map_data(cfg)
    map_data_optimized(cfg)

    # Forward-fill TraumaAssessment (semi-static features)
    _forward_fill_concept(cfg, "TraumaAssessment")

    logger.info("Creating TSDS")
    tsds = TSDS(cfg, base)
    logger.info(tsds.concepts.keys())
    
    
   