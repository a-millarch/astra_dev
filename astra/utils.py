import os
import io
import base64
import hashlib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from rich.logging import RichHandler

pd.options.mode.chained_assignment = None

# Project root directory, derived from this file's location.
# Works regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

class ProjectManager:
    """
    Manages project directory settings in a compute instance environment.

    This class helps in setting and managing working directories within a 
    specific compute instance structure.

    Attributes:
        init_dir (str): Initial directory when the class is instantiated
        compute_name (str): Name of the compute instance extracted from the initial directory

    Methods:
        set_workdir(workdir): Changes the current working directory to a specified path

    Example:
        >>> pm = ProjectManager()
        >>> pm.set_workdir('my_project')
    """
    def __init__(self, workdir=None, set_workdir=True):

        self.init_dir = os.getcwd()
        self.workdir = self.init_dir
        self.compute_name = self.init_dir.split('clusters/')[1].split('/code')[0]
        if workdir:
            if set_workdir:
                self.set_workdir(workdir)
            else:
                self.absolute_workdir = self.return_workdir(workdir)
                
    def return_workdir(self, workdir):
        return f'/mnt/batch/tasks/shared/LS_root/mounts/clusters/{self.compute_name}/code/Users/{workdir}'
        
    def set_workdir(self, workdir):
        new_workdir = self.return_workdir(workdir)

        if os.path.exists(new_workdir):
            self.workdir = new_workdir
            os.chdir(new_workdir)
            print(f"Workdir: {new_workdir}")
        else:
            print(f"Directory does not exist: {new_workdir}")
        

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_logging_initialized = False


def setup_logging(level=logging.INFO, log_dir=None):
    """Configure the ``astra`` logger hierarchy.

    Call once at each application entry point (``train.py``,
    ``run_inference.py``, ``make_data.py``).  All module loggers created via
    ``logging.getLogger(__name__)`` under the ``astra`` package automatically
    inherit the handlers configured here.

    Args:
        level: Console log level (``logging.INFO`` or ``logging.DEBUG``).
        log_dir: Directory for the rotating DEBUG file log
            (``<log_dir>/astra.log``, daily rotation, 30-day retention).
            *None* (default) resolves to the ``ASTRA_LOG_DIR`` environment
            variable if set, else ``PROJECT_ROOT / 'logging'``.  Pass
            ``False`` (or set ``ASTRA_LOG_DIR=`` to empty) to disable file
            logging. File-handler setup failures (read-only dirs) degrade to
            console-only with a warning instead of crashing.
    """
    global _logging_initialized
    if _logging_initialized:
        return logging.getLogger('astra')

    root = logging.getLogger('astra')
    root.setLevel(logging.DEBUG)  # let handlers decide what to show

    # Remove bootstrap handler (or any prior handlers)
    for h in root.handlers[:]:
        root.removeHandler(h)
        h.close()

    # --- Console handler (Rich for colored output) ---
    # On narrow-encoding consoles (Windows cp1252), characters like '→' or
    # '✓' in log messages make the stream's strict encoder raise, and Rich
    # then dumps a "Logging error" traceback per record. Degrade unencodable
    # characters to '?' instead.
    import sys
    try:
        enc = (getattr(sys.stdout, 'encoding', '') or '').lower().replace('-', '')
        if enc != 'utf8' and hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(errors='replace')
    except (AttributeError, ValueError, OSError):
        pass
    console = RichHandler(
        markup=True,
        show_time=False,
        show_level=False,
        show_path=False,
        omit_repeated_times=False,
    )
    console.setLevel(level)
    console.setFormatter(logging.Formatter(
        '%(asctime)s\t%(levelname)-8s\t%(message)s\t%(filename)s:%(lineno)d',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    root.addHandler(console)

    # --- File handler (TimedRotatingFileHandler) ---
    # Default ON: no entry point ever passed log_dir, so the intended file
    # log never materialized. None -> ASTRA_LOG_DIR env var -> PROJECT_ROOT/logging.
    if log_dir is None:
        env_dir = os.environ.get('ASTRA_LOG_DIR')
        if env_dir is not None:
            log_dir = env_dir or False       # empty string disables
        else:
            log_dir = PROJECT_ROOT / 'logging'
    if log_dir:
        try:
            log_path = Path(log_dir)
            log_path.mkdir(parents=True, exist_ok=True)
            file_handler = TimedRotatingFileHandler(
                filename=str(log_path / 'astra.log'),
                when='midnight',
                interval=1,
                backupCount=30,
                encoding='utf-8',
            )
            file_handler.setLevel(logging.DEBUG)  # file always captures DEBUG
            file_handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s.%(funcName)s:%(lineno)d | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
            ))
            root.addHandler(file_handler)
            root.info("File log: %s (rotating daily, 30-day retention; "
                      "override with ASTRA_LOG_DIR)", log_path / 'astra.log')
        except OSError as e:
            root.warning("Could not set up file logging in %s (%s) — "
                         "console only. Set ASTRA_LOG_DIR to override.",
                         log_dir, e)

    # --- Suppress noisy third-party loggers ---
    for name in ('matplotlib', 'matplotlib.font_manager', 'optuna',
                 'urllib3', 'numba', 'PIL', 'fontTools'):
        logging.getLogger(name).setLevel(logging.WARNING)

    _logging_initialized = True
    return root


# Bootstrap: minimal console handler so logs aren't lost before an entry
# point calls setup_logging().  setup_logging() clears and replaces this.
_bootstrap_logger = logging.getLogger('astra')
if not _bootstrap_logger.handlers:
    _bootstrap_logger.setLevel(logging.INFO)
    _bh = RichHandler(
        markup=True, show_time=False, show_level=False,
        show_path=False, omit_repeated_times=False,
    )
    _bh.setFormatter(logging.Formatter(
        '%(asctime)s\t%(levelname)-8s\t%(message)s\t%(filename)s:%(lineno)d',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))
    _bootstrap_logger.addHandler(_bh)

# Module-level logger for functions in this file.
logger = logging.getLogger('astra')


def _resolve_fit_dpi(fig, fit_long_side_px, dpi_floor=50, dpi_ceiling=1200,
                     pad_inches=0.1):
    """Compute the highest DPI that renders *fig* within ``fit_long_side_px``.

    Calls ``fig.canvas.draw()`` first so ``constrained_layout`` (if enabled) has
    settled artist positions. Then takes the max of tight bbox and raw figsize as
    the long-side estimate (constrained_layout can expand beyond the tight bbox
    at save time). Adds ``2 * pad_inches`` to account for the padding matplotlib's
    ``bbox_inches='tight'`` applies on save. Result is clamped to [dpi_floor, dpi_ceiling].
    """
    width_in, height_in = fig.get_size_inches()
    # Force a draw so constrained_layout finalizes all artist positions before
    # we measure — without this, get_tightbbox returns stale/narrower values.
    try:
        fig.canvas.draw()
    except Exception:
        # best-effort: matplotlib draw may fail headless
        pass
    try:
        renderer = fig.canvas.get_renderer()
        tight = fig.get_tightbbox(renderer)
        tight_long_in = max(tight.width, tight.height)
    except Exception:
        tight_long_in = 0.0
    # Use whichever is larger: the measured tight bbox or the configured figsize.
    # constrained_layout can grow the figure beyond its initial figsize when
    # accommodating legends / outside text.
    long_in = max(tight_long_in, max(width_in, height_in))
    if long_in <= 0:
        return dpi_ceiling
    # savefig with bbox_inches='tight' adds pad_inches on each side (default 0.1).
    long_in_padded = long_in + 2 * pad_inches
    target = int(fit_long_side_px // long_in_padded)
    return max(dpi_floor, min(dpi_ceiling, target))


def _check_figure_size_limits(png_path, max_long_side_px=None, max_bytes=None):
    """Log a warning if saved PNG exceeds pixel/byte caps. Does not modify the file."""
    if max_long_side_px is not None:
        try:
            from PIL import Image
            with Image.open(png_path) as im:
                w, h = im.size
            long_side = max(w, h)
            if long_side > max_long_side_px:
                logger.warning(
                    f"{os.path.basename(png_path)}: longest side {long_side}px > cap {max_long_side_px}px "
                    f"(actual {w}x{h}). Consider lowering dpi or figsize."
                )
        except Exception as e:
            logger.debug(f"Could not check pixel size for {png_path}: {e}")
    if max_bytes is not None:
        try:
            size = os.path.getsize(png_path)
            if size > max_bytes:
                logger.warning(
                    f"{os.path.basename(png_path)}: {size / 1e6:.2f} MB > cap {max_bytes / 1e6:.2f} MB."
                )
        except OSError:
            pass


def save_figure(fig, filename, save_dir='reports/studyfigs', dpi=1200,
                max_long_side_px=None, max_bytes=None, fit_long_side_px=None):
    """Save *fig* as PNG + base64 text sidecar.

    Args:
        fig: matplotlib Figure
        filename: stem (no extension)
        save_dir: output directory for the PNG; base64 goes to ``<save_dir>/base64/``
        dpi: rasterization DPI (default 1200 for legacy callers).
        max_long_side_px: warn when the saved PNG's longest side exceeds this.
        max_bytes: warn when the saved PNG size exceeds this.
        fit_long_side_px: if set, compute the largest DPI that keeps the tight-bbox
            output within this pixel budget, overriding ``dpi``. Preferred for
            journal submissions — picks the highest sharpness that still fits the cap.
    """
    import matplotlib.pyplot as plt

    if fit_long_side_px is not None:
        dpi = _resolve_fit_dpi(fig, fit_long_side_px)
        logger.debug(f"{filename}: auto-dpi={dpi} to fit {fit_long_side_px}px cap")

    os.makedirs(save_dir, exist_ok=True)
    png_path = os.path.join(save_dir, f'{filename}.png')
    fig.savefig(png_path, dpi=dpi, bbox_inches='tight')

    # If auto-fit still overshoots (constrained_layout expanded the figure
    # more than we estimated), re-save with a corrected DPI. Hard guarantees
    # the pixel cap regardless of layout engine behaviour.
    if fit_long_side_px is not None:
        try:
            from PIL import Image
            with Image.open(png_path) as im:
                actual_long = max(im.size)
            if actual_long > fit_long_side_px:
                corrected_dpi = max(10, int(dpi * fit_long_side_px / actual_long))
                logger.debug(
                    f"{filename}: re-saving at dpi={corrected_dpi} "
                    f"(initial {dpi} produced {actual_long}px, over {fit_long_side_px}px cap)"
                )
                fig.savefig(png_path, dpi=corrected_dpi, bbox_inches='tight')
                dpi = corrected_dpi
        except Exception as e:
            logger.debug(f"Could not verify/correct size for {png_path}: {e}")

    # Save base64 version (same bytes as the PNG on disk)
    base64_dir = os.path.join(save_dir, 'base64')
    os.makedirs(base64_dir, exist_ok=True)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
    base64_path = os.path.join(base64_dir, f'{filename}_base64.txt')
    with open(base64_path, 'w') as f:
        f.write(base64_image)

    _check_figure_size_limits(png_path, max_long_side_px, max_bytes)
    plt.close(fig)

def save_base64(fig, save_path, dpi=1200, max_long_side_px=None, max_bytes=None,
                fit_long_side_px=None):
    """Save a base64 version of *fig* alongside *save_path* in a ``base64/`` sibling dir."""
    if fit_long_side_px is not None:
        dpi = _resolve_fit_dpi(fig, fit_long_side_px)
    parent = os.path.dirname(save_path)
    stem = Path(save_path).stem
    base64_dir = os.path.join(parent, 'base64')
    os.makedirs(base64_dir, exist_ok=True)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=dpi, bbox_inches='tight')
    buffer.seek(0)
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    with open(os.path.join(base64_dir, f'{stem}_base64.txt'), 'w') as f:
        f.write(b64)
    if os.path.exists(save_path):
        _check_figure_size_limits(save_path, max_long_side_px, max_bytes)

def ensure_parent_dir(path):
    """Create parent directory of *path* if it does not exist."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

def get_cfg(cfg_path=None):
    import yaml
    if cfg_path is None:
        cfg_path = PROJECT_ROOT / "configs" / "defaults.yaml"
    with open(cfg_path) as file:
        return yaml.safe_load(file)

cfg = get_cfg()

def count_csv_rows(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        next(f)  # skip header
        return sum(1 for line in f)
        
def get_base_df(base_df_path=None):
    if base_df_path is None:
        base_df_path = cfg["base_df_path"]
    return pd.read_pickle(base_df_path)


def get_bin_df(bin_df_path=None):
    if bin_df_path is None:
        bin_df_path = cfg["bin_df_path"]
    bin_df = pd.read_pickle(bin_df_path)
    expected_freqs = set(cfg["bin_intervals"].values())
    actual_freqs = set(bin_df["bin_freq"].unique())
    if expected_freqs != actual_freqs:
        raise ValueError(
            f"bin_df is stale — frequencies don't match cfg['bin_intervals']. "
            f"Expected: {sorted(expected_freqs)}, Got: {sorted(actual_freqs)}. "
            f"Regenerate bin_df with the current bin_intervals config."
        )
    return bin_df


def get_train_test_split(cfg, base_df=None, return_indices=False):
    """
    Get train/test split using the same strategy as the hybrid model.

    This ensures both EBM and hybrid models use the exact same patients for evaluation.

    Args:
        cfg: Configuration dictionary
        base_df: Base dataframe (if None, will load using get_base_df())
        return_indices: If True, return indices instead of dataframes

    Returns:
        If return_indices=False: (train_df, test_df)
        If return_indices=True: (train_indices, test_indices)
    """
    if base_df is None:
        base_df = get_base_df()

    # Apply exclusion criteria from config profile
    from astra.data.datasets import resolve_exclusion_criteria, apply_exclusion_criteria
    criteria = resolve_exclusion_criteria(cfg)
    if criteria:
        base_df = apply_exclusion_criteria(base_df, criteria)

    # Sort by date to ensure temporal ordering
    base_df = base_df.sort_values('ServiceDate').reset_index(drop=True)

    # Get split strategy from config
    holdout_type = cfg.get("holdout_type", "temporal")

    if holdout_type == "temporal":
        # Use temporal split based on date
        split_date = pd.to_datetime(cfg["holdout_split_date"])

        train_mask = base_df.ServiceDate <= split_date
        test_mask = base_df.ServiceDate > split_date

        if return_indices:
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            return train_indices, test_indices
        else:
            train_df = base_df[train_mask].copy()
            test_df = base_df[test_mask].copy()
            return train_df, test_df

    elif holdout_type == "random":
        # Random split (stratified by target if possible)
        holdout_fraction = cfg.get("holdout_fraction", 0.2)
        seed = cfg.get("seed", 2024)

        try:
            # Try stratified split
            train_df, test_df = train_test_split(
                base_df,
                test_size=holdout_fraction,
                random_state=seed,
                stratify=base_df[cfg["target"]],
                shuffle=True
            )
        except ValueError:
            # Fall back to non-stratified if not enough samples
            train_df, test_df = train_test_split(
                base_df,
                test_size=holdout_fraction,
                random_state=seed,
                shuffle=True
            )

        if return_indices:
            train_indices = train_df.index.values
            test_indices = test_df.index.values
            return train_indices, test_indices
        else:
            return train_df, test_df

    else:
        raise ValueError(f"Unknown holdout_type: {holdout_type}. Use 'temporal' or 'random'.")


def align_dataframes(df_a, df_b, fill_value=0.0):
    """
    Aligns two dataframes by adding missing columns to both and ensuring the same column order.
    Non-numeric column names come first, followed by numeric column names sorted in ascending order.

    Args:
    df_a (pd.DataFrame): First dataframe
    df_b (pd.DataFrame): Second dataframe
    fill_value (float): Value to fill in new columns (default: 0.0)

    Returns:
    tuple: (aligned_df_a, aligned_df_b)
    """

    # Get the union of all columns
    all_columns = set(df_a.columns) | set(df_b.columns)

    # Identify missing columns in each dataframe
    missing_in_a = all_columns - set(df_a.columns)
    missing_in_b = all_columns - set(df_b.columns)

    # Add missing columns to df_a
    for col in missing_in_a:
        df_a[col] = fill_value

    # Add missing columns to df_b
    for col in missing_in_b:
        df_b[col] = fill_value

    # Separate columns into non-numeric and numeric
    non_numeric_columns = [col for col in all_columns if not col.isdigit()]
    numeric_columns = sorted([col for col in all_columns if col.isdigit()], key=int)

    # Combine the columns with non-numeric first, followed by numeric
    sorted_columns = non_numeric_columns + numeric_columns

    # Reorder columns in both dataframes
    df_a_aligned = df_a.reindex(columns=sorted_columns)
    df_b_aligned = df_b.reindex(columns=sorted_columns)

    return df_a_aligned, df_b_aligned


def create_enumerated_id(df, string_col, datetime_col):
    # Combine the string and datetime columns to create a unique identifier
    df["unique_id"] = df[string_col].astype(str) + df[datetime_col].astype(str)

    # Drop duplicates to ensure each unique combination is only considered once
    df_unique = df.drop_duplicates(subset=["unique_id"]).copy()

    # Create an enumerated ID column
    df_unique["PID"] = range(1, len(df_unique) + 1)

    # Merge the new ID column back into the original DataFrame
    df = df.merge(df_unique[["unique_id", "PID"]], on="unique_id", how="left")

    # Drop the unique_id column as it's no longer needed
    df.drop(columns=["unique_id"], inplace=True)

    return df


def make_inference_pid(cpr_hash: str, service_date) -> str:
    """Deterministic PID from CPR_hash + ServiceDate for deployment use."""
    date_str = str(service_date)[:10].replace('-', '')
    return str(cpr_hash)[:8] + date_str


def mark_keywords_in_df(
    df,
    text_column,
    keywords,
    timestamp=None,
    base_timestamp=None,
    t_delta=12,
    new_column="keyword_present",
):
    """
    Adds a new column to the DataFrame marking whether any of the keywords are present in the text column
    and checks if two datetime columns are within 12 hours apart.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the text and datetime data.
    text_column (str): The name of the column with natural language text.
    keywords (list): A list of keywords to search for.
    datetime_col1 (str): The first datetime column.
    datetime_col2 (str): The second datetime column.
    new_column (str): The name of the new column to be added. Defaults to 'keyword_present'.

    Returns:
    pd.DataFrame: The original DataFrame with an additional column marking keyword presence and datetime condition.
    """
    # Join the list of keywords into a regex pattern
    keyword_pattern = "|".join(keywords)

    # Check if keywords are present in the text column
    df[new_column] = df[text_column].str.contains(keyword_pattern, case=False, na=False)

    if timestamp and base_timestamp:
        # Check if the two datetime columns are within 12 hours of each other
        df[timestamp] = pd.to_datetime(df[timestamp], errors="coerce")
        df[base_timestamp] = pd.to_datetime(df[base_timestamp], errors="coerce")

        df[f"within_{t_delta}_hours"] = (
            df[timestamp] - df[base_timestamp]
        ) <= pd.Timedelta(hours=t_delta)
    return df


def expand_datetime_rows(df):
    """WIP for medicine"""
    # Convert start and end columns to datetime
    df["start"] = pd.to_datetime(df["start"])
    df["end"] = pd.to_datetime(df["end"])

    # Reduce dimensions
    df = (
        df[["PID", "TIMESTAMP", "FEATURE", "VALUE", "start", "end"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    logger.info(f" df shape after reducing dimensions: {df.shape}")
    # Now, we want a row per minute from administration to seponation

    expanded_rows = []
    for _, row in df.iterrows():
        if pd.notnull(row["end"]):
            # Create a date range for each minute between start and end
            date_range = pd.date_range(start=row["start"], end=row["end"], freq="min")

        else:
            # If end is NaN, create a single-element range with just the start time
            date_range = pd.DatetimeIndex([row["start"]])

        # Create a new DataFrame for the current row's expanded data
        expanded_data = pd.DataFrame(
            {
                "PID": row["PID"],
                "TIMESTAMP": date_range,
                "VALUE": row["VALUE"],
                "FEATURE": row["FEATURE"],
            }
        )

        # Append to the list
        expanded_rows.append(expanded_data)

    # Concatenate all expanded DataFrames into a single DataFrame
    final_df = pd.concat(expanded_rows, ignore_index=True)
    return final_df.reset_index(drop=True)


def stratified_split_dataframe(df, target_column, test_size=0.2, random_state=None):
    """
    Splits a DataFrame into training and validation sets while retaining
    the same proportion of positive outcomes in the target column.

    Parameters:
    - df (pd.DataFrame): The input DataFrame to split.
    - target_column (str): The name of the target column in the DataFrame.
    - test_size (float): The proportion of the dataset to include in the validation set.
    - random_state (int, optional): Controls the randomness of the split.

    Returns:
    - train_df (pd.DataFrame): The training dataset.
    - val_df (pd.DataFrame): The validation dataset.
    """

    # Define features and target
    X = df.drop(target_column, axis=1)  # Features
    y = df[target_column]  # Target variable

    # Perform stratified split
    _, val_indices = train_test_split(
        df.index, test_size=test_size, stratify=y, random_state=random_state
    )

    # Create a HOLDOUT column
    df["HOLDOUT"] = False
    df.loc[val_indices, "HOLDOUT"] = True

    return df


def md5_checksum(file_path):
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def find_columns_with_word(dataframe, word):
    matching_columns = [col for col in dataframe.columns if word in col]
    return matching_columns


def ensure_datetime(df, column_name):
    if not pd.api.types.is_datetime64_any_dtype(df[column_name]):
        # df = df[df[column_name].notnull()]
        df[column_name] = pd.to_datetime(df[column_name], errors="coerce")
    return df


def is_file_present(file_path: str):
    return os.path.isfile(file_path)


def are_files_present(directory, filenames, extension):
    return all(
        os.path.isfile(os.path.join(directory, f"{filename}{extension}"))
        for filename in filenames
    )


def clear_mem():
    import torch
    import gc

    # Delete: any unused Python objects
    gc.collect()

    # Clear the PyTorch CUDA cache
    torch.cuda.empty_cache()

    # Total memory
    total_memory = torch.cuda.get_device_properties(0).total_memory

    # Allocated memory
    allocated_memory = torch.cuda.memory_allocated(0)

    # Cached memory
    cached_memory = torch.cuda.memory_reserved(0)

    # Free memory
    free_memory = total_memory - allocated_memory - cached_memory

    print(f"Total GPU memory: {total_memory / 1e9:.2f} GB")
    print(f"Allocated GPU memory: {allocated_memory / 1e9:.2f} GB")
    print(f"Cached GPU memory: {cached_memory / 1e9:.2f} GB")
    print(f"Free GPU memory: {free_memory / 1e9:.2f} GB")


# -----------------------------------------------------------------------------------------------------------
# Common tools
def inches_to_cm(inches):
    return inches * 2.54


def feet_to_cm(feet):
    return feet * 30.48


def pounds_to_kg(pounds):
    return pounds * 0.45359237


def ounces_to_kg(ounces):
    return ounces * 0.0283495231


def dict_to_list(input_dict):
    result = []
    for i, v in input_dict.items():
        for item in v:
            result.append(item)
    return result


def list_from_col(df, col_name, target_col="col_cat", return_col="TQIP_name"):
    return df[df[target_col] == col_name][return_col].tolist()



def get_concept(concept, cfg) -> dict:
    """get concept from name"""

    drop_cols = cfg["drop_features"][concept]
    ts_cat_names = cfg.get("dataset", {}).get("ts_cat_names", [])
    concept_dict = {}

    for agg_func in cfg["agg_func"][concept]:
        logger.debug(f"Loading {concept}.agg_func: {agg_func}")
        df = pd.read_csv(f"data/interim/mapped/{concept}_{agg_func}.csv")
        # if features to drop, drop now.
        if concept in ts_cat_names:
            # Categorical: FEATURE is constant (e.g. "ADT"), filter on VALUE
            if drop_cols:
                df = df[~df.VALUE.isin(drop_cols)]
        else:
            try:
                df = df[~df.FEATURE.isin(drop_cols + [np.nan])]
            except Exception:
                logger.debug(f"drop_features filter failed for {concept}; dropping NaN FEATUREs only")
                df = df[~df.FEATURE.isin([np.nan])]
        df["VALUE"] = pd.to_numeric(df["VALUE"], errors="coerce")
        concept_dict[agg_func] = df
    return concept_dict

#!TODO Check if obsolete below
# _______________________________________________________________
# Transformation tools


def get_delta_time(base_df, input_df, base_df_date, input_df_date, mode="days"):
    """
    Returns input dataframe with a column containing time difference between the specified datetime variable and the base dataframe datetime variable.
    Use mode parameter to calculate difference in either days, hours or minutes.

    Parameters
    ----------
    base_df: pandas dataframe
        dataframe containing baseline datetime variable (t0)
        e.g. start of surgery

    input_df: pandas dataframe
        dataframe containing datetime variable (tx) for the event
        e.g. datetime of diagnosis

    base_df_date: str
        name of column for base_df datetimevariable.

    input_df_date: str
        name of column for inpu_df datetimevariable.

    mode: str
        'days', 'hours' or 'minutes'
    """

    input_df = input_df[input_df[input_df_date].notnull()].copy(deep=True)
    input_df[input_df_date] = pd.to_datetime(input_df[input_df_date])

    # input_df.loc[:, input_df_date] = pd.to_datetime(input_df[input_df_date], errors = 'coerce')
    base_df[base_df_date] = pd.to_datetime(base_df[base_df_date])

    df = input_df.merge(base_df[["CPR_hash", base_df_date]], on="CPR_hash", how="left")

    if mode.lower() in ["days", "d"]:
        df.loc[:, "delta_days"] = (df[input_df_date] - df[base_df_date]).dt.days

    elif mode.lower() in ["hours", "h"]:
        df.loc[:, "delta_hours"] = (df[input_df_date] - df[base_df_date]).dt.hour

    elif mode.lower() in ["minutes", "m"]:
        df.loc[:, "delta_minutes"] = (df[input_df_date] - df[base_df_date]).astype(
            "timedelta64[m]"
        )

    else:
        print("wrong mode. Use days, hours or minutes")

    return df


def remove_solved(df, df_date_col_name, solved_date="Løst_dato", solved_before=0):
    df.loc[:, solved_date] = pd.to_datetime(df[solved_date], errors="coerce")
    df.drop(
        df[
            df[solved_date] + pd.DateOffset(solved_before) <= df[df_date_col_name]
        ].index,
        inplace=True,
    )
    return df


def convert_numeric_col(df, num_col, var_name, conv_factor, decimals):
    # df = df[pd.to_numeric(df[num_col], errors='coerce').notnull()]
    try:
        df.loc[:, var_name] = (df[num_col].astype(float) * conv_factor).round(decimals)
    except Exception:
        logger.debug(f"Numeric conversion failed for {num_col}; using raw values for {var_name}")
        df.loc[:, var_name] = df[num_col]
    return df


def filt_delta_var(df, min_val=None, max_val=None, delta_var="days"):
    delta_var = "delta_" + delta_var
    if max_val == None:
        df = df[df[delta_var] >= min_val].sort_values(by=["CPR_hash", delta_var])
    elif min_val == None:
        df = df[df[delta_var] <= max_val].sort_values(by=["CPR_hash", delta_var])
    elif min_val != None and max_val != None:
        df = df[df[delta_var].between(min_val, max_val, inclusive="both")].sort_values(
            by=["CPR_hash", delta_var]
        )
    else:
        print("Specify at least either min or max value")
    return df


def get_latest_var(df, time_var, delta_var="days"):
    delta_var = "delta_" + delta_var
    df.loc[:, "delta_abs"] = df[delta_var].abs()
    df.sort_values(["CPR_hash", "delta_abs"], inplace=True)
    df.drop_duplicates(["CPR_hash", time_var], inplace=True)
    df.drop(columns=["delta_abs"], inplace=True)
    return df


def add_to_base_df(base_df, base_df_date_col_name, input_df, col_name, to_binary=False):
    """
    Adds a variable from input_df to base_df, either as (pseudo) binary or value as is.

    Parameters
    ----------
    col_name: str
        name for the new column to be added
    """

    # in case of using this function with input_df created/filtered in a loop
    if input_df.empty:
        base_df.loc[:, col_name] = 0
        return base_df

    else:
        if to_binary:
            input_df.loc[:, col_name] = 1
            output_df = base_df.merge(
                input_df[["CPR_hash", base_df_date_col_name, col_name]],
                on=["CPR_hash", base_df_date_col_name],
                how="left",
            ).drop_duplicates()
            output_df[col_name].fillna(0, inplace=True)
            output_df.loc[:, col_name] = output_df[col_name].astype(int)

        else:
            # base_df.drop(columns = col_name, inplace = True, errors='ignore')
            output_df = base_df.merge(
                input_df[["CPR_hash", base_df_date_col_name, col_name]],
                how="left",
                on=["CPR_hash", base_df_date_col_name],
            )

        return output_df


## High level transformation
def add_categorical_variable(
    base_df,
    df,
    filt_date,
    var_name,
    base_df_date_col_name,
    df_date_col_name=None,
    filt_column=None,
    filt_val=None,
    solved_date=None,
    solved_before=0,
    min_val=0,
    max_val=None,
    delta_var="days",
):
    """
    Adding a categorical variable to base_df from df in the following steps:

    - Filtering df columns by filt_var (tupple or string)
    - Getting delta_time columns (mode can either be days, hours or minutes)
    - Filter df based on delta_time rules (e.g only -10 to +10 hours from baseline datetime)
    - Filter df to get the row closest to baseline datetime
    - Adds the variable to base_df and returns

    Parameters
    ----------
    base_df : pandas dataframe
        target df where new variable should be added

    df: pandas dataframe
        df containing the new variable

    filt_date: str
        colunmn name of df datetime used for calculating delta days

    var_name: str
        name for new continous column in base_df

    filt_column: str
        column name of df to be filtered (e.g "fruit_type")

    filt_val: tuple | string
        inclusive values to filt_column (e.g ("apple", "banana", "mango") )
    """

    # filtering df. Try tuple else string
    try:
        # backup_df = df.copy(deep=True)
        df = df[df[filt_column].str.startswith(filt_val, na=False)]
    except Exception:
        logger.debug(f"startswith filter failed for {filt_column}; falling back to equality match")
        df = df[df[filt_column] == filt_val]

    if df.empty:
        base_df.loc[:, var_name] = 0
        return base_df

    else:
        df[filt_date] = pd.to_datetime(df[filt_date])
        base_df[base_df_date_col_name] = pd.to_datetime(base_df[base_df_date_col_name])

        df = get_delta_time(
            base_df,
            df,
            base_df_date=base_df_date_col_name,
            input_df_date=filt_date,
            mode=delta_var,
        )

        if solved_date != None:
            df = remove_solved(df, base_df_date_col_name, solved_date, solved_before)

        df = filt_delta_var(df, min_val, max_val, delta_var)

        df = get_latest_var(df, base_df_date_col_name, delta_var)

        df = add_to_base_df(
            base_df, base_df_date_col_name, df, var_name, to_binary=True
        )

        return df


def add_continous_variable(
    base_df,
    df,
    filt_column,
    filt_val,
    filt_date,
    num_col,
    var_name,
    base_df_date_col_name,
    conv_factor=None,
    decimals=2,
    min_val=0,
    max_val=None,
    delta_var="days",
):
    """
    Adding a continous variable to base_df from df in the following steps:

    - Filtering df columns by filt_var (tupple or string)
    - Getting delta_time columns (mode can either be days, hours or minutes)
    - Filter df based on delta_time rules (e.g only -10 to +10 hours from baseline datetime)
    - Filter df to get the row closest to baseline datetime
    - Convert the numerical value and round decimals if needed
    - Adds the variable to base_df and returns

    Parameters
    ----------
    base_df : pandas dataframe
        target df where new variable should be added

    df: pandas dataframe
        df containing the new variable

    filt_column: str
        column name of df to be filtered (e.g "fruit_type")

    filt_var: tuple | string
        inclusive values to filt_column (e.g ("apple", "banana", "mango") )

    filt_date: str
        colunmn name of df datetime used for calculating delta days

    num_col: str
        column name of the numerical/continous variable

    var_name: str
        name for new continous column in base_df
    """
    # filtering df. Try tuple else string
    try:
        df = df[df[filt_column].isin(filt_val)]
    except Exception:
        logger.debug(f"isin filter failed for {filt_column}; falling back to equality match")
        df = df[df[filt_column] == filt_val]

    df[filt_date] = pd.to_datetime(df[filt_date])
    base_df[base_df_date_col_name] = pd.to_datetime(base_df[base_df_date_col_name])
    # collecting delta_time column
    df = get_delta_time(
        base_df,
        df,
        base_df_date=base_df_date_col_name,
        input_df_date=filt_date,
        mode=delta_var,
    )

    # filtering on delta_var
    df = filt_delta_var(df, min_val, max_val, delta_var)

    # latest, min, max, mean?
    df = get_latest_var(df, base_df_date_col_name, delta_var=delta_var)

    df = convert_numeric_col(df, num_col, var_name, conv_factor, decimals)
    df = add_to_base_df(base_df, base_df_date_col_name, df, var_name, to_binary=False)

    return df