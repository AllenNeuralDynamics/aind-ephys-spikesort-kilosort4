import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# GENERAL IMPORTS
import os
import sys
import argparse
import numpy as np
from pathlib import Path
import shutil
import json
import time
from pprint import pprint
import logging
from datetime import datetime, timedelta

# SPIKEINTERFACE
import spikeinterface as si
import spikeinterface.extractors as se
import spikeinterface.sorters as ss
import spikeinterface.curation as sc

# AIND
from aind_data_schema.core.processing import DataProcess

try:
    from aind_log_utils import log
    HAVE_AIND_LOG_UTILS = True
except ImportError:
    HAVE_AIND_LOG_UTILS = False

# LOCAL
URL = "https://github.com/AllenNeuralDynamics/aind-ephys-spikesort-kilosort4"
VERSION = "1.0"

SORTER_NAME = "kilosort4"

data_folder = Path("../data")
results_folder = Path("../results")
scratch_folder = Path("../scratch")

# Define argument parser
parser = argparse.ArgumentParser(description="Spike sort ecephys data with Kilosort4")

raise_if_fails_group = parser.add_mutually_exclusive_group()
raise_if_fails_help = "Whether to raise an error in case of failure or continue. Default True (raise)"
raise_if_fails_group.add_argument("--raise-if-fails", action="store_true", help=raise_if_fails_help)
raise_if_fails_group.add_argument("static_raise_if_fails", nargs="?", default="true", help=raise_if_fails_help)

skip_motion_correction_group = parser.add_mutually_exclusive_group()
skip_motion_correction_help = "Whether to skip Kilosort motion correction. Default: False"
skip_motion_correction_group.add_argument("--skip-motion-correction", action="store_true", help=skip_motion_correction_help)
skip_motion_correction_group.add_argument("static_skip_motion_correction", nargs="?", help=skip_motion_correction_help)

min_drift_channels_group = parser.add_mutually_exclusive_group()
min_drift_channels_help = (
    "Minimum number of channels to enable Kilosort motion correction. Default is 96."
)
min_drift_channels_group.add_argument("static_min_channels_for_drift", nargs="?", help=min_drift_channels_help)
min_drift_channels_group.add_argument("--min-drift-channels", default="96", help=min_drift_channels_help)

clear_cache_group = parser.add_mutually_exclusive_group()
clear_cache_group_help = (
    "Force pytorch to free up memory reserved for its cache in between memory-intensive operations. "
    "Note that setting `clear_cache=True` is NOT recommended unless you encounter GPU out-of-memory errors, "
    "since this can result in slower sorting."
)
clear_cache_group.add_argument("--clear-cache", action="store_true", help=clear_cache_group_help)
clear_cache_group.add_argument("static_clear_cache", nargs="?", default="false", help=clear_cache_group_help)

n_jobs_group = parser.add_mutually_exclusive_group()
n_jobs_help = (
    "Number of jobs to use for parallel processing. Default is -1 (all available cores). "
    "It can also be a float between 0 and 1 to use a fraction of available cores"
)
n_jobs_group.add_argument("static_n_jobs", nargs="?", default="-1", help=n_jobs_help)
n_jobs_group.add_argument("--n-jobs", default="-1", help=n_jobs_help)

parser.add_argument("--params", default=None, help="Path to the parameters file or JSON string. If given, it will override all other arguments.")


if __name__ == "__main__":
    args = parser.parse_args()

    PARAMS = args.params

    if PARAMS is not None:
        if Path(PARAMS).is_file():
            with open(PARAMS, "r") as f:
                spikesorting_params = json.load(f)
        else:
            spikesorting_params = json.loads(PARAMS)
        SKIP_MOTION_CORRECTION = spikesorting_params.pop("skip_motion_correction", False)
        MIN_DRIFT_CHANNELS = spikesorting_params.pop("min_drift_channels", 96)
        RAISE_IF_FAILS = spikesorting_params.pop("raise_if_fails", True)
        CLEAR_CACHE = spikesorting_params.pop("clear_cache", False)
    else:
        SKIP_MOTION_CORRECTION = True if args.static_skip_motion_correction and args.static_skip_motion_correction.lower() == "true" else args.skip_motion_correction
        MIN_DRIFT_CHANNELS = args.static_min_channels_for_drift or args.min_drift_channels
        MIN_DRIFT_CHANNELS = int(MIN_DRIFT_CHANNELS)
        RAISE_IF_FAILS = True if args.static_raise_if_fails and args.static_raise_if_fails.lower() == "true" else args.raise_if_fails
        CLEAR_CACHE = True if args.static_clear_cache and args.static_clear_cache.lower() == "true" else args.clear_cache

    N_JOBS = args.static_n_jobs or args.n_jobs
    N_JOBS = int(N_JOBS) if not N_JOBS.startswith("0.") else float(N_JOBS)
    # Use CO_CPUS env variable if available
    N_JOBS_CO = os.getenv("CO_CPUS")
    N_JOBS = int(N_JOBS_CO) if N_JOBS_CO is not None else N_JOBS

    # look for subject and data_description JSON files
    subject_id = "undefined"
    session_name = "undefined"
    for f in data_folder.iterdir():
        # the file name is {recording_name}_subject.json
        if "subject.json" in f.name:
            with open(f, "r") as file:
                subject_id = json.load(file)["subject_id"]
        # the file name is {recording_name}_data_description.json
        if "data_description.json" in f.name:
            with open(f, "r") as file:
                session_name = json.load(file)["name"]

    if HAVE_AIND_LOG_UTILS:
        log.setup_logging(
            "Spikesort Kilosort4 Ecephys",
            subject_id=subject_id,
            asset_name=session_name,
        )
    else:
        logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")

    data_process_prefix = "data_process_spikesorting"

    job_kwargs = spikesorting_params.pop("job_kwargs")
    job_kwargs["n_jobs"] = N_JOBS
    si.set_global_job_kwargs(**job_kwargs)

    sorter_params = spikesorting_params["sorter"]

    ####### SPIKESORTING ########
    logging.info(f"\n\nSPIKE SORTING WITH {SORTER_NAME.upper()}\n")

    logging.info(f"\tRAISE_IF_FAILS: {RAISE_IF_FAILS}")
    logging.info(f"\tSKIP_MOTION_CORRECTION: {SKIP_MOTION_CORRECTION}")
    logging.info(f"\tMIN_DRIFT_CHANNELS: {MIN_DRIFT_CHANNELS}")
    logging.info(f"\tN_JOBS: {N_JOBS}")

    sorting_params = None

    si.set_global_job_kwargs(**job_kwargs)
    t_sorting_start_all = time.perf_counter()

    # check if test
    if (data_folder / "preprocessing_pipeline_output_test").is_dir():
        logging.info("\n*******************\n**** TEST MODE ****\n*******************\n")
        preprocessed_folder = data_folder / "preprocessing_pipeline_output_test"
    else:
        preprocessed_folder = data_folder

    # try results here
    spikesorted_raw_output_folder = scratch_folder / "spikesorted_raw"
    spikesorting_data_processes = []

    preprocessed_folders = [p for p in preprocessed_folder.iterdir() if p.is_dir() and "preprocessed_" in p.name]
    for recording_folder in preprocessed_folders:
        datetime_start_sorting = datetime.now()
        t_sorting_start = time.perf_counter()
        spikesorting_notes = ""

        recording_name = ("_").join(recording_folder.name.split("_")[1:])
        binary_json_file = preprocessed_folder / f"binary_{recording_name}.json"
        binary_pickle_file = preprocessed_folder / f"binary_{recording_name}.pkl"
        sorting_output_folder = results_folder / f"spikesorted_{recording_name}"
        sorting_output_process_json = results_folder / f"{data_process_prefix}_{recording_name}.json"

        logging.info(f"Sorting recording: {recording_name}")
        try:
            if binary_json_file.is_file():
                logging.info(f"Loading recording from binary JSON")
                recording = si.load(binary_json_file, base_folder=preprocessed_folder)
            elif binary_pickle_file.is_file():
                logging.info(f"Loading recording from binary PKL")
                recording = si.load(binary_pickle_file, base_folder=preprocessed_folder)
            else:
                recording = si.load(recording_folder)
            logging.info(recording)
        except Exception as e:
            logging.info(f"Skipping spike sorting for {recording_name}.")
            # create an empty result file (needed for pipeline)
            sorting_output_folder.mkdir(parents=True, exist_ok=True)
            error_file = sorting_output_folder / "error.txt"
            error_file.write_text("Too many bad channels")
            continue

        # we need to concatenate segments for KS
        split_segments = False
        if recording.get_num_segments() > 1:
            logging.info("Concatenating multi-segment recording")
            recording = si.concatenate_recordings([recording])
            split_segments = True

        if recording.get_num_channels() < MIN_DRIFT_CHANNELS:
            logging.info("Drift correction not enabled due to low number of channels")
            sorter_params["do_correction"] = False

        if SKIP_MOTION_CORRECTION:
            logging.info("Drift correction disabled")
            sorter_params["do_correction"] = False

        if CLEAR_CACHE:
            logging.info("Setting clear_cache to True")
            sorter_params["clear_cache"] = True

        # run ks4
        try:
            sorting = ss.run_sorter(
                SORTER_NAME,
                recording,
                output_folder=spikesorted_raw_output_folder / recording_name,
                verbose=False,
                delete_output_folder=False,
                remove_existing_folder=True,
                **sorter_params,
            )
            logging.info(f"\tRaw sorting output: {sorting}")
            n_original_units = int(len(sorting.unit_ids))
            spikesorting_notes += f"\n- KS4 found {n_original_units} units, "
            if sorting_params is None:
                sorting_params = sorting.sorting_info["params"]

            # safe delete the output folder
            try:
                shutil.rmtree(spikesorted_raw_output_folder / recording_name / "sorter_output")
            except Exception as e:
                logging.info(f"\tError deleting sorter output folder: {e}")

            # remove empty units
            sorting = sorting.remove_empty_units()
            # remove spikes beyond num_Samples (if any)
            sorting = sc.remove_excess_spikes(sorting=sorting, recording=recording)
            n_non_empty_units = int(len(sorting.unit_ids))
            n_empty_units = n_original_units - n_non_empty_units
            # save params in output
            sorting_outputs = dict(empty_units=n_empty_units)
            logging.info(f"\tSorting output without empty units: {sorting}")
            spikesorting_notes += f"{len(sorting.unit_ids)} after removing empty templates.\n"

            # split back to get original segments
            if split_segments:
                logging.info("Splitting sorting into multiple segments")
                sorting = si.split_sorting(sorting, recording)

            # save results
            logging.info(f"\tSaving results to {sorting_output_folder}")
            sorting = sorting.save(folder=sorting_output_folder)
            shutil.copy(
                spikesorted_raw_output_folder / recording_name / "spikeinterface_log.json", sorting_output_folder
            )
        except Exception as e:
            log_file = spikesorted_raw_output_folder / recording_name / "spikeinterface_log.json"
            with open(log_file, "r") as f:
                spike_sorter_log = json.load(f)
            logging.info("\n\tSPIKE SORTING FAILED!\nError log:\n")
            pprint(spike_sorter_log)
            if RAISE_IF_FAILS:
                raise Exception(e)
            else:
                # save log to results
                (sorting_output_folder).mkdir(parents=True, exist_ok=True)
                shutil.copy(log_file, sorting_output_folder)
                sorting_outputs = dict()
                sorting_params = dict()

        t_sorting_end = time.perf_counter()
        elapsed_time_sorting = np.round(t_sorting_end - t_sorting_start, 2)

        spikesorting_process = DataProcess(
            name="Spike sorting",
            software_version=VERSION,  # either release or git commit
            start_date_time=datetime_start_sorting,
            end_date_time=datetime_start_sorting + timedelta(seconds=np.floor(elapsed_time_sorting)),
            input_location=str(data_folder),
            output_location=str(results_folder),
            code_url=URL,
            parameters=sorting_params,
            outputs=sorting_outputs,
            notes=spikesorting_notes,
        )
        with open(sorting_output_process_json, "w") as f:
            f.write(spikesorting_process.model_dump_json(indent=3))

    t_sorting_end_all = time.perf_counter()
    elapsed_time_sorting_all = np.round(t_sorting_end_all - t_sorting_start_all, 2)
    logging.info(f"SPIKE SORTING time: {elapsed_time_sorting_all}s")
