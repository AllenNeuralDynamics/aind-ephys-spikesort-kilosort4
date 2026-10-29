# Spike sorting with Kilosort4 for AIND ephys pipeline
## aind-ephys-spikesort-kilosort4


### Description

This capsule is designed to spike sort ephys data using [Kilosort4](https://github.com/MouseLand/Kilosort/) for the AIND pipeline.

This capsule spike sorts preprocessed ephys stream and applies a minimal curation to:

- remove empty units
- remove excess spikes (falling beyond the end of the recording)


### Inputs

The `data/` folder must include the output of the [aind-ephys-preprocessing](https://github.com/AllenNeuralDynamics/aind-ephys-preprocessing), containing 
the `data/preprocessed_{recording_name}` folder.

### Parameters

Parameters are read from a `params.json` file that must be in the root folder of the capsule (next to `code/`, i.e. `../params.json` from `code/`), using the `spikesorting.kilosort4` section (`skip_motion_correction`, `min_drift_channels`, `raise_if_fails`, `n_jobs`, `job_kwargs`, `sorter`, ...) and the optional top-level `logging` section.

A full list of parameters can be found in the  `aind-ephys-pipeline` repository, and are defined in this [schema](https://github.com/AllenNeuralDynamics/aind-ephys-pipeline/blob/1.5.0/pipeline/default_params_schema.json).

### Output

The output of this capsule is the following:

- `results/spikesorted_{recording_name}` folder, containing the spike sorted data saved by SpikeInterface and the spike sorting log
- `results/data_process_spikesorting_{recording_name}.json` file, a JSON file containing a `DataProcess` object from the [aind-data-schema](https://aind-data-schema.readthedocs.io/en/stable/) package.

