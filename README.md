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

The `code/run` script takes the following arguments:

```bash
  --raise-if-fails      Whether to raise an error in case of failure or continue. Default True (raise)
  --skip-motion-correction
                        Whether to skip Kilosort4 motion correction. Default: True
  --min-drift-channels MIN_DRIFT_CHANNELS
                        Minimum number of channels to enable Kilosort4 motion correction. Default is 96.
  --n-jobs N_JOBS       Number of jobs to use for parallel processing. Default is -1 (all available cores). 
                        It can also be a float between 0 and 1 to use a fraction of available cores
  --params-file PARAMS_FILE
                        Optional json file with parameters
  --params-str PARAMS_STR
                        Optional json string with parameters

```

A list of spike sorting parameters can be found in the `code/params.json`:

```json
{
    "job_kwargs": {
        "chunk_duration": "1s",
        "progress_bar": false
    },
    "sorter": {
        "batch_size": 60000,
        "nblocks": 5,
        "Th_universal": 9,
        "Th_learned": 8,
        "do_CAR": true,
        "invert_sign": false,
        "nt": 61,
        "shift": null,
        "scale": null,
        "artifact_threshold": null,
        "nskip": 25,
        "whitening_range": 32,
        "highpass_cutoff": 300,
        "binning_depth": 5,
        "sig_interp": 20,
        "drift_smoothing": [0.5, 0.5, 0.5],
        "nt0min": null,
        "dmin": null,
        "dminx": 32,
        "min_template_size": 10,
        "template_sizes": 5,
        "nearest_chans": 10,
        "nearest_templates": 100,
        "max_channel_distance": null,
        "templates_from_data": true,
        "n_templates": 6,
        "n_pcs": 6,
        "Th_single_ch": 6,
        "acg_threshold": 0.2,
        "ccg_threshold": 0.25,
        "cluster_downsampling": 20,
        "cluster_pcs": 64,
        "x_centers": null,
        "duplicate_spike_ms": 0.25,
        "scaleproc": null,
        "save_preprocessed_copy": false,
        "torch_device": "auto",
        "bad_channels": null,
        "clear_cache": false,
        "save_extra_vars": false,
        "do_correction": true,
        "keep_good_only": false,
        "skip_kilosort_preprocessing": false,
        "use_binary_file": null,
        "delete_recording_dat": true
    }
}
```

### Output

The output of this capsule is the following:

- `results/spikesorted_{recording_name}` folder, containing the spike sorted data saved by SpikeInterface and the spike sorting log
- `results/data_process_spikesorting_{recording_name}.json` file, a JSON file containing a `DataProcess` object from the [aind-data-schema](https://aind-data-schema.readthedocs.io/en/stable/) package.

