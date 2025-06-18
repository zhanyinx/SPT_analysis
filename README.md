# Single Particle Tracking analysis

This repository contains the code used in the study [here](https://www.biorxiv.org/content/10.1101/2022.03.03.482826v1)

There are different pipelines present:

- Spot detection and tracking

- Mean square displacement analysis

- dual color imaging analysis


The source code is contained in the folder source

To run each pipelines, bash scripts are provided within bin folder

The streamlit folder contains the application that can be used to navigate the analysed data

More detailed README (working in progress)

## Nextflow pipeline

The repository now includes a Nextflow workflow (`main.nf`) that runs the full
analysis in a reproducible manner. It performs spot detection, tracking with cell
ID assignment, motion correction and the downstream MSD and directionality
calculations. Default parameters are defined in `conf/base.config`.

Basic usage:

```bash
nextflow run main.nf --input <tiff_dir> --spt_path <path_to_repo> \
                     --fiji <path_to_fiji> --mask_dir <mask_directory>
```
The workflow automatically loads parameters from `conf/base.config`.
Override any parameter on the command line if needed.
