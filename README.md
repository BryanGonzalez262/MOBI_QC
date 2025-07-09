[![DOI](https://zenodo.org/badge/657341621.svg)](https://zenodo.org/doi/10.5281/zenodo.10383685)

# A Python Pipeline for Quality Control of Multimodal Datasets

[![Build](https://github.com/childmindresearch/MOBI_QC/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/childmindresearch/MOBI_QC/actions/workflows/test.yaml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/childmindresearch/MOBI_QC/branch/main/graph/badge.svg?token=22HWWFWPW5)](https://codecov.io/gh/childmindresearch/MOBI_QC)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
![stability-stable](https://img.shields.io/badge/stability-stable-green.svg)
[![LGPL--3.0 License](https://img.shields.io/badge/license-LGPL--3.0-blue.svg)](https://github.com/childmindresearch/MOBI_QC/blob/main/LICENSE)
[![pages](https://img.shields.io/badge/api-docs-blue)](https://childmindresearch.github.io/MOBI_QC)

**MOBI_QC** is a python quality control pipeline designed for efficient, automated assessment of multimodal datasets (in xdf format), including EEG, Eye-tracking, Physiological signals, Audio and Video data. The pipeline performs preprocessing, computes a comprehensive quality control metrics for each modality and generates a PDF report summarizing the results for easy review and documentation.

## Features

- **Multimodal Compatibility**: Supports EEG, Eye-Tracking, ECG, EDA, Respiration, Audio, Video and Behavior data.
- **Automated Preprocessing**: Standardized cleaning and preprocessing for each modality.
- **Quality Control Metrics Computation**:
  - **EEG**: sampling rate, mean peak-to-peak amplitude, standard deviation of amplitude, absolute voltage range, high and low frequency noise, flat channels, line noise ratio, dropout percentage, signal-to-noise ratio, alpha power, spectral entropy, correlations between channels, number of missing samples.
  - **Eye-tracking**: validity of gaze point, gaze origin and pupil diameter, mean difference in percent valid data between right and left eyes.
  - **ECG**: average heart rate, signal-to-noise ratio, signal quality indices - kurtosis, power spectrum distribution, relative power in baseline.
  - **EDA**: signal integrity check, average skin conductance level (SCL), SCL standard deviation, SCL coefficient of variation, average amplitude of skin conductance response (SCR), SCR validity, signal-to-noise ratio.
  - **Respiration**: signal-to-noise-ratio, breath amplitude - mean, standard deviation and range, respiration rate - mean standard deviation and range, peak-to-peak interval - mean, standard deviation and range, baseline drift, autocorrelation at typical breath cycle.
  - **Audio**: percent of missing data, distribution of microphone samples - first quartile, third quartile, mean, standard deviation, minimum and maximum.
  - **Video**: sampling rate, percentage of frames with faces detected.
- **Generation of PDF Report**: A PDF report is generated for each subject data summarizing quality control metric and visualizations for each modality for review and documentation.
- **Export to csv**: Results from entire dataset are summarized and exported in a csv file to facilitate feature identification and descriptive statistics for the dataset.

## Requirements
- **pyxdf**: a python library for importing Extensible Data Format(XDF)
- **MNE-Python**: a python pacakge used for preproecessing of EEG
- **Neurokit2**: a python toolbox used for processing of physiological signals
- **librosa**: a python package for music and audio analysis
- **opencv-python**: a open-source python library for computer vision and machine learning

## Installation (TBA)

Install this package via :

```sh
pip install MOBI_QC
```

Or get the newest development version via:

```sh
pip install git+https://github.com/childmindresearch/MOBI_QC
```

## Quick start (TBA)

Short tutorial, maybe with a

```Python
import MOBI_QC

MOBI_QC.short_example()
```

## Links or References (TBA)

- [https://www.wikipedia.de](https://www.wikipedia.de)
