# Reproducibility materials: Nonnegative projection estimation of circular densities with Fejér inner certificates

This repository accompanies the manuscript *"Nonnegative projection estimation of circular densities with Fejér inner certificates"* by Hongyu Tang and Xinru Wang, submitted to the *Journal of the Korean Statistical Society*.

## Contents

- `code/`: analysis, simulation, diagnostic, verification, and table-checking scripts together with `requirements.txt` and `requirements-lock.txt`.
- `reproducibility_data.zip`: archive containing the `data/`, `results/`, `research/`, and `verification/` directories. Extract it at the repository root before running the verification or table-check scripts.
- `data/` (after extraction): UCI raw hourly and daily Bike Sharing data, original dataset documentation, processed complete-day records, and provenance metadata.
- `results/` (after extraction): retained simulation outputs, certificate diagnostics, timing benchmarks, and empirical application outputs. Directory names record original experiment identifiers; numeric seeds are not execution dates.
- `research/` (after extraction): experiment protocols documenting the simulation and diagnostic designs.
- `manuscript/tables.tex`: snapshot of the six quantitative tables (Tables 2--7 in the submitted manuscript, labels `tab:t1`--`tab:t6`) used by `code/check_reported_tables.py`.
- `verification/` (after extraction): independent-directory rerun and audit logs.
- `LICENSE`: MIT licence for the analysis code (see Data source below for the UCI dataset licence).
- `SHA256SUMS.txt`: SHA-256 checksums for all tracked files in this repository.

## Data source

Fanaee-T, H. (2013). *Bike Sharing*. UCI Machine Learning Repository. https://doi.org/10.24432/C5W894.

The dataset's original documentation and provenance are retained in `data/raw/Readme.txt` and `data/provenance.json`. The dataset is licensed under CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/); retain attribution and licence information when redistributing. Hourly rental counts are analysed as 24 hourly shares and do not assume exact within-hour event times.

## Verify reported manuscript tables

From the repository root, extract the data archive and verify all six quantitative manuscript tables (Tables 2--7, including Monte Carlo standard errors in Table 2) against the saved experimental summaries using Python 3 (standard library only):

```bash
python -m zipfile -e reproducibility_data.zip .
python code/check_reported_tables.py
```

This checks the reported manuscript tables against the saved summaries to the displayed precision; it does not rerun the Monte Carlo simulations from scratch or perform interval-arithmetic certification.

## Re-running experiments

The recorded fixed-degree timing benchmark (Table 6) used Python 3.12.10, NumPy 2.5.4, SciPy 1.18.1, CVXPY 1.9.3, and SCS 3.3.1 (full package lock in `code/requirements-lock.txt`, with raw samples, seeds, and timing metadata in `results/benchmark_v12/`). To rerun in an isolated environment:

```bash
python -m pip install -r code/requirements-lock.txt
python code/benchmark_v12.py
```

Run experiment scripts in a **separate copy** where the corresponding `results/` output directory does not already exist; scripts refuse to overwrite saved outputs. Elapsed wall-clock times depend on hardware, solver builds, and caches.

## Verification summary and scope

- **Table audit**: `code/check_reported_tables.py` verifies all six quantitative tables (Tables 2--7) against the saved summaries.
- **Main simulation reproduction**: all 1,440 final fits and 21,600 candidate scores in the six-scenario comparison were rerun independently and matched the saved records (`verification/main_simulation_reproduction.md`).
- **Certificate and solver checks**: `verify_certified.py`, `verify_binned_certificate.py`, `audit_precision_tradeoff.py`, and `audit_certificate_diagnostics.py` were verified against saved records and source hashes (`verification/`).
- **Numerical scope**: all implementation diagnostics are floating-point verifications (with exact rational checks reported separately for the 12 boundary cases in Appendix E), not machine-checkable interval-arithmetic proofs.

## Licence

- Analysis code: MIT License (see `LICENSE`).
- UCI Bike Sharing dataset: CC BY 4.0 (Fanaee-T, 2013).
- Repository URL: https://github.com/THY02/Nonnegative-projection-estimation-of-circular-densities-with-Fej-r-inner-certificates
