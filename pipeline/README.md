# Pipeline (Deploy Repo)

This folder contains the copied upstream V3 scripts in `pipeline/scripts/v3/` and a deploy-safe orchestrator in `pipeline/run_pipeline.py`.

## What `run_pipeline.py` does

`run_pipeline.py` is deterministic packaging for app runtime data. It syncs required artifacts into the single app contract:

- `${TRISS_DATA_DIR}/final/profiles/*`
- `${TRISS_DATA_DIR}/final/network/*`
- `${TRISS_DATA_DIR}/final/publications/*`
- `${TRISS_DATA_DIR}/final/analysis/*`
- `${TRISS_DATA_DIR}/final/embeddings/*`
- `${TRISS_DATA_DIR}/final/report/report.pdf`

Then it writes `${TRISS_DATA_DIR}/final/_BUILD_INFO.json`.

## Stage order

1. `sync_profiles`
2. `sync_network`
3. `sync_publications`
4. `sync_analysis`
5. `sync_embeddings`
6. `sync_report`
7. `build_info`

## Usage

```bash
python pipeline/run_pipeline.py
python pipeline/run_pipeline.py --dry-run
python pipeline/run_pipeline.py --stage sync_profiles --stage sync_network
python pipeline/run_pipeline.py --from-stage sync_publications --to-stage build_info
python pipeline/run_pipeline.py --only sync_profiles,sync_publications,build_info
```

## Config inputs

Values are loaded in precedence order:
1. environment variables,
2. `config/settings.local.yml`,
3. defaults.

Used variables:
- `TRISS_DATA_DIR`
- `TRISS_SOURCE_SHARED_DATA_DIR`
- `TRISS_SOURCE_PIPELINE_DIR`
- `TRISS_SOURCE_REPORT_DIR`

## Legacy V3 script path variables

All scripts under `pipeline/scripts/v3/**/*.py` were rewritten to remove machine-specific absolute paths.
They now read source roots from environment variables:
- `TRISS_SOURCE_PIPELINE_DIR`
- `TRISS_SOURCE_SHARED_DATA_DIR`
- `TRISS_SOURCE_REPORT_DIR`
- `TRISS_SOURCE_PROJECT_DIR` (optional umbrella root)

If unset, scripts resolve relative defaults from their location inside this deploy repo.
