# Pipeline (Deploy Repo)

This pipeline is now runtime-contract-native:
- no syncing from external `../` source trees,
- all runtime reads/writes are under `${TRISS_DATA_DIR}`,
- startup can bootstrap `${TRISS_DATA_DIR}/final` from a remote baseline archive.

## Stages (`pipeline/run_pipeline.py`)

1. `ensure_runtime_data`
Ensures `${TRISS_DATA_DIR}/final` contains required artifacts.
If required files are missing, it attempts baseline initialization from `TRISS_BASELINE_URL`.

2. `incremental_researcher_update`
Render-safe researcher patch updater. Applies optional CSV patches from:
- `${TRISS_DATA_DIR}/raw/updates/researchers/profiles_patch.csv`
- `${TRISS_DATA_DIR}/raw/updates/researchers/summaries_patch.csv`

3. `baseline_build` (offline only)
Packages `${TRISS_DATA_DIR}/final` into a baseline tarball and manifest for remote hosting.
Blocked when `TRISS_ENV=render`.

4. `build_info`
Writes `${TRISS_DATA_DIR}/final/_BUILD_INFO.json`.

## Usage

```bash
# Default startup-safe flow
python pipeline/run_pipeline.py

# Only ensure data + write build info
python pipeline/run_pipeline.py --only ensure_runtime_data,build_info

# Render-safe incremental update
python pipeline/run_pipeline.py --only incremental_researcher_update,build_info

# Offline baseline packaging
python pipeline/run_pipeline.py --only baseline_build,build_info
```

## Config inputs

Values are loaded in precedence order:
1. environment variables
2. `config/settings.local.yml`
3. defaults

Used variables:
- `TRISS_DATA_DIR`
- `TRISS_ENV`
- `TRISS_BASELINE_URL`
- `TRISS_BASELINE_SHA256`
- `TRISS_BASELINE_TIMEOUT_SECONDS`
- `TRISS_BASELINE_OUTPUT_PATH`

