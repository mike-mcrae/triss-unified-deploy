# TRISS Unified Deploy

Single deployable folder containing:
- unified app (`app/backend`, `app/frontend`),
- pipeline scripts (`pipeline/scripts/v3`) and orchestrator (`pipeline/run_pipeline.py`),
- single runtime data contract (`data/final/*`).

## Repository layout

- `app/backend`: FastAPI API
- `app/frontend`: React/Vite UI
- `pipeline/scripts/v3`: copied upstream V3 pipeline scripts
- `pipeline/run_pipeline.py`: runtime initializer + baseline/incremental pipeline runner
- `data/`: runtime contract root (`raw`, `interim`, `final`)
- `config/`: settings files
- `ops/`: operational scripts + render blueprint + checks
- `tests/smoke`: smoke tests

## Config precedence

1. environment variables
2. `config/settings.local.yml`
3. defaults in code

Core env vars:
- `TRISS_DATA_DIR` (default `./data` locally, `/var/data` on Render)
- `TRISS_ENV` (`local` or `render`)
- `TRISS_QUERY_EMBED_MODEL`
- `TRISS_CORS_ORIGINS`
- `TRISS_BASELINE_URL`
- `TRISS_BASELINE_SHA256`
- `TRISS_BASELINE_TIMEOUT_SECONDS`
- `TRISS_BASELINE_OUTPUT_PATH`

## Local: build data and run app

1. Backend deps:
```bash
pip install -r app/backend/requirements.txt
```

2. Optional frontend deps:
```bash
cd app/frontend && npm ci && cd ../..
```

3. Ensure runtime artifacts (bootstraps from remote baseline if needed):
```bash
bash ops/build.sh
```

4. Run backend:
```bash
bash ops/start.sh
```

5. (Optional) run frontend dev server:
```bash
cd app/frontend
npm run dev
```

6. Smoke + healthcheck:
```bash
python -m unittest tests/smoke/test_data_contract.py
python -m app.healthcheck
```

## Render: deploy and update flow

### Deploy

- Use `ops/render.yaml` (Blueprint).
- Default mode is persistent disk (`/var/data`) on backend + cron service.
- Set `TRISS_DATA_DIR=/var/data` and `TRISS_BASELINE_URL` on backend.

### Re-run pipeline updates

- Cron service runs:
```bash
python pipeline/run_pipeline.py --only incremental_researcher_update,build_info
```
- You can also run same command as one-off Render job.

### Baseline workflow

1. Build baseline artifact offline:
```bash
python pipeline/run_pipeline.py --only baseline_build,build_info
```
2. Upload the generated tarball from `data/interim/baseline/` to remote storage.
3. Set `TRISS_BASELINE_URL` (and optional `TRISS_BASELINE_SHA256`) in Render.
4. Backend startup idempotently runs `ensure_runtime_data`; if `final/` exists it skips bootstrap.

### Where data lives

- Runtime artifacts live on persistent disk: `/var/data/final/*`.
- Build logs live in `/var/data/interim/logs/*`.

### How updates appear in app

- Backend always reads from `/var/data/final/*`.
- Startup never hard-fails on empty data; it starts in degraded mode and reports init status in `/api/health`.

## Large-file safety and checks

- Review `ops/size_report.md` and `ops/size_report.csv`.
- Run tracked-size guard before commit:
```bash
python ops/check_artifact_limits.py
```
- Policy: do not commit `data/raw`, `data/interim`, large `data/final` artifacts.

## Notes

- Runtime data contract is documented in `data/README.md`.
- Migration mapping and script inventory are in `MIGRATION_PLAN.md`.
