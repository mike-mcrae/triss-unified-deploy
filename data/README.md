# Data Contract

The app reads **only** from `${TRISS_DATA_DIR}/final/`.

Runtime initialization behavior:
- If required files already exist in `${TRISS_DATA_DIR}/final`, startup loads immediately.
- If missing, startup runs `pipeline/run_pipeline.py --only ensure_runtime_data,build_info`.
- `ensure_runtime_data` attempts remote baseline bootstrap using `TRISS_BASELINE_URL`.

## Required runtime artifacts

| Final path | Producer stage (`pipeline/run_pipeline.py`) | Required schema/keys |
|---|---|---|
| `profiles/1. profiles_summary.csv` | `ensure_runtime_data` / `incremental_researcher_update` | `n_id`, `firstname`, `lastname`, `school`, `department`, `email`, `n_publications` |
| `profiles/4. triss_researcher_summary.csv` | `ensure_runtime_data` / `incremental_researcher_update` | `n_id`, `one_line_summary`, `overall_research_area`, `topics`, `subfields` |
| `network/7.researcher_similarity_matrix.csv` | `ensure_runtime_data` | index/columns are `n_id`, values are cosine similarities |
| `network/8. user_to_other_publication_similarity_openai.csv` | `ensure_runtime_data` | `query_n_id`, `target_n_id`, `publication_index`, `title`, `similarity` |
| `publications/4.All measured publications.csv` | `ensure_runtime_data` | `n_id`, `article_id`, `Title`, `abstract`, `main_url`, `school`, `department` |
| `publications/3. All listed publications 2019 +.csv` | `ensure_runtime_data` | fallback publications table |
| `analysis/global/global_cluster_descriptions.json` | `ensure_runtime_data` | `{cluster_id: {topic_name, description, keywords}}` |
| `analysis/global/global_cluster_assignments.csv` | `ensure_runtime_data` | `n_id`, `school`, `department`, `cluster` |
| `analysis/global/global_cluster_centroids.npy` | `ensure_runtime_data` | cluster centroids array |
| `analysis/global/global_umap_coordinates.csv` | `ensure_runtime_data` | `n_id`, `cluster`, `umap_x`, `umap_y` |
| `analysis/global/global_publication_umap_coordinates.csv` | `ensure_runtime_data` | `article_id`, `predicted_cluster`, `umap_x`, `umap_y` |
| `analysis/global/stage3/policy_domains_metadata.json` | `ensure_runtime_data` | list of `{policy_domain_id, title, description, order_index}` |
| `analysis/global/stage3/global_cluster_policy_weights.csv` | `ensure_runtime_data` | cluster-to-policy weights |
| `analysis/global/stage3/researcher_policy_weights.csv` | `ensure_runtime_data` | researcher-to-policy weights |
| `analysis/schools/*/*` | `ensure_runtime_data` | school topic descriptions/centroids/assignments/UMAP |
| `analysis/global_mpnet/*` | `ensure_runtime_data` | MPNet global UMAP and topic descriptors |
| `embeddings/v3/mpnet/by_researcher/n_*.json` | `ensure_runtime_data` | MPNet researcher vectors |
| `embeddings/v3/mpnet/by_publication/article_*.json` | `ensure_runtime_data` | MPNet publication vectors |
| `embeddings/v3/openai/by_researcher/n_*.json` | `ensure_runtime_data` | OpenAI researcher vectors for topic expert ranking |
| `report/report.pdf` | `ensure_runtime_data` | report PDF served by API |
| `_BUILD_INFO.json` | `build_info` | build metadata, counts, checksums |

## Folder policy

- `data/raw/`: source/raw inputs (gitignored)
- `data/interim/`: logs and intermediate outputs (gitignored)
- `data/final/`: runtime contract (gitignored by default; allowlist small files only)
