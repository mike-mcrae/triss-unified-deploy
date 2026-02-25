# Data Contract

The app reads **only** from `${TRISS_DATA_DIR}/final/`.

## Required runtime artifacts

| Final path | Producer stage (`pipeline/run_pipeline.py`) | Required schema/keys |
|---|---|---|
| `profiles/1. profiles_summary.csv` | `sync_profiles` | `n_id`, `firstname`, `lastname`, `school`, `department`, `email`, `n_publications` |
| `profiles/4. triss_researcher_summary.csv` | `sync_profiles` | `n_id`, `one_line_summary`, `overall_research_area`, `topics`, `subfields` |
| `network/7.researcher_similarity_matrix.csv` | `sync_network` | index/columns are `n_id`, values are cosine similarities |
| `network/8. user_to_other_publication_similarity_openai.csv` | `sync_network` | `query_n_id`, `target_n_id`, `publication_index`, `title`, `similarity` |
| `publications/4.All measured publications.csv` | `sync_publications` | `n_id`, `article_id`, `Title`, `abstract`, `main_url`, `school`, `department` |
| `publications/3. All listed publications 2019 +.csv` | `sync_publications` | fallback publications table |
| `analysis/global/global_cluster_descriptions.json` | `sync_analysis` | `{cluster_id: {topic_name, description, keywords}}` |
| `analysis/global/global_cluster_assignments.csv` | `sync_analysis` | `n_id`, `school`, `department`, `cluster` |
| `analysis/global/global_cluster_centroids.npy` | `sync_analysis` | cluster centroids array |
| `analysis/global/global_umap_coordinates.csv` | `sync_analysis` | `n_id`, `cluster`, `umap_x`, `umap_y` |
| `analysis/global/global_publication_umap_coordinates.csv` | `sync_analysis` | `article_id`, `predicted_cluster`, `umap_x`, `umap_y` |
| `analysis/global/stage3/policy_domains_metadata.json` | `sync_analysis` | list of `{policy_domain_id, title, description, order_index}` |
| `analysis/global/stage3/global_cluster_policy_weights.csv` | `sync_analysis` | cluster-to-policy weights |
| `analysis/global/stage3/researcher_policy_weights.csv` | `sync_analysis` | researcher-to-policy weights |
| `analysis/schools/*/*` | `sync_analysis` | school topic descriptions/centroids/assignments/UMAP |
| `analysis/global_mpnet/*` | `sync_analysis` | MPNet global UMAP and topic descriptors |
| `embeddings/v3/mpnet/by_researcher/n_*.json` | `sync_embeddings` | MPNet researcher vectors |
| `embeddings/v3/mpnet/by_publication/article_*.json` | `sync_embeddings` | MPNet publication vectors |
| `embeddings/v3/openai/by_researcher/n_*.json` | `sync_embeddings` | OpenAI researcher vectors for topic expert ranking |
| `report/report.pdf` | `sync_report` | report PDF served by API |
| `_BUILD_INFO.json` | `build_info` | build metadata, counts, checksums |

## Folder policy

- `data/raw/`: source/raw inputs (gitignored)
- `data/interim/`: logs and intermediate outputs (gitignored)
- `data/final/`: runtime contract (gitignored by default; allowlist small files only)
