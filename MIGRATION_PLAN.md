# MIGRATION_PLAN

## Phase 1 Mapping Snapshot

### Located Roots
| Area | Path | Notes |
|---|---|---|
| Unified app root | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-unified-app` | Active app with FastAPI backend + React frontend |
| Pipeline root | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026` | Full V3 pipeline scripts + embeddings + analysis outputs |
| Shared data root | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/1. data` | Existing `1. data` raw/interim/final source tables |
| Report source root | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-report-app-v3` | Report PDF served by API |

### Pipeline Script Inventory (all discovered scripts)
Discovered `73` Python scripts under `triss-pipeline-2026/0. scripts/v3`.

| Script | Inputs (current) | Outputs (current) |
|---|---|---|
| `0. scripts/v3/0.scrape_names_from_rss.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/1.parse_rss_htmls.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/10. aggregate_profiles_v3.py` | profiles_parsed.json; staff_cleaned.csv | profiles_summary_v3.csv |
| `0. scripts/v3/11. crossref_doi_discovery.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/12. crossref_doi_discovery_pass2.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/13. crossref_doi_fix_errors.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/14. openalex_doi_fallback.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/15. filter_editor_appointments.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/16. split_by_doi.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/17. collect_abstracts_doi.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/18. collect_abstracts_semanticscholar.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/19. combine_missing_abstracts.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/20. scrape_abstracts_web.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/21. test_selenium_extraction.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/22. test_seleniumbase_uc.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/23. scrape_abstracts_seleniumbase.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/24. scrape_abstracts_parallel.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/25. test_google_scaling.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/26. scrape_url_tara_single.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/27. autonomous_scrape_robust.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/28. backfill_from_final.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/29. backfill_from_full_merged.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/3. json_to_csv.py` | 1. data/1. raw/0. profiles/v3/profiles_parsed.json; 1. data/1. raw/staff_cleaned.csv | 1. data/1. raw/0. profiles/v3/0.all_listed_pubs.csv |
| `0. scripts/v3/30. remove_empty_rows.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/31. merge_abstracts_final.py` | abstract collection outputs + publication tables | 1. data/3. final/4.All measured publications.csv |
| `0. scripts/v3/32. add_main_url.py` | 4.All measured publications.csv | 4.All measured publications.csv (main_url enriched) |
| `0. scripts/v3/33. post_hoc_url_fix.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/34. process_books_other.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/35. scrape_books_selenium_llm.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/36. scrape_books_web_search.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/4. count_pub_types.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/5. extract_year_llm.py` | 0.all_listed_pubs.csv | 0.all_listed_pubs_enriched.csv |
| `0. scripts/v3/6. extract_authors_title_llm.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/7. extract_authors_title_openai_batch.py` | 0.all_listed_pubs_enriched.csv | 0.all_listed_pubs_fully_enriched.csv + batch logs |
| `0. scripts/v3/7b. extract_authors_title_openai_cleanup.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/8. filter_and_count_2019.py` | 0.all_listed_pubs_fully_enriched.csv | 2.all_listed_pubs_2019.csv; 2b.publication_types_totals_2019.csv |
| `0. scripts/v3/9. filter_pub_types.py` | 0.all_listed_pubs_fully_enriched.csv; 2.all_listed_pubs_2019.csv | 0c.all_listed_publications_filter.csv; 2c.all_listed_publications_2019.csv |
| `0. scripts/v3/98. propagate_article_ids.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/99.testing_book_recovery.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/_old/11. openalex_doi_discovery.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/clean_reconcile_publications.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/debug_abstract_stats.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/debug_crossref.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/debug_detailed_analysis.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/debug_entry_1.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/debug_google_search.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/debug_ids.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/debug_openalex.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/debug_openalex_filter.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/debug_propagation.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/debug_scrape_filter.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/debug_scrape_text.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/debug_verify_fix_entry_1.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/embeddings/1. create_text_embeddings.py` | 1. data/3. final/4.All measured publications.csv | 1. data/4. embeddings/v3/{openai|mpnet}/by_publication/article_*.json |
| `0. scripts/v3/embeddings/2. compute_similarities.py` | .../by_publication/article_*.json | .../by_researcher/n_*.json; 7.researcher_similarity_matrix_<model>.csv; 8.user_to_other_publication_similarity_<model>.csv |
| `0. scripts/v3/embeddings/3. run_semantic_clustering.py` | 4.All measured publications.csv; openai embeddings | 1. data/5. analysis/global/* (cluster descriptions, assignments, centroids, UMAP) |
| `0. scripts/v3/embeddings/3. run_semantic_clustering_10_topics.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/embeddings/3. run_semantic_clustering_mpnet.py` | 4.All measured publications.csv; mpnet embeddings | 1. data/5. analysis/global_mpnet/* and schools_mpnet/* |
| `0. scripts/v3/embeddings/3. run_semantic_clustering_mpnet_20_topics.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/embeddings/4. visualise_clusters.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/embeddings/5. run_stage2_macro_themes.py` | global cluster outputs + openai researcher embeddings | 1. data/5. analysis/global/stage2/* |
| `0. scripts/v3/embeddings/6. run_stage3_policy_domains.py` | stage2 outputs + global assignments | 1. data/5. analysis/global/stage3/policy_domain_embeddings.npy; policy_domains_metadata.json; macro_theme_policy_weights.csv |
| `0. scripts/v3/embeddings/6b. run_stage3_cluster_linking.py` | stage3 policy embeddings + global/school clusters | 1. data/5. analysis/global/stage3/global_cluster_policy_weights.csv; researcher_policy_weights.csv; policy_domains_metadata.json |
| `0. scripts/v3/generate_counts.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/generate_structured_json.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/report/4. create_descriptive_tables.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/report/5. create_pub_types_table.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/report/6. create_distilled_synthesis.py` | v2 synthesis + global/school cluster descriptions | 1. data/5. analysis/global/triss_distilled_synthesis_v3.json |
| `0. scripts/v3/report/7. generate_themes_latex.py` | triss_distilled_synthesis_v3.json | 1. data/5. analysis/global/latex/*.tex; school theme tex |
| `0. scripts/v3/report/check_counts.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/report/debug_discrepancy.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/report/debug_json.py` | See script constants / helper files | Not part of runtime contract or script-local output |
| `0. scripts/v3/verify_2e.py` | See script constants / helper files | Not part of runtime contract or script-local output |

### DAG (current -> target contract)
```text
data/raw (staff_cleaned.csv, profiles_parsed.json, scrape outputs)
  -> data/interim (flattened pubs, LLM enrichments, DOI/abstract recovery)
  -> data/final (profiles summary, measured publications, similarity tables)
  -> analysis (global/school clusters, UMAP coords, stage2/stage3 policy linking)
  -> app runtime caches (network/map/expert/report endpoints)

Target contract after consolidation:
data/raw -> data/interim -> data/final -> app reads ONLY data/final/*
with analysis + embedding artifacts copied/promoted into namespaced paths under data/final/.
```
### Exact App Runtime Reads (current absolute layout)
| Runtime artifact | Current source path | Type | Required keys/columns used by app |
|---|---|---|---|
| `profiles_summary` | `../1. data/3. final/1. profiles_summary.csv` | csv | n_id, firstname, lastname, department, school, email, n_publications |
| `researcher_summaries` | `../1. data/3. final/4. triss_researcher_summary.csv` | csv | n_id, one_line_summary, overall_research_area, topics, subfields |
| `researcher_similarity_matrix` | `../1. data/3. final/7.researcher_similarity_matrix.csv` | csv | row_index=n_id, columns=n_id, value=cosine_similarity |
| `publication_similarity_openai` | `../1. data/3. final/8. user_to_other_publication_similarity_openai.csv` | csv | query_n_id, target_n_id, publication_index, title, similarity |
| `publications_measured` | `../triss-pipeline-2026/1. data/3. final/4.All measured publications.csv` | csv | n_id, article_id, Title, abstract, main_url, school, department |
| `publications_2019_fallback` | `../triss-pipeline-2026/1. data/3. final/3. All listed publications 2019 +.csv` | csv | n_id, article_id, Title, Year Descending |
| `distilled_synthesis` | `../triss-pipeline-2026/1. data/5. analysis/global/triss_distilled_synthesis_v3.json` | json | core_identity_summary, cross_cutting_themes, school_themes |
| `global_cluster_descriptions` | `../triss-pipeline-2026/1. data/5. analysis/global/global_cluster_descriptions.json` | json | {cluster_id:{topic_name,description,keywords}} |
| `global_cluster_assignments` | `../triss-pipeline-2026/1. data/5. analysis/global/global_cluster_assignments.csv` | csv | n_id, school, department, cluster |
| `global_cluster_centroids` | `../triss-pipeline-2026/1. data/5. analysis/global/global_cluster_centroids.npy` | npy | shape=(k,embedding_dim) |
| `global_umap_coords_openai` | `../triss-pipeline-2026/1. data/5. analysis/global/global_umap_coordinates.csv` | csv | n_id, cluster, umap_x, umap_y |
| `global_pub_umap_openai` | `../triss-pipeline-2026/1. data/5. analysis/global/global_publication_umap_coordinates.csv` | csv | article_id, predicted_cluster, umap_x, umap_y |
| `global_umap_coords_mpnet` | `../triss-pipeline-2026/1. data/5. analysis/global_mpnet/global_umap_coordinates.csv` | csv | n_id, cluster, umap_x, umap_y |
| `global_pub_umap_mpnet` | `../triss-pipeline-2026/1. data/5. analysis/global_mpnet/global_publication_umap_coordinates.csv` | csv | article_id, predicted_cluster, umap_x, umap_y |
| `policy_domains_metadata` | `../triss-pipeline-2026/1. data/5. analysis/global/stage3/policy_domains_metadata.json` | json | [{policy_domain_id,title,description,order_index}] |
| `cluster_policy_weights` | `../triss-pipeline-2026/1. data/5. analysis/global/stage3/global_cluster_policy_weights.csv` | csv | cluster_id, hard_policy_domain, policy_domain_0..4 |
| `researcher_policy_weights` | `../triss-pipeline-2026/1. data/5. analysis/global/stage3/researcher_policy_weights.csv` | csv | n_id, hard_policy_domain, policy_domain_0..4 |
| `mpnet_researcher_embeddings` | `../triss-pipeline-2026/1. data/4. embeddings/v3/mpnet/by_researcher/n_*.json` | jsonl | n_id, embeddings.abstracts_mean |
| `mpnet_publication_embeddings` | `../triss-pipeline-2026/1. data/4. embeddings/v3/mpnet/by_publication/article_*.json` | jsonl | article_id, n_id, embeddings.abstract |
| `openai_researcher_embeddings` | `../triss-pipeline-2026/1. data/4. embeddings/v3/openai/by_researcher/n_*.json` | jsonl | n_id, embeddings.abstracts_mean |
| `school_analysis_bundle` | `../triss-pipeline-2026/1. data/5. analysis/schools/*/(descriptions,centroids,assignments,topic_counts,publication_umap)` | mixed | school_cluster_descriptions.json, school_cluster_centroids.npy, school_cluster_assignments.csv, school_topic_counts.csv, school_publication_umap_coordinates.csv |
| `report_pdf` | `../triss-report-app-v3/latex/report.pdf` | pdf | binary PDF served via /api/report/v3/pdf |

### Runtime Schema Notes
- `profiles_summary.csv`: requires `n_id`, identity fields (`firstname`, `lastname`, `email`), org fields (`school`, `department`), and publication counters.
- `4.All measured publications.csv`: requires `article_id`, `n_id`, `Title`, `abstract`, `main_url`, plus school/department for filtering.
- `global_cluster_descriptions.json`: per-cluster map consumed by map/report/topic labels (`topic_name`, `description`, `keywords`).
- `stage3/*.csv/json`: powers Policy Domains page + policy coloring/filtering + policy experts ranking.
- `by_researcher`/`by_publication` embeddings: MPNet vectors for expert-search query embedding and publication ranking; OpenAI researcher vectors for key-area centroid ranking.
- `schools/*` bundle: school-level topics/expert lists and school topic publication matching.

### Migration Mapping (what moves where in deploy repo)
| Current source | Target in `triss-unified-deploy/` |
|---|---|
| `triss-unified-app/backend/*` | `app/backend/*` |
| `triss-unified-app/frontend/*` | `app/frontend/*` |
| `triss-pipeline-2026/0. scripts/v3/*` | `pipeline/scripts/v3/*` |
| `triss-pipeline-2026/0. scripts/v3/report/*` | `pipeline/scripts/v3/report/*` |
| `triss-pipeline-2026/0. scripts/v3/embeddings/*` | `pipeline/scripts/v3/embeddings/*` |
| runtime data artifacts listed above | `data/final/*` (single app contract) |
| report PDF (`triss-report-app-v3/latex/report.pdf`) | `data/final/report/report.pdf` (or static asset path under app) |

### Path Normalization Status
- `app/backend/main.py` now reads only `${TRISS_DATA_DIR}/final/*` and no longer depends on absolute machine paths.
- `pipeline/run_pipeline.py` is path-normalized and env/config driven (`TRISS_*` variables).
- All legacy scripts in `pipeline/scripts/v3/**/*.py` were second-pass refactored to remove machine-specific absolute constants (`/Users/...`), using env-overridable root variables (`TRISS_SOURCE_PIPELINE_DIR`, `TRISS_SOURCE_SHARED_DATA_DIR`, `TRISS_SOURCE_REPORT_DIR`, `TRISS_SOURCE_PROJECT_DIR`).
