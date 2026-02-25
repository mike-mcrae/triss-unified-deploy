# Size Report (Phase 1)

Scan basis: files >= 1 MB across current app source, pipeline source, shared data source, and report source. Raw rows are in `ops/size_report.csv`.

## Totals By Folder
| Folder | Total bytes | Total size |
|---|---:|---:|
| `app/` | 1690708713 | 1612.39 MB |
| `pipeline/` | 3411888895 | 3253.83 MB |
| `data/raw/` | 261718964 | 249.59 MB |
| `data/interim/` | 1407349782 | 1342.15 MB |
| `data/final/` | 841029834 | 802.07 MB |

## GitHub Risk Flags
- GitHub hard limit risk: files > 100 MB cannot be pushed without LFS; these are hard-fail for normal git push.
- Count of >100 MB files in scan: **8**.
- Major offenders:
  - `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/1. data/3. final/8. user_to_other_publication_similarity_openai.csv` (586.55 MB) -> class `D`
  - `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/1. data/2. intermediate/Embeddings/openai/abstract_embeddings_index.json` (492.43 MB) -> class `D`
  - `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-unified-app/backend/.venv/lib/python3.13/site-packages/torch/lib/libtorch_cpu.dylib` (214.08 MB) -> class `B`
  - `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/.venv/lib/python3.13/site-packages/torch/lib/libtorch_cpu.dylib` (214.08 MB) -> class `B`
  - `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/0. scripts/temp_venv/lib/python3.13/site-packages/torch/lib/libtorch_cpu.dylib` (214.08 MB) -> class `B`
  - `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/1. data/2. intermediate/Embeddings/mpnet/abstract_embeddings_index.json` (129.15 MB) -> class `D`
  - `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-unified-app/backend/.venv/lib/python3.13/site-packages/llvmlite/binding/libllvmlite.dylib` (116.29 MB) -> class `B`
  - `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/0. scripts/temp_venv/lib/python3.13/site-packages/llvmlite/binding/libllvmlite.dylib` (116.29 MB) -> class `B`

## Render Risk Flags
- Large runtime artifacts (tens to hundreds of MB) increase build time, slug size, startup latency, and memory pressure.
- Recommendation: Mode B (persistent disk `/var/data`) for runtime data; keep app slug code-only.
- Keep embedding/vector artifacts and large similarity tables off git slug; hydrate from persistent disk or object storage.

## Decision Class Summary
| Class | Meaning | Count (>=1MB scan) |
|---|---|---:|
| `A` | safe to commit to git | 260 |
| `B` | do not commit (dependencies/caches/build artifacts) | 238 |
| `C` | runtime artifact -> persistent disk | 7 |
| `D` | too large -> object storage or persistent disk download | 3 |

## Explicit A/B/C/D Decisions For Critical Artifacts
| Artifact | Size | Class | Deployment action | Why |
|---|---:|---|---|---|
| `../1. data/3. final/1. profiles_summary.csv` | 0.69 MB | `A` | Commit allowed (if needed) or generate in pipeline | Small enough, required baseline metadata |
| `../1. data/3. final/4. triss_researcher_summary.csv` | 0.26 MB | `A` | Commit allowed | Small-medium table used by profile cards |
| `../1. data/3. final/7.researcher_similarity_matrix.csv` | 2.13 MB | `A` | Commit allowed if below threshold | Runtime critical and currently manageable |
| `../1. data/3. final/8. user_to_other_publication_similarity_openai.csv` | 586.55 MB | `D` | Do not commit; store in object storage or persistent disk | Exceeds GitHub hard limit by large margin |
| `../triss-pipeline-2026/1. data/4. embeddings/v3/mpnet/by_publication/*` | 72.08 MB total | `C` | Keep on persistent disk `/var/data/final/embeddings` | Runtime query ranking needs local fast access |
| `../triss-pipeline-2026/1. data/4. embeddings/v3/mpnet/by_researcher/*` | 5.14 MB total | `C` | Persistent disk | Runtime expert search model input |
| `../1. data/2. intermediate/Embeddings/openai/abstract_embeddings_index.json` | 492.43 MB | `D` | Do not deploy by default | Interim index, not required at app runtime |
| `../1. data/2. intermediate/Embeddings/mpnet/abstract_embeddings_index.json` | 129.15 MB | `D` | Do not deploy by default | Interim index, not required at app runtime |
| `backend/.venv/**, frontend/node_modules/**, pipeline .venv/**` | hundreds of MB | `B` | Never commit; recreate in build | Dependency/cache artifacts only |

## Top 25 Largest Files (from scan)
| Size MB | Decision | Path |
|---:|---|---|
| 586.55 | `D` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/1. data/3. final/8. user_to_other_publication_similarity_openai.csv` |
| 492.43 | `D` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/1. data/2. intermediate/Embeddings/openai/abstract_embeddings_index.json` |
| 214.08 | `B` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-unified-app/backend/.venv/lib/python3.13/site-packages/torch/lib/libtorch_cpu.dylib` |
| 214.08 | `B` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/.venv/lib/python3.13/site-packages/torch/lib/libtorch_cpu.dylib` |
| 214.08 | `B` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/0. scripts/temp_venv/lib/python3.13/site-packages/torch/lib/libtorch_cpu.dylib` |
| 129.15 | `D` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/1. data/2. intermediate/Embeddings/mpnet/abstract_embeddings_index.json` |
| 116.29 | `B` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-unified-app/backend/.venv/lib/python3.13/site-packages/llvmlite/binding/libllvmlite.dylib` |
| 116.29 | `B` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/0. scripts/temp_venv/lib/python3.13/site-packages/llvmlite/binding/libllvmlite.dylib` |
| 46.92 | `C` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/1. data/4. embeddings/v3/mpnet/8.user_to_other_publication_similarity_mpnet.csv` |
| 46.34 | `C` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/1. data/4. embeddings/v3/openai/8.user_to_other_publication_similarity_openai.csv` |
| 37.52 | `B` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/0. scripts/temp_venv/lib/python3.13/site-packages/grpc/_cython/cygrpc.cpython-313-darwin.so` |
| 36.31 | `A` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/1. data/3. final/profiles_publications_full_merged_abstract_cleaned_with_ids_and_llm.csv` |
| 35.66 | `A` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/1. data/3. final/profiles_publications_full_merged_abstract_cleaned.csv` |
| 29.91 | `A` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/1. data/3. final/profiles_publications_full_merged_abstract_cleaning.csv` |
| 29.72 | `B` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-unified-app/backend/.venv/lib/python3.13/site-packages/torch/lib/libtorch_python.dylib` |
| 29.72 | `B` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/.venv/lib/python3.13/site-packages/torch/lib/libtorch_python.dylib` |
| 29.72 | `B` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/0. scripts/temp_venv/lib/python3.13/site-packages/torch/lib/libtorch_python.dylib` |
| 28.47 | `A` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/1. data/3. final/3. profiles_publications_full_merged.csv` |
| 27.83 | `A` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/1. data/3. final/profiles_publications_with_abstract_wordcounts.csv` |
| 27.76 | `A` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/1. data/2. intermediate/3. profiles_publications_full_merged.csv` |
| 27.76 | `B` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/1. data/1. raw/0. profiles/profiles_publications_full_merged.csv` |
| 27.73 | `A` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/0. scripts/v3/_batch_runs/authors_title/output.jsonl` |
| 25.84 | `A` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/0. scripts/v3/_batch_runs/authors_title/input.jsonl` |
| 22.65 | `C` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/1. data/3. final/Embeddings_flat/openai_abstracts_embeddings.csv` |
| 22.53 | `C` | `/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/1. data/3. final/Embeddings_flat/openai_one_line_summary_embeddings.csv` |

## Recommended Deployment Strategy
1. Use Render persistent disk mounted at `/var/data` and set `TRISS_DATA_DIR=/var/data`.
2. Build/pipeline jobs write to `/var/data/final` (single runtime contract).
3. Keep git repo code + small metadata only; enforce >90 MB pre-commit guard.
4. For any artifact still too large for disk sync windows, host in object storage and checksum-download during build/start.
