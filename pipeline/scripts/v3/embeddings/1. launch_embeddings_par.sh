#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/0. scripts/v3/embeddings/1. create_text_embeddings.py"
PYTHON_BIN="/Users/mikemcrae/Dropbox/Google Drive/Personal/2021/PhD/TRISS Project/triss-pipeline-2026/0. scripts/temp_venv/bin/python"

# Assuming 4793 rows in the 4. All measured publications.csv
TOTAL=4793
CHUNK=200
MAX_JOBS=10
MODELS=("mpnet" "openai")

NUM_CHUNKS=$(( (TOTAL + CHUNK - 1) / CHUNK ))

echo "Launching TRISS v3 embedding jobs"
echo "Total pubs: $TOTAL"
echo "Chunk size: $CHUNK"
echo "Chunks: $NUM_CHUNKS"
echo "Models: ${MODELS[*]}"
echo "Max parallel screens: $MAX_JOBS"
echo "----------------------------------"

job_count=0

for model in "${MODELS[@]}"; do
  for ((i=0; i<NUM_CHUNKS; i++)); do

    start=$(( i * CHUNK ))
    end=$(( start + CHUNK ))
    if [ "$end" -gt "$TOTAL" ]; then
      end=$TOTAL
    fi

    session="triss_${model}_v3_${start}_${end}"

    echo "▶ Launching $session"

    screen -dmS "$session" bash -c "
      echo 'Started $session';
      \"$PYTHON_BIN\" \"$SCRIPT\" --model $model --start_idx $start --end_idx $end;
      echo 'Finished $session';
      exit
    "

    job_count=$((job_count + 1))

    # throttle number of concurrent screens
    while [ "$(screen -ls | grep -c triss_)" -ge "$MAX_JOBS" ]; do
      sleep 10
    done

  done
done

echo "----------------------------------"
echo "All jobs dispatched."
screen -ls | grep triss_ || echo "None"
exit 0
