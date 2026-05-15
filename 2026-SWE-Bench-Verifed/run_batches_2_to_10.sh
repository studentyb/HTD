#!/bin/bash
# Run SWE-bench Verified Batch 2 through 10 sequentially.
# Each batch runs 50 instances, saves the report, and cleans Docker images before the next batch.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

INSTANCE_FILE="verified_all_500_instances.txt"
RESULT_DIR="batch_results"
mkdir -p "$RESULT_DIR"

LOG_FILE="$RESULT_DIR/batch_runner.log"

# Helper to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "============================================"
log "Starting sequential run: Batch 2 -> Batch 10"
log "============================================"

for BATCH_NUM in $(seq 2 10); do
    START=$(( (BATCH_NUM - 1) * 50 + 1 ))
    END=$(( BATCH_NUM * 50 ))

    log ""
    log "========================================"
    log "Batch $BATCH_NUM: lines $START-$END of $INSTANCE_FILE"
    log "========================================"

    # Read instance IDs for this batch
    mapfile -t INSTANCES < <(sed -n "${START},${END}p" "$INSTANCE_FILE")
    
    if [ ${#INSTANCES[@]} -eq 0 ]; then
        log "Warning: no instances found for Batch $BATCH_NUM, skipping."
        continue
    fi

    OUTPUT="$RESULT_DIR/batch_${BATCH_NUM}_report.json"

    log "Running a_only_batch_runner.py with ${#INSTANCES[@]} instances..."
    
    # Run the batch
    if python3 a_only_batch_runner.py \
            --no-pause \
            --instances "${INSTANCES[@]}" \
            --output "$OUTPUT" \
            --cleanup; then
        log "Batch $BATCH_NUM completed successfully. Report: $OUTPUT"
    else
        log "ERROR: Batch $BATCH_NUM failed or was interrupted."
        # Continue to next batch so one failure doesn't block the rest
    fi

done

log ""
log "============================================"
log "All batches 2-10 finished."
log "============================================"
