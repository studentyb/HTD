#!/bin/bash
# Run SWE-bench Verified Batch 4 through 10 as 50 micro-batches.
# Each micro-batch runs 7 instances, saves the report, and cleans Docker images immediately.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

INSTANCE_FILE="verified_all_500_instances.txt"
RESULT_DIR="batch_results"
mkdir -p "$RESULT_DIR"

LOG_FILE="$RESULT_DIR/micro_batch_runner.log"

# Batch 4-10 covers lines 151-500 (350 instances total)
START_LINE=151
END_LINE=500
TOTAL_INSTANCES=$((END_LINE - START_LINE + 1))
MICRO_BATCH_SIZE=7
NUM_MICRO_BATCHES=50

# Helper to log with timestamp
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "============================================"
log "Starting Micro-Batch Run: Batch 4-10"
log "Total instances: $TOTAL_INSTANCES (lines $START_LINE-$END_LINE)"
log "Micro-batch size: $MICRO_BATCH_SIZE"
log "Number of micro-batches: $NUM_MICRO_BATCHES"
log "============================================"

for MICRO_NUM in $(seq 1 $NUM_MICRO_BATCHES); do
    # Calculate line range for this micro-batch
    MICRO_START=$(( START_LINE + (MICRO_NUM - 1) * MICRO_BATCH_SIZE ))
    MICRO_END=$(( MICRO_START + MICRO_BATCH_SIZE - 1 ))
    
    # Clamp to END_LINE for the last batch
    if [ $MICRO_END -gt $END_LINE ]; then
        MICRO_END=$END_LINE
    fi

    log ""
    log "========================================"
    log "Micro-Batch $MICRO_NUM/$NUM_MICRO_BATCHES: lines $MICRO_START-$MICRO_END"
    log "========================================"

    # Read instance IDs for this micro-batch
    mapfile -t INSTANCES < <(sed -n "${MICRO_START},${MICRO_END}p" "$INSTANCE_FILE")
    
    if [ ${#INSTANCES[@]} -eq 0 ]; then
        log "Warning: no instances found for micro-batch $MICRO_NUM, skipping."
        continue
    fi

    OUTPUT="$RESULT_DIR/micro_batch_${MICRO_NUM}_report.json"

    log "Running a_only_batch_runner.py with ${#INSTANCES[@]} instances..."
    
    # Run the micro-batch with cleanup enabled
    RUN_OK=0
    if python3 a_only_batch_runner.py \
            --no-pause \
            --instances "${INSTANCES[@]}" \
            --output "$OUTPUT" \
            --cleanup; then
        log "Micro-Batch $MICRO_NUM completed successfully. Report: $OUTPUT"
        RUN_OK=1
    else
        log "ERROR: Micro-Batch $MICRO_NUM failed or was interrupted."
        # Continue to next micro-batch so one failure doesn't block the rest
    fi

    # Extra safety: force cleanup again regardless of run success/failure
    log "Force-cleaning images for micro-batch $MICRO_NUM..."
    python3 /tmp/cleanup_batch_images.py --instances "${INSTANCES[@]}" >> "$LOG_FILE" 2>&1 || true

    # Extra safety: report disk usage after cleanup
    DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}')
    log "Disk usage after micro-batch $MICRO_NUM: $DISK_USAGE"

done

log ""
log "============================================"
log "All 50 micro-batches for Batch 4-10 finished."
log "============================================"
