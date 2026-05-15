#!/bin/bash
# Resume SWE-bench Verified Batch 4-10 micro-batches.
# Skips already completed micro-batches and cleans images after each run.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

INSTANCE_FILE="verified_all_500_instances.txt"
RESULT_DIR="batch_results"
mkdir -p "$RESULT_DIR"

LOG_FILE="$RESULT_DIR/micro_batch_runner_resume.log"

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
log "Resuming Micro-Batch Run: Batch 4-10"
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

    OUTPUT="$RESULT_DIR/micro_batch_${MICRO_NUM}_report.json"

    # Skip if already completed successfully
    if [ -f "$OUTPUT" ]; then
        resolved=$(python3 -c "import json; d=json.load(open('$OUTPUT')); print(d.get('a_resolved',0))" 2>/dev/null || echo "0")
        total=$(python3 -c "import json; d=json.load(open('$OUTPUT')); print(d.get('total_instances',0))" 2>/dev/null || echo "0")
        if [ "$total" -gt 0 ]; then
            log "Skipping Micro-Batch $MICRO_NUM/$NUM_MICRO_BATCHES (already done: $resolved/$total)"
            continue
        fi
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

    log "Running a_only_batch_runner.py with ${#INSTANCES[@]} instances..."
    
    # Pre-build images to avoid runtime failures
    # Write instance list to temp file for prepare_images.py
    TMP_INSTANCES_FILE=$(mktemp)
    printf '%s\n' "${INSTANCES[@]}" > "$TMP_INSTANCES_FILE"
    log "Pre-building images for micro-batch $MICRO_NUM..."
    python3 prepare_images.py \
        --instance-ids-file "$TMP_INSTANCES_FILE" \
        --max-workers 2 >> "$LOG_FILE" 2>&1 || log "WARNING: Some images failed to pre-build, will attempt during run."
    rm -f "$TMP_INSTANCES_FILE"

    # Run the micro-batch
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
    fi

    # Force cleanup again regardless of run success/failure
    log "Force-cleaning images for micro-batch $MICRO_NUM..."
    python3 /tmp/cleanup_batch_images.py --instances "${INSTANCES[@]}" >> "$LOG_FILE" 2>&1 || true

    # Report disk usage after cleanup
    DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}')
    log "Disk usage after micro-batch $MICRO_NUM: $DISK_USAGE"

done

log ""
log "============================================"
log "All 50 micro-batches for Batch 4-10 finished."
log "============================================"

# Second pass: retry any failed batches (like 15, 16)
log ""
log "============================================"
log "Second pass: Checking for failed batches..."
log "============================================"

for MICRO_NUM in $(seq 1 $NUM_MICRO_BATCHES); do
    OUTPUT="$RESULT_DIR/micro_batch_${MICRO_NUM}_report.json"
    
    # Check if report exists and has valid data
    if [ -f "$OUTPUT" ]; then
        total=$(python3 -c "import json; d=json.load(open('$OUTPUT')); print(d.get('total_instances',0))" 2>/dev/null || echo "0")
        if [ "$total" -gt 0 ]; then
            continue
        fi
    fi
    
    # Missing or invalid report - retry this batch
    MICRO_START=$(( START_LINE + (MICRO_NUM - 1) * MICRO_BATCH_SIZE ))
    MICRO_END=$(( MICRO_START + MICRO_BATCH_SIZE - 1 ))
    if [ $MICRO_END -gt $END_LINE ]; then
        MICRO_END=$END_LINE
    fi
    
    log ""
    log "========================================"
    log "Retry Micro-Batch $MICRO_NUM/$NUM_MICRO_BATCHES: lines $MICRO_START-$MICRO_END"
    log "========================================"
    
    mapfile -t INSTANCES < <(sed -n "${MICRO_START},${MICRO_END}p" "$INSTANCE_FILE")
    if [ ${#INSTANCES[@]} -eq 0 ]; then
        continue
    fi
    
    TMP_INSTANCES_FILE=$(mktemp)
    printf '%s\n' "${INSTANCES[@]}" > "$TMP_INSTANCES_FILE"
    log "Pre-building images for retry micro-batch $MICRO_NUM..."
    python3 prepare_images.py \
        --instance-ids-file "$TMP_INSTANCES_FILE" \
        --max-workers 2 >> "$LOG_FILE" 2>&1 || log "WARNING: Some images failed to pre-build."
    rm -f "$TMP_INSTANCES_FILE"
    
    if python3 a_only_batch_runner.py \
            --no-pause \
            --instances "${INSTANCES[@]}" \
            --output "$OUTPUT" \
            --cleanup; then
        log "Retry Micro-Batch $MICRO_NUM completed successfully."
    else
        log "ERROR: Retry Micro-Batch $MICRO_NUM failed again."
    fi
    
    python3 /tmp/cleanup_batch_images.py --instances "${INSTANCES[@]}" >> "$LOG_FILE" 2>&1 || true
    DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}')
    log "Disk usage after retry micro-batch $MICRO_NUM: $DISK_USAGE"
done

log ""
log "============================================"
log "All retries completed."
log "============================================"
