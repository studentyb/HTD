#!/bin/bash
# Run No-Trace baseline on all 500 SWE-bench Verified instances
# 50 batches x 10 instances each
# Auto-cleanup after each batch

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

INSTANCE_FILE="verified_all_500_instances.txt"
RESULT_DIR="batch_results/no_trace"
LOG_FILE="$RESULT_DIR/no_trace_runner.log"
BATCH_SIZE=10
TOTAL_INSTANCES=500
NUM_BATCHES=50

mkdir -p "$RESULT_DIR"

echo "========================================" | tee -a "$LOG_FILE"
echo "No-Trace Baseline Experiment" | tee -a "$LOG_FILE"
echo "Total: $TOTAL_INSTANCES instances, $NUM_BATCHES batches x $BATCH_SIZE each" | tee -a "$LOG_FILE"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"

# Read all instances into array
mapfile -t ALL_INSTANCES < "$INSTANCE_FILE"

# Track overall stats
TOTAL_RESOLVED=0
TOTAL_PROMPT_TOKENS=0
TOTAL_COMPLETION_TOKENS=0

for BATCH_NUM in $(seq 1 $NUM_BATCHES); do
    START_IDX=$(( (BATCH_NUM - 1) * BATCH_SIZE ))
    END_IDX=$(( START_IDX + BATCH_SIZE - 1 ))

    # Clamp for last batch (shouldn't happen with exact 500/10)
    if [ $END_IDX -ge $TOTAL_INSTANCES ]; then
        END_IDX=$((TOTAL_INSTANCES - 1))
    fi

    # Extract instances for this batch
    INSTANCES=()
    for i in $(seq $START_IDX $END_IDX); do
        INSTANCES+=("${ALL_INSTANCES[$i]}")
    done

    BATCH_OUTPUT="$RESULT_DIR/no_trace_batch_${BATCH_NUM}_report.json"

    # Skip if already completed
    if [ -f "$BATCH_OUTPUT" ]; then
        resolved=$(python3 -c "import json; d=json.load(open('$BATCH_OUTPUT')); print(d.get('resolved',0))" 2>/dev/null || echo "0")
        total=$(python3 -c "import json; d=json.load(open('$BATCH_OUTPUT')); print(d.get('total_instances',0))" 2>/dev/null || echo "0")
        echo "[$(date '+%H:%M:%S')] Skipping Batch $BATCH_NUM/$NUM_BATCHES (already done: $resolved/$total)" | tee -a "$LOG_FILE"
        TOTAL_RESOLVED=$((TOTAL_RESOLVED + resolved))
        continue
    fi

    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Batch $BATCH_NUM/$NUM_BATCHES: instances $((START_IDX+1))-$((END_IDX+1))" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    # Run the batch
    BATCH_START=$(date +%s)
    if python3 no_trace_batch_runner.py \
            --no-pause \
            --instances "${INSTANCES[@]}" \
            --output "$BATCH_OUTPUT" \
            --cleanup >> "$LOG_FILE" 2>&1; then
        echo "[$(date '+%H:%M:%S')] Batch $BATCH_NUM completed successfully" | tee -a "$LOG_FILE"
    else
        echo "[$(date '+%H:%M:%S')] WARNING: Batch $BATCH_NUM failed or was interrupted" | tee -a "$LOG_FILE"
    fi
    BATCH_END=$(date +%s)
    BATCH_TIME=$((BATCH_END - BATCH_START))

    # Parse results
    if [ -f "$BATCH_OUTPUT" ]; then
        resolved=$(python3 -c "import json; d=json.load(open('$BATCH_OUTPUT')); print(d.get('resolved',0))" 2>/dev/null || echo "0")
        total=$(python3 -c "import json; d=json.load(open('$BATCH_OUTPUT')); print(d.get('total_instances',0))" 2>/dev/null || echo "0")
        ptokens=$(python3 -c "import json; d=json.load(open('$BATCH_OUTPUT')); print(d.get('total_tokens',{}).get('prompt',0))" 2>/dev/null || echo "0")
        ctokens=$(python3 -c "import json; d=json.load(open('$BATCH_OUTPUT')); print(d.get('total_tokens',{}).get('completion',0))" 2>/dev/null || echo "0")

        TOTAL_RESOLVED=$((TOTAL_RESOLVED + resolved))
        TOTAL_PROMPT_TOKENS=$((TOTAL_PROMPT_TOKENS + ptokens))
        TOTAL_COMPLETION_TOKENS=$((TOTAL_COMPLETION_TOKENS + ctokens))

        echo "[$(date '+%H:%M:%S')] Batch $BATCH_NUM result: $resolved/$total resolved, time=${BATCH_TIME}s, tokens=prompt:${ptokens}+completion:${ctokens}" | tee -a "$LOG_FILE"
    fi

    # Force cleanup regardless of success
    echo "[$(date '+%H:%M:%S')] Force-cleaning images for batch $BATCH_NUM..." | tee -a "$LOG_FILE"
    python3 /tmp/cleanup_batch_images.py --instances "${INSTANCES[@]}" >> "$LOG_FILE" 2>&1 || true

    # Disk usage
    DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}')
    echo "[$(date '+%H:%M:%S')] Disk usage after batch $BATCH_NUM: $DISK_USAGE" | tee -a "$LOG_FILE"

    # Progress summary
    COMPLETED=$((BATCH_NUM))
    REMAINING=$((NUM_BATCHES - BATCH_NUM))
    echo "[$(date '+%H:%M:%S')] PROGRESS: $TOTAL_RESOLVED/$((COMPLETED * BATCH_SIZE)) resolved so far ($COMPLETED/$NUM_BATCHES batches done, $REMAINING remaining)" | tee -a "$LOG_FILE"
done

echo "" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
echo "All 50 batches finished!" | tee -a "$LOG_FILE"
echo "Final: $TOTAL_RESOLVED/$TOTAL_INSTANCES resolved" | tee -a "$LOG_FILE"
echo "Total tokens: prompt=$TOTAL_PROMPT_TOKENS, completion=$TOTAL_COMPLETION_TOKENS" | tee -a "$LOG_FILE"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "========================================" | tee -a "$LOG_FILE"
