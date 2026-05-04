#!/usr/bin/env bash
#
# idle_shutdown.sh
# Version: 1.0.0
# Purpose: Auto-stop a RunPod GPU Pod when idle for N minutes.
# Author: Bini (with Copilot)
#
# Requirements:
#   - RUNPOD_API_KEY must be set in environment
#   - RUNPOD_POD_ID must be set in environment
#   - Script installed to /usr/local/bin by setup.sh
#   - Cron runs this script every 1–5 minutes
#

echo "$(date) [HEARTBEAT] Cron fired" >> /var/log/idle_shutdown.log


### --- CONFIGURATION -------------------------------------------------------

# Idle threshold in minutes before shutdown
IDLE_THRESHOLD=10

# Log file
LOG_FILE="/var/log/idle_shutdown.log"

# Temp file to track consecutive idle minutes
STATE_FILE="/tmp/idle_minutes.state"

### --- SAFETY CHECKS -------------------------------------------------------

if [[ -z "$RUNPOD_API_KEY" ]]; then
    echo "$(date) [ERROR] RUNPOD_API_KEY is not set" >> "$LOG_FILE"
    exit 1
fi

if [[ -z "$RUNPOD_POD_ID" ]]; then
    echo "$(date) [ERROR] RUNPOD_POD_ID is not set" >> "$LOG_FILE"
    exit 1
fi

### --- GPU UTILIZATION CHECK ----------------------------------------------

# Query GPU utilization (0–100)
UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1)

if [[ -z "$UTIL" ]]; then
    echo "$(date) [ERROR] Could not read GPU utilization" >> "$LOG_FILE"
    exit 1
fi

### --- IDLE TRACKING -------------------------------------------------------

# Load previous idle count
if [[ -f "$STATE_FILE" ]]; then
    IDLE_MINUTES=$(cat "$STATE_FILE")
else
    IDLE_MINUTES=0
fi

# Update idle counter
if (( UTIL < 10 )); then
    IDLE_MINUTES=$((IDLE_MINUTES + 1))
    echo "$(date) [INFO] GPU idle (${UTIL}%). Idle minutes: ${IDLE_MINUTES}" >> "$LOG_FILE"
else
    IDLE_MINUTES=0
    echo "$(date) [INFO] GPU active (${UTIL}%). Reset idle counter." >> "$LOG_FILE"
fi

# Save updated state
echo "$IDLE_MINUTES" > "$STATE_FILE"

### --- SHUTDOWN LOGIC ------------------------------------------------------

if (( IDLE_MINUTES >= IDLE_THRESHOLD )); then
    echo "$(date) [ACTION] Idle threshold reached. Stopping Pod..." >> "$LOG_FILE"

    # GraphQL mutation to stop the Pod
    curl -s -X POST "https://api.runpod.io/graphql" \
       -H "Content-Type: application/json" \
       -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
       -d "{
           \"query\": \"mutation { podStop(input: { podId: \\\"${RUNPOD_POD_ID}\\\" }) { id desiredStatus } }\"
       }" >> "$LOG_FILE" 2>&1


    echo "$(date) [DONE] Pod stop request sent." >> "$LOG_FILE"

    # Reset idle counter
    echo "0" > "$STATE_FILE"
fi

exit 0
