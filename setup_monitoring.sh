#!/bin/bash
#
# Setup script for drift monitoring scheduled tasks
# This script sets up cron jobs for regular drift monitoring

set -e  # Exit on error

# Directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Configuration
CONFIG_FILE="${1:-config_monitoring.yaml}"
VENV_PATH="${2:-$HOME/venv/superstore-marketing}"
PYTHON_PATH="$VENV_PATH/bin/python"
LOGDIR="$SCRIPT_DIR/logs"
EMAIL="$3"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file $CONFIG_FILE not found"
    exit 1
fi

# Create log directory if it doesn't exist
mkdir -p "$LOGDIR"

# Detect if we're running in a virtual environment
if [[ -z "$VIRTUAL_ENV" && ! -f "$PYTHON_PATH" ]]; then
    echo "Warning: Virtual environment not detected at $VENV_PATH"
    echo "Using system Python instead"
    PYTHON_PATH=$(which python3)
fi

echo "Using Python at: $PYTHON_PATH"
echo "Using config file: $CONFIG_FILE"

# Create the cron job
CRON_COMMAND="$PYTHON_PATH $SCRIPT_DIR/src/monitoring/scheduled_drift_check.py --config $CONFIG_FILE --log-to-mlflow"

# Add email alerts if provided
if [ ! -z "$EMAIL" ]; then
    CRON_COMMAND="$CRON_COMMAND --send-alerts --email $EMAIL"
    echo "Email alerts will be sent to: $EMAIL"
fi

# Append logging
CRON_COMMAND="$CRON_COMMAND >> $LOGDIR/drift_check.log 2>&1"

# Create cron job for daily run at 1AM
CRON_JOB="0 1 * * * $CRON_COMMAND"

# Check for frequency override in config
FREQUENCY=$(grep -A1 "frequency:" "$CONFIG_FILE" | tail -n1 | cut -d'"' -f2)
TIME=$(grep -A1 "time:" "$CONFIG_FILE" | tail -n1 | cut -d'"' -f2)
DAY=$(grep -A1 "day:" "$CONFIG_FILE" | tail -n1 | cut -d'"' -f2)

# Extract hour and minute
HOUR=$(echo $TIME | cut -d':' -f1)
MINUTE=$(echo $TIME | cut -d':' -f2)

# Configure cron schedule based on config
if [ "$FREQUENCY" = "hourly" ]; then
    CRON_JOB="0 * * * * $CRON_COMMAND"
    echo "Setting up hourly monitoring"
elif [ "$FREQUENCY" = "daily" ]; then
    CRON_JOB="$MINUTE $HOUR * * * $CRON_COMMAND"
    echo "Setting up daily monitoring at $TIME"
elif [ "$FREQUENCY" = "weekly" ]; then
    # Convert day name to number (0=Sunday, 1=Monday, etc.)
    case "$DAY" in
        Sunday) DAY_NUM=0 ;;
        Monday) DAY_NUM=1 ;;
        Tuesday) DAY_NUM=2 ;;
        Wednesday) DAY_NUM=3 ;;
        Thursday) DAY_NUM=4 ;;
        Friday) DAY_NUM=5 ;;
        Saturday) DAY_NUM=6 ;;
        *) DAY_NUM=1 ;;  # Default to Monday
    esac
    CRON_JOB="$MINUTE $HOUR * * $DAY_NUM $CRON_COMMAND"
    echo "Setting up weekly monitoring on $DAY at $TIME"
else
    echo "Using default daily schedule at 1AM"
fi

# Install the cron job
echo "Installing cron job..."
(crontab -l 2>/dev/null || echo "") | grep -v "scheduled_drift_check.py" | { cat; echo "$CRON_JOB"; } | crontab -

# Test run
echo "Running a test check..."
$PYTHON_PATH $SCRIPT_DIR/src/monitoring/scheduled_drift_check.py --config $CONFIG_FILE

echo "Monitoring setup complete. Cron job installed:"
echo "$CRON_JOB"
echo ""
echo "Usage:"
echo "  Manual run: $PYTHON_PATH $SCRIPT_DIR/src/monitoring/scheduled_drift_check.py --config $CONFIG_FILE"
echo "  View logs: cat $LOGDIR/drift_check.log"
echo "  Remove schedule: crontab -l | grep -v 'scheduled_drift_check.py' | crontab -" 