#!/bin/bash
launch_file=`sed -n 's/^LAUNCH_FILE *= *"\(.*\)"/\1/p' topic_names.py`
launch_pkg=`sed -n 's/^LAUNCH_PKG *= *"\(.*\)"/\1/p' topic_names.py`
img_topic=`sed -n 's/^IMAGE_TOPIC *= *"\(.*\)"/\1/p' topic_names.py`

# Failsafe to make sure that pip install -e has been executed
# This is necessary to ensure that the package needed are present
# eval "$(conda shell.bash hook)"
# conda activate vint_deployment

# source /opt/ros/noetic/setup.bash

# # Navigate to the directory containing the package
# cd /workspace/src/visualnav-transformer
# # Install the package in editable mode
# pip install -e train/

# Change back the directory to the working dir with the navigate.py script
cd /workspace/src/NaiviBridger/deployment/src


SESSION="naivibridger_session"
# Start tmux session detached
tmux new-session -d -s $SESSION

# --- Create top 3 panes ---
# Pane 0 exists by default
tmux split-window -h -t $SESSION:0.0          # Creates Pane 1 (0 left, 1 right)
tmux split-window -h -t $SESSION:0.1          # Creates Pane 2 (0 left, 1 middle, 2 right)

# --- Create big bottom pane ---
tmux split-window -v -t $SESSION:0.0          # Splits pane 0 vertically → Pane 3 at bottom
tmux join-pane -h -t $SESSION:0.3             # Merge all bottom splits into pane 3 (if needed)



tmux select-pane -t $SESSION:0.0
tmux send-keys "python3 navigate.py $@" Enter


tmux select-pane -t $SESSION:0.1
tmux send-keys "python3 pd_controller.py" Enter 


tmux select-pane -t $SESSION:0.2
tmux send-keys "python3 topic_hz_monitor.py" Enter

tmux select-pane -t $SESSION:0.3
tmux send-keys "python3 monitor.py" Enter

# Attach to the session
tmux attach -t $SESSION
