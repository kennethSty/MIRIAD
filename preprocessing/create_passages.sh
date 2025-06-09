#!/bin/bash

# Number of shards to process in each screen session
step=5

# Loop through all shards, starting new screen session for each range of shards
for ((i=0; i<100; i=i+step)); do
    start=$i
    end=$((i+step-1))

    # Create a detached screen session that runs the Python script for the given shard range
    screen -dmS "shard_${start}_${end}" bash -c "python3 create_passages.py $start $end; exec sh"
done