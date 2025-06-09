#!/bin/bash

# Number of shards to process in each screen session
step=1  # Each session will cover a single shard

# Loop through all shards, starting a new screen session for each shard
for ((i=0; i<80; i=i+step)); do
    # Skip shard number 20
    if [ "$i" -eq 20 ]; then
        continue
    fi

    start=$i
    end=$i  # Since each session covers a single shard, start and end are the same

    # Create a detached screen session that runs the specified Python script for the given shard
    screen -dmS "shard_${start}" bash -c "python3 generate_dataset.py $start $end; exec sh"
done
