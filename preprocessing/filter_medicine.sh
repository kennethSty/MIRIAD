#!/bin/bash

# Total shard count
total_shard_count=99

# Number of shards to process in each session
shards_per_session=5

# Loop to create screen sessions
for (( i=0; i <= $total_shard_count; i=i+$shards_per_session)); do
  # Calculate end shard for this session
  end_shard=$((i + shards_per_session - 1))
  
  # Limit the end_shard to the total_shard_count
  if [ $end_shard -gt $total_shard_count ]; then
    end_shard=$total_shard_count
  fi

  # Create a screen session that runs the script for a shard range
  screen -dmS "shard_$i-$end_shard" bash -c "python filter_medicine.py $i $end_shard"
done