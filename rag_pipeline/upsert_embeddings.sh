#!/bin/bash

CONFIG_FILE="config.yaml"

# Define parameters
EMBEDDING_DIR=$(yq -r '.output_dir' $CONFIG_FILE)
echo "EMBEDDING_DIR: $EMBEDDING_DIR"
# Extract MODEL_NAME and get the last element after splitting by '/'
MODEL_NAME=$(yq -r '.model_name' $CONFIG_FILE | awk -F'/' '{print $NF}')  # MODEL_NAME="all-MiniLM-L6-v2"
echo "Extracted MODEL_NAME: $MODEL_NAME"
CONTENT=$(yq -r '.content' $CONFIG_FILE)
CPUS=(0)  # Set equal to num of embedding files. Adjust to available cores if necessary
VECTOR_SIZE=$(yq '.vector_size' $CONFIG_FILE)
HOST=$(yq -r '.qdrant_host' $CONFIG_FILE)
PORT=$(yq -r '.qdrant_port' $CONFIG_FILE)
CHECKPOINT_DIR=$(yq '.checkpoint_dir' $CONFIG_FILE)
BATCH_SIZE=$(yq '.upsert_batch_size' $CONFIG_FILE)
CHECK_IF_EXISTS=$(yq '.check_if_point_exists' $CONFIG_FILE)
RESUME=$(yq '.resume' $CONFIG_FILE)
OVERWRITE=$(yq '.overwrite' $CONFIG_FILE)

# Dynamically generate the string
# COLLECTION_NAME="miriad_4.4M_${MODEL_NAME}_${CONTENT}"
COLLECTION_NAME="miriad_${MODEL_NAME}_${CONTENT}"
echo "COLLECTION_NAME: $COLLECTION_NAME"

# Dynamically generate file paths in Bash
# EMBEDDING_FILES=($(ls ${EMBEDDING_DIR}/miriad_4.4M_${MODEL_NAME}_${CONTENT}_embeddings_rank*.npy | sort))
EMBEDDING_FILES=($(ls ${EMBEDDING_DIR}/miriad_${MODEL_NAME}_${CONTENT}_embeddings_rank*.npy | sort))

# Check if the number of files matches the CPUs
if [ ${#EMBEDDING_FILES[@]} -ne ${#CPUS[@]} ]; then
    echo "Error: Number of embedding files (${#EMBEDDING_FILES[@]}) does not match the number of CPUs (${#CPUS[@]})!"
    exit 1
fi

# Step 1: Ensure the collection is created before running parallel tasks
python create_collection.py \
    --collection_name "$COLLECTION_NAME" \
    --vector_size $VECTOR_SIZE \
    --host "$HOST" \
    --port $PORT \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --check_if_exists $CHECK_IF_EXISTS \
    --overwrite_existing_collection $OVERWRITE

# Step 2: Run parallel processes
# Loop over CPUs and launch the Python script in parallel
for i in "${!CPUS[@]}"; do
    CPU="${CPUS[$i]}"
    FILE_PATH="${EMBEDDING_FILES[$i]}"

    # Run the Python script with taskset to bind it to a specific CPU
    taskset -c "$CPU" python upsert_embeddings.py \
        --file_path "$FILE_PATH" \
        --collection_name "$COLLECTION_NAME" \
        --vector_size $VECTOR_SIZE \
        --host "$HOST" \
        --port $PORT \
        --checkpoint_dir "$CHECKPOINT_DIR" \
        --batch_size $BATCH_SIZE \
        --check_if_exists $CHECK_IF_EXISTS \
        --resume $RESUME \
        --overwrite_existing_collection $OVERWRITE &
done

# Wait for all processes to complete
wait

echo "All processes completed."
