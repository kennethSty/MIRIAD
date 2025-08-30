#!/bin/bash
echo "Starting Qdrant with NFS-safe configuration..."
docker run \
  --name qdrant-server \
  -p 6333:6333 \
  -p 6334:6334 \
  -v /home/IAIS/kstyppa/agents/MIRIAD/qdrant_storage:/qdrant/storage:z \
  -e QDRANT_DISABLE_TELEMETRY=true \
  --rm \
  qdrant/qdrant
