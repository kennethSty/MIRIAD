#!/bin/bash
docker run \
  -p 6333:6333 \
  -p 6334:6334 \
  -v /home/IAIS/kstyppa/agents/MIRIAD/qdrant_storage:/qdrant/storage:z \
  qdrant/qdrant
