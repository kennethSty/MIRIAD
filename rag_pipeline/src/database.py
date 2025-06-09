from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import Filter, FieldCondition, Range, Match
from qdrant_client.http.models import PointStruct
from qdrant_client.http import models
from termcolor import colored

import uuid

def generate_uuid():
    return str(uuid.uuid4())

class QdrantHandler:
    def __init__(
        self,
        qdrant_host,
        qdrant_port,
        db_name,
        add_to_existing=False,
        distance="cosine",
    ):
        self.client = QdrantClient(qdrant_host, port=qdrant_port)
        self.collection_name = db_name
        self.distance = distance

        self._check_qdrant()
        if add_to_existing == False and self._collection_exists():
            raise Exception(
                f"Collection '{self.collection_name}' already exists. Set 'add_to_existing=True' in the config to add to existing collection, or run delete_collection('{self.collection_name}')."
            )

    def _check_qdrant(self):
        try:
            status = self.client.list_full_snapshots()
        except:
            # raise Exception("Qdrant is not running. Please start Qdrant by running `./qdrant.sh` and try again.")
            raise(
                colored(
                    "Error: Qdrant is not running. Please start Qdrant by running `./qdrant.sh` and try again.",
                    "red",
                )
            )

    def _collection_exists(self):
        try:
            status = self.client.get_collection(self.collection_name)
            return True
        except:
            return False

    def delete_collection(self, collection_name):
        return self.client.delete_collection(collection_name)

    def point_exists(self, qa_id):
        # check if entry already exists in db, and skip if it does
        if self._collection_exists() == False:
            return False
        qa_id_value = qa_id  # Make sure this is the correct type
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="qa_id",
                        match=models.MatchValue(value=qa_id_value),
                    ),
                ]
            ),
        )
        results = results[0]
        # Check if any entry is found

        return len(results) > 0

    def upsert(self, embeddings, batch_data, offset, check_if_exists=False):
        assert len(batch_data) == len(embeddings)

        if not self._collection_exists():
            distance_fn = {
                "cosine": Distance.COSINE,
                "dot": Distance.DOT,
                "euclidian": Distance.EUCLID,
                "manhattan": Distance.MANHATTAN,
            }

            DIM_SIZE = embeddings.shape[1]
            DISTANCE_FN = distance_fn[self.distance]

            print(
                f"Database {self.collection_name} does not exist. Creating now with {DIM_SIZE} dimensions and distance metric {self.distance}..."
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=DIM_SIZE, distance=DISTANCE_FN, on_disk=True),
                hnsw_config=models.HnswConfigDiff(on_disk=True),
            )

        points = []
        for i, (emb, metadata) in enumerate(zip(embeddings, batch_data)):
            if check_if_exists:
                # check if entry already exists in db, and skip if it does
                already_exists = self.point_exists(metadata["qa_id"])

                # Check if any entry is found
                if already_exists:
                    print("Entry already exists in database. Skipping...")
                    continue

            point = PointStruct(
                id=generate_uuid(), # generate globally unique uuid for each point
                vector=emb,
                payload={
                    "paper_title": metadata["paper_title"],
                    "question": metadata["question"],
                    "answer": metadata["answer"],
                    "qa_id": metadata["qa_id"],
                },
            )

            points.append(point)

        self.client.upsert(
            collection_name=self.collection_name, wait=True, points=points
        )


class QdrantHandlerPassageText:
    def __init__(
        self,
        qdrant_host,
        qdrant_port,
        db_name,
        add_to_existing=False,
        distance="cosine",
    ):
        self.client = QdrantClient(qdrant_host, port=qdrant_port)
        self.collection_name = db_name
        self.distance = distance

        self._check_qdrant()
        if add_to_existing == False and self._collection_exists():
            raise Exception(
                f"Collection '{self.collection_name}' already exists. Set 'add_to_existing=True' in the config to add to existing collection, or run delete_collection('{self.collection_name}')."
            )

    def _check_qdrant(self):
        try:
            status = self.client.list_full_snapshots()
        except:
            colored(
                "Error: Qdrant is not running. Please start Qdrant by running `./qdrant.sh` and try again.",
                "red",
            )

    def _collection_exists(self):
        try:
            status = self.client.get_collection(self.collection_name)
            return True
        except:
            return False

    def delete_collection(self, collection_name):
        return self.client.delete_collection(collection_name)

    def point_exists(self, passage_chunk_id):
        # check if entry already exists in db, and skip if it does
        if self._collection_exists() == False:
            return False
        passage_chunk_id_value = passage_chunk_id  # Make sure this is the correct type
        results = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="passage_chunk_id",
                        match=models.MatchValue(value=passage_chunk_id_value),
                    ),
                ]
            ),
        )
        results = results[0]
        # Check if any entry is found

        return len(results) > 0

    def upsert(self, embeddings, passage_chunk_ids, decoded_texts, check_if_exists=False):
        assert len(passage_chunk_ids) == len(embeddings)
        assert len(passage_chunk_ids) == len(decoded_texts)

        if not self._collection_exists():
            distance_fn = {
                "cosine": Distance.COSINE,
                "dot": Distance.DOT,
                "euclidian": Distance.EUCLID,
                "manhattan": Distance.MANHATTAN,
            }

            DIM_SIZE = embeddings.shape[1]
            DISTANCE_FN = distance_fn[self.distance]

            print(
                f"Database {self.collection_name} does not exist. Creating now with {DIM_SIZE} dimensions and distance metric {self.distance}..."
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=DIM_SIZE, distance=DISTANCE_FN, on_disk=True),
                hnsw_config=models.HnswConfigDiff(on_disk=True),
            )

        points = []
        for i, (emb, passage_chunk_id, text) in enumerate(zip(embeddings, passage_chunk_ids, decoded_texts)):
            if check_if_exists:
                # check if entry already exists in db, and skip if it does
                already_exists = self.point_exists(passage_chunk_id)

                # Check if any entry is found
                if already_exists:
                    print(f"Entry {passage_chunk_id} already exists in database. Skipping...")
                    continue

            point = PointStruct(
                id=generate_uuid(), # generate globally unique uuid for each point
                vector=emb,
                payload={
                    "passage_chunk_id": passage_chunk_id,
                    "passage_text": text,
                },
            )

            points.append(point)
            
            if len(points) % 1000 == 0 and len(points) != 0:
                self.client.upsert(
                    collection_name=self.collection_name, wait=True, points=points
                )
                points = []
                if len(points) % 200000 == 0:
                    print(f"Upserted {i+1} points!")
