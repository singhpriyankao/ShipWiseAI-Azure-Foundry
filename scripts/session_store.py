"""
Where the Orchestrator keeps one PM's session between messages: the feature
being evaluated, the constraints so far, and the sub-agent results already
produced. This is what lets "here are my targets" attach to a feature the PM
described in an earlier message, and what the second round's follow-ups will
build on.

Separate from the `recommendations` container on purpose. That one holds
accepted recommendations waiting on an outcome; this one holds working state
that changes every turn and is not an outcome record.

CosmosSessionStore needs the container to exist first (created by hand in the
Portal: database `shipwiseai`, container `sessions`, partition key
`/session_id`). InMemorySessionStore is for tests and for running the
Orchestrator without it.
"""
import os
from datetime import datetime, timezone

CONTAINER_NAME = "sessions"


class InMemorySessionStore:
    def __init__(self):
        self._sessions = {}

    def get(self, session_id):
        return self._sessions.get(session_id)

    def put(self, session_id, state):
        self._sessions[session_id] = state


class CosmosSessionStore:
    def __init__(self, endpoint=None, key=None):
        from azure.cosmos import CosmosClient
        client = CosmosClient(endpoint or os.environ["COSMOS_ENDPOINT"], key or os.environ["COSMOS_KEY"])
        self._container = client.get_database_client("shipwiseai").get_container_client(CONTAINER_NAME)

    def get(self, session_id):
        from azure.cosmos.exceptions import CosmosResourceNotFoundError
        try:
            doc = self._container.read_item(item=session_id, partition_key=session_id)
        except CosmosResourceNotFoundError:
            return None
        return doc.get("state")

    def put(self, session_id, state):
        self._container.upsert_item({
            "id": session_id,
            "session_id": session_id,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "state": state,
        })


# --------------------------------------------------------------------------
# Sessions for the hosted app
# --------------------------------------------------------------------------

SESSION_TYPE = "session"


class OwnedSessionStore:
    """The hosted app's store: a session belongs to the user who made it. `get` returns nothing for anyone else,
    so one user can never read or continue another's evaluation by guessing its id. This in-memory version is
    for tests; SharedContainerSessionStore is the one the server uses."""

    def __init__(self):
        self._docs = {}

    def get(self, session_id, user_id):
        doc = self._docs.get(session_id)
        return doc["state"] if doc and doc["user_id"] == user_id else None

    def put(self, session_id, user_id, state):
        existing = self._docs.get(session_id)
        if existing and existing["user_id"] != user_id:
            raise PermissionError("that session belongs to another user")
        self._docs[session_id] = {"user_id": user_id, "state": state}


class SharedContainerSessionStore:
    """Sessions as documents of type "session" in the `recommendations` container (partition key /id), because the
    account is on provisioned throughput at its 1,000 RU/s free-tier limit and a new container would be refused
    or billed. The document id is "session:<session id>"."""

    def __init__(self, endpoint=None, key=None):
        from azure.cosmos import CosmosClient
        client = CosmosClient(endpoint or os.environ["COSMOS_ENDPOINT"], key or os.environ["COSMOS_KEY"])
        self._container = client.get_database_client("shipwiseai").get_container_client("recommendations")

    @staticmethod
    def _doc_id(session_id):
        return f"session:{session_id}"

    def _read(self, session_id):
        from azure.cosmos.exceptions import CosmosResourceNotFoundError
        doc_id = self._doc_id(session_id)
        try:
            return self._container.read_item(item=doc_id, partition_key=doc_id)
        except CosmosResourceNotFoundError:
            return None

    def get(self, session_id, user_id):
        doc = self._read(session_id)
        return doc.get("state") if doc and doc.get("user_id") == user_id else None

    def put(self, session_id, user_id, state):
        existing = self._read(session_id)
        if existing and existing.get("user_id") != user_id:
            raise PermissionError("that session belongs to another user")
        self._container.upsert_item({"id": self._doc_id(session_id), "type": SESSION_TYPE, "session_id": session_id, "user_id": user_id,
                                     "updated_at": datetime.now(timezone.utc).isoformat(), "state": state})
