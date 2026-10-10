from collections.abc import AsyncIterator

from finsight.clients.sec import SecClient
from finsight.core.config import get_settings
from finsight.storage.local import LocalDocumentStore


def get_document_store() -> LocalDocumentStore:
    settings = get_settings()
    return LocalDocumentStore(settings.raw_document_directory)

async def get_sec_client() -> AsyncIterator[SecClient]:
    async with SecClient() as client:
        yield client
