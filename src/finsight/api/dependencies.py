from collections.abc import AsyncIterator

from finsight.clients.sec import SecClient


async def get_sec_client() -> AsyncIterator[SecClient]:
    async with SecClient() as client:
        yield client
