"""Generated compressed preview atlas for curated deep-zoom targets."""

from __future__ import annotations

import base64
from functools import lru_cache
import json
import zlib

from .target_thumbnail_chunk_0 import CHUNK as _CHUNK_0
from .target_thumbnail_chunk_1 import CHUNK as _CHUNK_1
from .target_thumbnail_chunk_2 import CHUNK as _CHUNK_2
from .target_thumbnail_chunk_3 import CHUNK as _CHUNK_3

THUMBNAIL_WIDTH = 48
THUMBNAIL_HEIGHT = 30

_ATLAS_B85 = _CHUNK_0 + _CHUNK_1 + _CHUNK_2 + _CHUNK_3


@lru_cache(maxsize=1)
def _atlas() -> dict[str, str]:
    raw = zlib.decompress(base64.b85decode(_ATLAS_B85.encode("ascii")))
    return json.loads(raw.decode("ascii"))


def thumbnail_bytes(target_id: str) -> bytes:
    try:
        return _atlas()[target_id].encode("ascii")
    except KeyError as error:
        raise KeyError(f"Unknown target thumbnail: {target_id}") from error


def thumbnail_ids() -> tuple[str, ...]:
    return tuple(_atlas())
