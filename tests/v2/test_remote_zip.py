"""Local reference tests for selective remote-ZIP member decoding."""

from __future__ import annotations

import io
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

import pytest

from tessscope.v2.data.remote_zip import decode_local_member, local_member_range


@pytest.mark.parametrize("compression", [ZIP_STORED, ZIP_DEFLATED])
def test_decode_local_member_matches_zipfile(compression: int) -> None:
    expected = (b"range-based BBBC006 member\n" * 1000) + bytes(range(256))
    stream = io.BytesIO()
    with ZipFile(stream, mode="w", compression=compression) as zip_file:
        zip_file.writestr("nested/example.bin", expected)
    stream.seek(0)
    with ZipFile(stream) as zip_file:
        info = zip_file.getinfo("nested/example.bin")
        payload = stream.getvalue()[info.header_offset :]
    assert decode_local_member(payload, info) == expected
    start, end = local_member_range(info)
    assert start == info.header_offset
    assert end > start + info.compress_size


def test_decode_local_member_rejects_truncated_payload() -> None:
    stream = io.BytesIO()
    with ZipFile(stream, mode="w", compression=ZIP_DEFLATED) as zip_file:
        zip_file.writestr("example.bin", b"payload" * 100)
    stream.seek(0)
    with ZipFile(stream) as zip_file:
        info = zip_file.getinfo("example.bin")
    with pytest.raises(EOFError):
        decode_local_member(stream.getvalue()[info.header_offset : info.header_offset + 35], info)
