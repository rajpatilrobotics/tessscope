"""Range-based ZIP access for extracting only selected BBBC006 channel members."""

from __future__ import annotations

import binascii
import io
import struct
import time
import zlib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipInfo

LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
LOCAL_FILE_HEADER_SIGNATURE = 0x04034B50


def fetch_range(
    url: str,
    start: int,
    end: int,
    *,
    timeout: float = 60.0,
    attempts: int = 4,
) -> bytes:
    """Fetch one inclusive HTTP byte range with bounded retries."""
    if start < 0 or end < start:
        raise ValueError(f"Invalid HTTP byte range: {start}-{end}")
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            request = Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urlopen(request, timeout=timeout) as response:
                payload = response.read()
                status = getattr(response, "status", None)
                content_range = response.headers.get("Content-Range", "")
            expected = end - start + 1
            if status != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                raise OSError(
                    f"Server did not honor range {start}-{end}: "
                    f"status={status}, content-range={content_range!r}"
                )
            if len(payload) != expected:
                raise OSError(f"Expected {expected} range bytes, received {len(payload)}")
            return payload
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            last_error = error
            if attempt + 1 < attempts:
                time.sleep(2**attempt)
    raise OSError(f"Failed HTTP range {start}-{end} from {url}") from last_error


class HTTPRangeFile(io.RawIOBase):
    """Small seekable view used only to read a remote ZIP central directory."""

    def __init__(self, url: str, size: int, block_bytes: int = 1024 * 1024) -> None:
        if size <= 0 or block_bytes <= 0:
            raise ValueError("Remote file size and block size must be positive")
        self.url = url
        self.size = size
        self.block_bytes = block_bytes
        self.position = 0
        self.cache: dict[int, bytes] = {}

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_CUR:
            offset += self.position
        elif whence == io.SEEK_END:
            offset += self.size
        elif whence != io.SEEK_SET:
            raise ValueError(f"Unknown seek mode: {whence}")
        self.position = max(0, min(self.size, offset))
        return self.position

    def readinto(self, buffer: bytearray | memoryview) -> int:
        if self.position >= self.size:
            return 0
        requested = min(len(buffer), self.size - self.position)
        completed = 0
        while completed < requested:
            block_index = self.position // self.block_bytes
            if block_index not in self.cache:
                start = block_index * self.block_bytes
                end = min(self.size, start + self.block_bytes) - 1
                self.cache[block_index] = fetch_range(self.url, start, end)
            block = self.cache[block_index]
            offset = self.position % self.block_bytes
            count = min(requested - completed, len(block) - offset)
            buffer[completed : completed + count] = block[offset : offset + count]
            self.position += count
            completed += count
        return completed


def local_member_range(info: ZipInfo, extra_allowance: int = 4096) -> tuple[int, int]:
    """Return a one-request range that normally includes header and compressed bytes."""
    encoded_name = info.filename.encode("utf-8" if info.flag_bits & 0x800 else "cp437")
    start = info.header_offset
    end = start + LOCAL_HEADER.size + len(encoded_name) + extra_allowance + info.compress_size
    return start, end - 1


def decode_local_member(payload: bytes, info: ZipInfo) -> bytes:
    """Validate one local ZIP header, decompress its bytes, and verify CRC32."""
    if len(payload) < LOCAL_HEADER.size:
        raise BadZipFile(f"Truncated local header for {info.filename}")
    (
        signature,
        _extract_version,
        flags,
        compression,
        _modified_time,
        _modified_date,
        _header_crc,
        _header_compressed_size,
        _header_uncompressed_size,
        name_length,
        extra_length,
    ) = LOCAL_HEADER.unpack(payload[: LOCAL_HEADER.size])
    if signature != LOCAL_FILE_HEADER_SIGNATURE:
        raise BadZipFile(f"Wrong local header signature for {info.filename}")
    if flags & 0x1:
        raise BadZipFile(f"Encrypted ZIP member is unsupported: {info.filename}")
    data_start = LOCAL_HEADER.size + name_length + extra_length
    data_end = data_start + info.compress_size
    if len(payload) < data_end:
        raise EOFError(f"Need {data_end} bytes for {info.filename}, received {len(payload)}")
    encoded_name = payload[LOCAL_HEADER.size : LOCAL_HEADER.size + name_length]
    encoding = "utf-8" if flags & 0x800 else "cp437"
    if encoded_name.decode(encoding) != info.orig_filename:
        raise BadZipFile(f"Local/central filename mismatch for {info.filename}")
    compressed = payload[data_start:data_end]
    if compression == ZIP_STORED:
        value = compressed
    elif compression == ZIP_DEFLATED:
        value = zlib.decompress(compressed, -zlib.MAX_WBITS)
    else:
        raise BadZipFile(f"Unsupported compression method {compression}: {info.filename}")
    if len(value) != info.file_size:
        raise BadZipFile(
            f"Uncompressed size mismatch for {info.filename}: {len(value)} != {info.file_size}"
        )
    crc = binascii.crc32(value) & 0xFFFFFFFF
    if crc != info.CRC:
        raise BadZipFile(f"CRC mismatch for {info.filename}: {crc:08x} != {info.CRC:08x}")
    return value


def fetch_member(url: str, info: ZipInfo) -> bytes:
    """Fetch and decode one selected member without downloading its full archive."""
    start, end = local_member_range(info)
    payload = fetch_range(url, start, end)
    try:
        return decode_local_member(payload, info)
    except EOFError:
        # A local extra field longer than the allowance is rare but valid.
        header = fetch_range(url, info.header_offset, info.header_offset + 65535)
        fields = LOCAL_HEADER.unpack(header[: LOCAL_HEADER.size])
        name_length, extra_length = fields[-2:]
        exact_end = (
            info.header_offset
            + LOCAL_HEADER.size
            + name_length
            + extra_length
            + info.compress_size
            - 1
        )
        return decode_local_member(fetch_range(url, info.header_offset, exact_end), info)


def atomic_write(path: Path, payload: bytes) -> None:
    """Write one generated external asset without exposing a partial final file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_bytes(payload)
    temporary.replace(path)
