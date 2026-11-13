"""VALENC1, the same engine envelope the desktop and phone clients already use.

Header: 8-byte magic b"VALENC1\\x00", uint32 chunk size.
Each chunk: uint32 cipher_len (ciphertext + 16-byte GCM tag), 12-byte nonce, cipher bytes.
"""

from __future__ import annotations

import os
import struct
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"VALENC1\x00"
CHUNK = 1 << 20
DEFAULT_KEY = Path(os.environ.get("VALHALLA_KEY_FILE", r"C:\Ai\valhalla-key.bin"))


def load_key(path: Path | None = None) -> bytes:
    key_path = path or DEFAULT_KEY
    data = key_path.read_bytes()
    if len(data) != 32:
        raise RuntimeError(f"VALENC1 密钥必须是 32 字节，当前是 {len(data)}。")
    return data


def encrypt_file(src: Path, dest: Path, key: bytes) -> None:
    aes = AESGCM(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fin, dest.open("wb") as fout:
        fout.write(MAGIC)
        fout.write(struct.pack("<I", CHUNK))
        while True:
            plain = fin.read(CHUNK)
            if not plain:
                break
            nonce = os.urandom(12)
            cipher = aes.encrypt(nonce, plain, MAGIC)
            fout.write(struct.pack("<I", len(cipher)))
            fout.write(nonce)
            fout.write(cipher)


def decrypt_file(src: Path, dest: Path, key: bytes) -> None:
    aes = AESGCM(key)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as fin, dest.open("wb") as fout:
        magic = fin.read(8)
        if magic != MAGIC:
            raise RuntimeError("不是 VALENC1 加密文件。")
        (chunk_size,) = struct.unpack("<I", fin.read(4))
        if chunk_size == 0 or chunk_size > (16 << 20):
            raise RuntimeError("VALENC1 分块大小异常。")
        while True:
            head = fin.read(4)
            if not head:
                break
            if len(head) != 4:
                raise RuntimeError("加密文件不完整。")
            (clen,) = struct.unpack("<I", head)
            if clen <= 16 or clen - 16 > chunk_size:
                raise RuntimeError("加密文件分块长度异常。")
            nonce = fin.read(12)
            cipher = fin.read(clen)
            if len(nonce) != 12 or len(cipher) != clen:
                raise RuntimeError("加密文件不完整。")
            plain = aes.decrypt(nonce, cipher, MAGIC)
            fout.write(plain)
