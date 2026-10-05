"""
Back up the files git ignores, and put them back after a clone.

WHY
    Some of what the pipeline reads is ignored by git on purpose: the local
    copies of the phone database's tables, the matchup cache, two smaller
    derived caches. They were ignored because they are large (790 MB between
    them) and can in principle be rebuilt - but rebuilding them needs the
    phone's database to be reachable, or half an hour of parsing, and two of
    them are over GitHub's 100 MB limit for a single file. A working folder
    that is about to be deleted takes all of them with it.

    So they are kept here compressed, at about 7% of their size, under
    local_backup/ - one .xz per file, mirrored by relative path, with a
    manifest of sizes and hashes. A fresh clone runs `restore` once and has
    everything the pipeline expects, where it expects it. The knowledge-graph
    output and the run logs ride along; they are small and were also local
    only. The graph's AST cache does not: /graphify rebuilds it for free, and
    its 64-character file names made paths long enough for git on Windows to
    refuse the checkout in a deep folder.

    Not backed up, deliberately: .env. It holds the database password and
    this repository is public. .env.example lists the keys it needs. For the
    same reason `pack` refuses a file in which a value from .env occurs - a
    log quoting a failed connection, say - rather than publish it. That
    check reads bytes: it sees a value written out in a text file or a
    pickle, not one inside a compressed file.

Run:  py tools/local_backup.py restore   (after a clone: put the files back)
      py tools/local_backup.py pack      (refresh the backups from the local files)
      py tools/local_backup.py verify    (do the backups still decompress to their recorded hashes?)
"""
import argparse
import glob
import hashlib
import json
import lzma
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKUP_DIR = "local_backup"
MANIFEST_NAME = "MANIFEST.json"
SUFFIX = ".xz"
# Measured on a 96 MB cache: preset 1 packs to 7.1% at 51 MB/s, preset 6 to
# 5.8% at 3 MB/s, gzip -6 to 10.3% at 29 MB/s.
PRESET = 1
CHUNK_BYTES = 1 << 20
MAX_PACKED_BYTES = 95 * 1024 * 1024        # GitHub refuses a single file over 100 MB
PATTERNS = ("phonedb_cache/*", "matchup_cache_v1.pkl", "team_box_cache_v1.pkl",
            "defensive_tracking_cache_v1.pkl", "graphify-out/*", "auto_push.log",
            "output/*.log", "sim_sandbox/*.log")
ENV_FILE = ".env"
MIN_SECRET_CHARS = 8                       # a shorter value (a port number, a short name) turns up by coincidence


def sha256_of(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def local_files():
    """Relative paths (forward slashes) of the non-empty files to back up."""
    found = set()
    for pattern in PATTERNS:
        for match in glob.glob(pattern, root_dir=PROJECT_ROOT, recursive=True, include_hidden=True):
            full = os.path.join(PROJECT_ROOT, match)
            if os.path.isfile(full) and os.path.getsize(full) > 0:
                found.add(match.replace(os.sep, "/"))
    return sorted(found)


def env_secrets():
    """The .env values long enough to search for, by key; empty when there is no .env."""
    path = os.path.join(PROJECT_ROOT, ENV_FILE)
    if not os.path.exists(path):
        return {}
    secrets = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            key, separator, value = line.strip().partition("=")
            value = value.strip().strip("\"'")
            if separator and not key.startswith("#") and len(value) >= MIN_SECRET_CHARS:
                secrets[key.strip()] = value.encode("utf-8")
    return secrets


def env_keys_found_in(path, secrets):
    """Keys of the .env values that occur in the file, read a chunk at a time."""
    if not secrets:
        return []
    overlap = max(len(value) for value in secrets.values()) - 1     # a value may straddle two chunks
    found, tail = set(), b""
    with open(path, "rb") as f:
        while chunk := f.read(CHUNK_BYTES):
            window = tail + chunk
            found.update(key for key, value in secrets.items() if value in window)
            tail = window[-overlap:]
    return sorted(found)


def refuse_leaks(relatives):
    """Stop before anything is packed if a file about to be published quotes a value from .env."""
    secrets = env_secrets()
    leaks = {}
    for relative in relatives:
        keys = env_keys_found_in(os.path.join(PROJECT_ROOT, *relative.split("/")), secrets)
        if keys:
            leaks[relative] = keys
    if leaks:
        listing = "; ".join(f"{relative} ({', '.join(keys)})" for relative, keys in leaks.items())
        raise SystemExit(f".env degeri gecen dosya paketlenmez, depo herkese acik: {listing}")


def packed_path(relative):
    return os.path.join(PROJECT_ROOT, BACKUP_DIR, *relative.split("/")) + SUFFIX


def manifest_path():
    return os.path.join(PROJECT_ROOT, BACKUP_DIR, MANIFEST_NAME)


def read_manifest():
    """The manifest, or an empty one when nothing has been packed yet."""
    if not os.path.exists(manifest_path()):
        return {}
    with open(manifest_path(), encoding="utf-8") as f:
        return json.load(f)


def pack_file(source, target):
    os.makedirs(os.path.dirname(target), exist_ok=True)
    partial = target + ".part"
    with open(source, "rb") as src, lzma.open(partial, "wb", preset=PRESET) as dst:
        while chunk := src.read(CHUNK_BYTES):
            dst.write(chunk)
    os.replace(partial, target)
    return os.path.getsize(target)


def unpack_to(source, target):
    """Decompress `source` to `target` through a partial file; return its hash."""
    os.makedirs(os.path.dirname(target), exist_ok=True)
    partial = target + ".part"
    digest = hashlib.sha256()
    with lzma.open(source, "rb") as src, open(partial, "wb") as dst:
        while chunk := src.read(CHUNK_BYTES):
            digest.update(chunk)
            dst.write(chunk)
    os.replace(partial, target)
    return digest.hexdigest()


def unpacked_hash(source):
    digest = hashlib.sha256()
    with lzma.open(source, "rb") as src:
        while chunk := src.read(CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def pack():
    """Bring local_backup/ in line with the local files; return the manifest."""
    old = read_manifest()
    relatives = local_files()
    refuse_leaks(relatives)
    manifest = {}
    t0 = time.time()
    for relative in relatives:
        source = os.path.join(PROJECT_ROOT, *relative.split("/"))
        target = packed_path(relative)
        digest, size = sha256_of(source), os.path.getsize(source)
        unchanged = old.get(relative, {}).get("sha256") == digest and os.path.exists(target)
        packed = os.path.getsize(target) if unchanged else pack_file(source, target)
        if packed > MAX_PACKED_BYTES:
            raise RuntimeError(f"{relative} sikistirilinca {packed / 1e6:.0f} MB - GitHub sinirinin "
                               f"ustunde, parcalanmasi gerekir")
        manifest[relative] = {"size": size, "sha256": digest, "packed_size": packed}
        print(f"  {'ayni ' if unchanged else 'paket'}  {size / 1e6:8.1f} MB -> {packed / 1e6:6.1f} MB  {relative}")
    for relative in sorted(set(old) - set(manifest)):          # no longer present locally
        stale = packed_path(relative)
        if os.path.exists(stale):
            os.remove(stale)
        print(f"  cikarildi  {relative}")
    os.makedirs(os.path.dirname(manifest_path()), exist_ok=True)
    with open(manifest_path(), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, sort_keys=True)
    total = sum(e["size"] for e in manifest.values())
    packed_total = sum(e["packed_size"] for e in manifest.values())
    print(f"{len(manifest)} dosya, {total / 1e6:.0f} MB -> {packed_total / 1e6:.0f} MB "
          f"({time.time() - t0:.0f} sn)")
    return manifest


def restore(target_root, force):
    """Put every backed-up file back under `target_root`; return how many were written."""
    manifest = read_manifest()
    if not manifest:
        raise SystemExit(f"{BACKUP_DIR}/{MANIFEST_NAME} yok - geri yuklenecek yedek bulunamadi")
    written = 0
    for relative, entry in sorted(manifest.items()):
        target = os.path.join(target_root, *relative.split("/"))
        if os.path.exists(target) and not force:
            same = os.path.getsize(target) == entry["size"]
            print(f"  {'yerinde' if same else 'FARKLI '}  {relative}"
                  + ("" if same else "  (uzerine yazmak icin --force)"))
            continue
        digest = unpack_to(packed_path(relative), target)
        if digest != entry["sha256"]:
            os.remove(target)
            raise RuntimeError(f"{relative}: acilan dosyanin ozeti manifestle uyusmuyor - yedek bozuk")
        written += 1
        print(f"  yazildi  {entry['size'] / 1e6:8.1f} MB  {relative}")
    print(f"{written} dosya geri yuklendi, {len(manifest) - written} dosya zaten yerindeydi")
    return written


def verify():
    """Every backup decompresses to its recorded hash; return the number that do not."""
    manifest = read_manifest()
    if not manifest:
        raise SystemExit(f"{BACKUP_DIR}/{MANIFEST_NAME} yok - dogrulanacak yedek bulunamadi")
    bad = 0
    for relative, entry in sorted(manifest.items()):
        source = packed_path(relative)
        try:
            intact = os.path.exists(source) and unpacked_hash(source) == entry["sha256"]
        except (lzma.LZMAError, EOFError):
            intact = False
        bad += 0 if intact else 1
        if not intact:
            print(f"  BOZUK  {relative}")
    print(f"{len(manifest) - bad}/{len(manifest)} yedek saglam")
    return bad


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("pack", "restore", "verify"))
    parser.add_argument("--target", default=PROJECT_ROOT,
                        help="restore: dosyalarin yazilacagi kok (varsayilan: proje klasoru)")
    parser.add_argument("--force", action="store_true",
                        help="restore: yerinde duran dosyalarin uzerine yaz")
    args = parser.parse_args()
    if args.action == "pack":
        pack()
        return 0
    if args.action == "restore":
        restore(os.path.abspath(args.target), args.force)
        return 0
    return 1 if verify() else 0


if __name__ == "__main__":
    sys.exit(main())
