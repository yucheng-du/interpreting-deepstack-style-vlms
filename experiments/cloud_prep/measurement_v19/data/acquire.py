#!/usr/bin/env python3
"""Cloud acquisition of GQA and DocVQA for the v19 confirmation / training data (CPU only; resumable).

Specify each target directory on the command line. Hashes and sizes come from
sources.json next to this file (--sources to override). Each subcommand writes <out>/receipt-<cmd>.json.

  gqa-questions  --out DIR                      Stanford questions1.2.zip (resumable, size check), extracts
                                                val/train balanced JSON (sha256 check where recorded)
  gqa-images     --split val|train --ids IDS --zip-index gqa_image_zip_index.json --out DIR [--source hf|zip]
                                                writes DIR/<imageId>.jpg for the listed ids only, each accepted on
                                                the official zip member's CRC32 and size. hf: pinned parquet shards
                                                of lmms-lab-encoder/GQA (sha256-checked); zip: HTTP range reads of the
                                                official images.zip members (no 21.8 GB download)
  docvqa-mirror  --split validation|train --out DIR [--columns-only] [--pages IDS]
                                                lmms-lab-encoder/DocVQA parquet -> DIR/<split>_annotations.json in
                                                the official Task-1 layout, plus DIR/<split>/documents/<page>.png for
                                                the listed pages (all pages without --pages). --columns-only reads
                                                the text columns by HTTP range requests (a few MB) plus only the
                                                row groups (100 rows, ~17 MB) that hold the listed pages
  docvqa-official --raw DIR --out DIR [--pages IDS]
                                                the user-placed RRC spdocvqa_qas.zip / spdocvqa_images.tar.gz (login
                                                needed, never downloaded here) -> same layout as docvqa-mirror

IDS is a text file (one id per line), a JSON list, or a pool/manifest JSONL (image_id / page fields are read);
--first N keeps the first N rows of a JSONL (the first N units of an ordered pool).
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import sys
import time
import urllib.request
import zipfile
import zlib

HERE = Path(__file__).resolve().parent
UA = {'User-Agent': 'research-data-fetch/1.0'}


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''): h.update(chunk)
    return h.hexdigest()


def fetch(url, dest, size, tries=60, wait=20):
    """Resumable single-connection download; done when the file has the expected size."""
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    for _ in range(tries):
        have = dest.stat().st_size if dest.exists() else 0
        if have == size: return dest
        if have > size: dest.unlink(); have = 0
        try:
            req = urllib.request.Request(url, headers={**UA, 'Range': f'bytes={have}-'})
            with urllib.request.urlopen(req, timeout=60) as r, dest.open('ab' if have and r.status == 206 else 'wb') as f:
                for chunk in iter(lambda: r.read(1 << 22), b''): f.write(chunk)
        except Exception as e:  # noqa: BLE001 - retried
            print(f'fetch retry after {type(e).__name__}: {e}', file=sys.stderr); time.sleep(wait)
    raise RuntimeError(f'{url}: incomplete after {tries} tries')


def byte_range(url, start, length, tries=10):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={**UA, 'Range': f'bytes={start}-{start + length - 1}'})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if r.status == 206 and len(data) == length: return data
            raise ValueError(f'range status {r.status}, {len(data)} bytes')
        except Exception as e:  # noqa: BLE001
            if i == tries - 1: raise
            print(f'range retry after {type(e).__name__}: {e}', file=sys.stderr); time.sleep(2 + 3 * i)


def read_ids(path, key='image_id', first=None):
    """Ids from text lines, a JSON list, or a JSONL (pool rows: image_id / pages; manifest rows: benchmark.image_id).
    first: only the first n JSONL rows (the first n units of an ordered pool)."""
    text = Path(path).read_text()
    if str(path).endswith('.jsonl'):
        out = []
        for line in [x for x in text.split('\n') if x.strip()][:first]:
            r = json.loads(line)
            if 'pages' in r: out += r['pages']
            elif key in r: out.append(str(r[key]))
            elif 'benchmark' in r: out.append(str(r['benchmark']['image_id']))
        return list(dict.fromkeys(out))
    try:
        v = json.loads(text)
        if isinstance(v, list): return [str(x) for x in v]
    except json.JSONDecodeError:
        pass
    return [s.strip() for s in text.splitlines() if s.strip() and not s.startswith('#')]


def write_receipt(out, name, value):
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / f'receipt-{name}.json').write_text(json.dumps(value, indent=1) + '\n')
    print(json.dumps(value)[:2000])


def hf_file(repo, revision, filename, expect, local_dir):
    """Pinned HF dataset file via huggingface_hub (resumable), accepted on the recorded sha256 and size."""
    from huggingface_hub import hf_hub_download
    path = Path(local_dir) / filename
    if not (path.exists() and path.stat().st_size == expect[1] and sha_file(path) == expect[0]):
        path = Path(hf_hub_download(repo_id=repo, repo_type='dataset', revision=revision, filename=filename, local_dir=local_dir))
        got = sha_file(path)
        if got != expect[0] or path.stat().st_size != expect[1]: raise ValueError(f'{filename}: sha256 {got} != {expect[0]}')
    return path


def gqa_questions(a, src):
    s = src['gqa_questions']; out = Path(a.out)
    z = fetch(s['url'], out / 'raw' / 'questions1.2.zip', s['bytes'])
    got = {}
    with zipfile.ZipFile(z) as zf:
        for member in a.members or list(s['members']):
            target = out / member
            if not (target.exists() and target.stat().st_size == zf.getinfo(member).file_size):
                with zf.open(member) as r, open(target, 'wb') as w:
                    for chunk in iter(lambda: r.read(1 << 22), b''): w.write(chunk)
            got[member] = sha_file(target)
            want = s['members'].get(member)
            if want and got[member] != want: raise ValueError(f'{member}: sha256 {got[member]} != recorded {want}')
    write_receipt(out, 'gqa-questions', dict(url=s['url'], zip_bytes=s['bytes'], members_sha256=got, recorded=s['members']))


def gqa_images(a, src):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    idx = json.load(open(a.zip_index))['members']
    need = [i for i in read_ids(a.ids, first=a.first) if not (out / f'{i}.jpg').exists()
            or (out / f'{i}.jpg').stat().st_size != idx.get(i, {}).get('bytes')]
    unknown = [i for i in need if i not in idx]
    need = set(need) - set(unknown)
    ok, bad = 0, []
    def accept(i, data):
        nonlocal ok
        m = idx[i]
        if len(data) == m['bytes'] and zlib.crc32(data) == m['crc32']:
            (out / f'{i}.jpg').write_bytes(data); ok += 1; return True
        bad.append(i); return False
    if a.source == 'hf':
        import pyarrow.parquet as pq
        s = src['gqa_images_hf']
        for fname, expect in s['files'][a.split].items():
            if not need: break
            path = hf_file(s['repo'], s['revision'], fname, expect, a.hf_dir or out.parent / 'hf-gqa')
            for b in pq.ParquetFile(path).iter_batches(batch_size=64, columns=['id', 'image']):
                for i, im in zip(b.column('id').to_pylist(), b.column('image').to_pylist()):
                    if i in need and accept(i, im['bytes']): need.discard(i)
            if a.delete_shards: os.remove(path)
    else:
        url = src['gqa_images_zip']['url']
        for i in sorted(need):
            m = idx[i]
            head = struct.unpack('<4s5H3L2H', byte_range(url, m['offset'], 30))
            if head[0] != b'PK\x03\x04': bad.append(i); continue
            data = byte_range(url, m['offset'] + 30 + head[-2] + head[-1], m['compressed_bytes'])
            data = zlib.decompress(data, -15) if m['compression'] == 8 else data
            if accept(i, data): need.discard(i)
            time.sleep(a.sleep)
    write_receipt(out, f'gqa-images-{a.split}', dict(source=a.source, written=ok, crc_mismatch=bad[:50], n_crc_mismatch=len(bad),
                                                     missing=sorted(need)[:50], n_missing=len(need), not_in_zip_index=unknown[:50],
                                                     n_not_in_zip_index=len(unknown)))
    if need or bad or unknown: sys.exit(3)


DOC_COLS = ['questionId', 'question', 'question_types', 'docId', 'ucsf_document_id', 'ucsf_document_page_no', 'answers', 'data_split']


def doc_row(r, page):
    return dict(questionId=int(r['questionId']) if str(r['questionId']).isdigit() else r['questionId'], question=r['question'],
                question_types=list(r['question_types'] or []), image=f'documents/{page}.png', docId=r['docId'],
                ucsf_document_id=r['ucsf_document_id'], ucsf_document_page_no=r['ucsf_document_page_no'],
                answers=list(r['answers']), data_split=r['data_split'])


def page_of(r, path=None):
    """Page stem: the mirror's image.path stem when present, else <ucsf_document_id>_<ucsf_document_page_no>."""
    stem = Path(path).stem if path else ''
    return stem or f"{r['ucsf_document_id']}_{r['ucsf_document_page_no']}"


def docvqa_mirror(a, src):
    import pyarrow.parquet as pq
    s = src['docvqa_hf']; out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    pages = set(read_ids(a.pages, first=a.first)) if a.pages else None
    rows, written, mismatch, files = [], 0, [], {}
    docdir = out / a.split / 'documents'
    for fname, expect in s['files'][a.split].items():
        if a.columns_only:
            import fsspec
            url = f"https://huggingface.co/datasets/{s['repo']}/resolve/{s['revision']}/{fname}"
            with fsspec.filesystem('http', headers=UA).open(url, block_size=1 << 20) as f:
                pf = pq.ParquetFile(f)
                t = pf.read(columns=DOC_COLS + ['image.path']).to_pylist()
                files[fname] = dict(expected_sha256=expect[0], read='text columns (and row groups of requested pages) via HTTP range')
                start, groups = 0, []
                for g in range(pf.metadata.num_row_groups):
                    n = pf.metadata.row_group(g).num_rows
                    if pages and any(page_of(r, (r.get('image') or {}).get('path')) in pages for r in t[start:start + n]): groups.append(g)
                    start += n
                for g in groups:  # only the row groups that hold requested pages
                    for r in pf.read_row_group(g, columns=['ucsf_document_id', 'ucsf_document_page_no', 'image']).to_pylist():
                        page = page_of(r, r['image'].get('path')); target = docdir / f'{page}.png'
                        if page in pages and not target.exists():
                            target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(r['image']['bytes']); written += 1
            for r in t: rows.append(doc_row(r, page_of(r, (r.get('image') or {}).get('path'))))
            continue
        path = hf_file(s['repo'], s['revision'], fname, expect, a.hf_dir or out / 'hf-docvqa')
        files[fname] = dict(sha256=expect[0])
        for b in pq.ParquetFile(path).iter_batches(batch_size=16, columns=DOC_COLS + ['image']):
            for r in b.to_pylist():
                page = page_of(r, r['image'].get('path'))
                rows.append(doc_row(r, page))
                if pages is None or page in pages:
                    target = docdir / f'{page}.png'
                    data = r['image']['bytes']
                    if target.exists() and target.read_bytes() != data: mismatch.append(page)
                    elif not target.exists():
                        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(data); written += 1
        if a.delete_shards: os.remove(path)
    seen = {}
    for r in rows: seen.setdefault(r['questionId'], r)
    data = list(seen.values())
    split = {'validation': 'val'}.get(a.split, a.split)
    ann = dict(dataset_name='docvqa', dataset_split=split, dataset_version=a.dataset_version,
               provenance=dict(repo=s['repo'], revision=s['revision'], note='HF mirror rebuilt into the official Task-1 layout'),
               data=data)
    target = out / f'{a.split}_annotations.json'
    target.write_text(json.dumps(ann, ensure_ascii=False) + '\n')
    missing = sorted(pages - {Path(r['image']).stem for r in data}) if pages else []
    write_receipt(out, f'docvqa-mirror-{a.split}', dict(annotations=str(target), annotations_sha256=sha_file(target), files=files, questions=len(data),
                  duplicate_rows=len(rows) - len(data), documents=len({r['ucsf_document_id'] for r in data}),
                  pages=len({r['image'] for r in data}), pages_written=written, page_bytes_differ=mismatch[:50],
                  requested_pages_absent=missing[:50], n_requested_pages_absent=len(missing), columns_only=a.columns_only))


def docvqa_official(a, src):
    import tarfile
    s = src['docvqa_official']; raw = Path(a.raw); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    qas, tar = raw / s['qas_zip'][0], raw / s['images_tar_bytes'][0]
    if sha_file(qas) != s['qas_zip'][1]: raise ValueError('spdocvqa_qas.zip sha256 differs from the v14 acquisition')
    if tar.stat().st_size != s['images_tar_bytes'][1]: raise ValueError('spdocvqa_images.tar.gz size differs')
    got = {}
    with zipfile.ZipFile(qas) as z:
        names = {Path(n).name: n for n in z.namelist()}
        for member, want in (s['val_member'], s['train_member']):
            if member not in names: continue
            (out / member).write_bytes(z.read(names[member])); got[member] = sha_file(out / member)
            if want and got[member] != want: raise ValueError(f'{member}: sha256 differs')
    pages = set(read_ids(a.pages, first=a.first)) if a.pages else set(Path(r['image']).stem for r in json.load(open(out / s['val_member'][0]))['data'])
    written = 0
    with tarfile.open(tar, 'r|gz') as tf:
        for m in tf:
            stem = Path(m.name).stem
            if m.isfile() and stem in pages and m.name.endswith('.png'):
                target = out / 'documents' / f'{stem}.png'
                if not target.exists():
                    target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(tf.extractfile(m).read()); written += 1
    write_receipt(out, 'docvqa-official', dict(members_sha256=got, pages_requested=len(pages), pages_written=written))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--sources', type=Path, default=HERE / 'sources.json')
    sub = p.add_subparsers(dest='cmd', required=True)
    q = sub.add_parser('gqa-questions'); q.add_argument('--out', required=True); q.add_argument('--members', nargs='*')
    g = sub.add_parser('gqa-images'); g.add_argument('--split', choices=['val', 'train'], required=True)
    g.add_argument('--ids', required=True); g.add_argument('--zip-index', required=True); g.add_argument('--out', required=True)
    g.add_argument('--source', choices=['hf', 'zip'], default='hf'); g.add_argument('--hf-dir')
    g.add_argument('--delete-shards', action='store_true'); g.add_argument('--sleep', type=float, default=0.3)
    d = sub.add_parser('docvqa-mirror'); d.add_argument('--split', choices=['validation', 'train'], required=True)
    d.add_argument('--out', required=True); d.add_argument('--columns-only', action='store_true'); d.add_argument('--pages')
    d.add_argument('--hf-dir'); d.add_argument('--delete-shards', action='store_true'); d.add_argument('--dataset-version', default='1.0')
    o = sub.add_parser('docvqa-official'); o.add_argument('--raw', required=True); o.add_argument('--out', required=True); o.add_argument('--pages')
    for sp in (g, d, o): sp.add_argument('--first', type=int, help='only the first n rows of a pool/manifest JSONL id file')
    a = p.parse_args(argv)
    src = json.loads(a.sources.read_text())
    dict(**{'gqa-questions': gqa_questions, 'gqa-images': gqa_images, 'docvqa-mirror': docvqa_mirror,
            'docvqa-official': docvqa_official})[a.cmd](a, src)


if __name__ == '__main__':
    main()
