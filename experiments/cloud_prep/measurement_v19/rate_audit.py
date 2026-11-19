"""Rate audit with the real NNCodec encoder (nncodec env; CPU, no model).

For every saved unit tensor and QP: complete bitstream bytes, zero-coefficient fraction, relative MSE,
decode == torch replica (element-wise), decoded values are integer multiples of the QP step, the sha256
of the decoded float32 tensor, and encode/decode/replica seconds.

Subcommands
  code          code units of a runner extract dir (features*.jsonl + features/<unit>/<key>.npy) into
                OUT/rate_units[.shard-i-of-n].jsonl, one line per unit and pass (resumable: only missing
                (stream, QP) pairs are coded; new QPs can be added later while the features exist).
                QPs: --qps on every stream, plus per stream the QPs the --configs/--configs-file need on that
                stream (levels.resolve_config with --offsets); --union-qps codes the union on every stream.
                --workers N processes; --shard i/n (hash of unit id) for several machines;
                --follow polls while `runner --mode extract` is still writing (extract-and-code), and
                --delete-features removes a unit's .npy files once it is coded and verified, keeping the
                first --keep-features units (feature order) for device-check.
  finalize      merge OUT (+ --inputs DIRS, other machines) into the runner byte table and summaries:
                rate_per_unit.jsonl, bytes.json, summary.json, --csv table, means.npz (unless --skip-means).
                bytes.json 'qps' lists QPs coded for every unit and stream, 'qps_by_stream' the QPs complete
                per stream, 'qps_partial' the rest.
  device-check  model env (torch with CUDA, no nncodec): on kept features, CPU replica vs device replica
                bitwise, and CPU replica sha256 vs the NNCodec decode sha256 recorded by `code`.
  estimate      CPU hours for a planned audit from measured per-QP timings (or old summary totals).
  (no subcommand) legacy call: --features --out --csv [--qps] [--no-extend] [--skip-means] [--preprint]
                = code (one process) + finalize, same output files as before.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np

from cloud_prep.measurement_v19 import levels

BASE_QPS = [-32, -16, -12, -8, -4, -2, 0, 2, 4, 6, 8, 12]
PREPRINT = {-32: 58.41, -12: 87.41, 0: 98.09, 4: 99.54}
SUBCOMMANDS = ('code', 'finalize', 'device-check', 'estimate')


def display(stream):
    return 'later%d' % (int(stream[5:]) + 1) if stream.startswith('later') else stream


# ---- pure helpers ------------------------------------------------------------------------------

def unit_dir(features, unit_id):
    return Path(features) / 'features' / unit_id.replace('/', '_')


def fkey(rec):
    """Coded-image key of a features record: runner feature_key (= unit id for a unit's first image, as in
    every old extract dir; unit@sha16 for further DocVQA pages). Rate rows and bytes.json use it as 'unit'."""
    return rec.get('key', rec['unit_id'])


def read_features(features, tolerate_tail=False):
    """Records of every features*.jsonl (base first, then shards), first record per unit kept.
    tolerate_tail: ignore a partial last line (file still being written)."""
    features = Path(features)
    files = [p for p in [features / 'features.jsonl'] if p.exists()] + sorted(features.glob('features.shard-*-of-*.jsonl'))
    out, seen = [], set()
    for p in files:
        lines = p.read_text().splitlines()
        for n, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                if tolerate_tail and n == len(lines):
                    continue
                raise
            if fkey(r) not in seen:
                seen.add(fkey(r)); out.append(r)
    return out


def unit_shard(unit_id, shard):
    """Hash shard (independent of file order): sha256(unit) mod n == i."""
    if shard is None:
        return True
    i, n = shard
    return int(hashlib.sha256(unit_id.encode()).hexdigest()[:12], 16) % n == i


def parse_shard(text):
    if text is None:
        return None
    i, n = (int(x) for x in str(text).split('/'))
    if n < 1 or not 0 <= i < n:
        raise ValueError(f'--shard needs 0 <= i < n, got {text!r}')
    return i, n


def default_main_key(recs):
    ks = {k for r in recs for k in r['streams']}
    return 'layer0' if 'layer0' in ks else 'main'


def needed_qps(qps=(), configs=(), keys=None, main_key='main', offsets=None):
    """Sorted union of explicit QPs and the QPs the configs need on these keys."""
    need = {int(q) for q in qps}
    if configs:
        need |= set(levels.config_qps(configs, keys, main_key, offsets))
    return sorted(need)


def stream_qps(qps=(), configs=(), keys=None, main_key='main', offsets=None):
    """{key: sorted QPs}: explicit QPs on every key plus, per key, the QPs the configs resolve to there."""
    per = {k: {int(q) for q in qps} for k in keys}
    for c in configs:
        for k, lv in levels.resolve_config(c, keys, main_key, offsets).items():
            kind, qp = levels.parse_level(lv)
            if kind == 'q':
                per[k].add(qp)
    return {k: sorted(v) for k, v in per.items()}


def qps_for(qps, stream):
    """QP list of one stream from a list (all streams) or a {stream: list} map (unknown stream: union)."""
    if isinstance(qps, dict):
        return qps[stream] if stream in qps else sorted(set().union(*map(set, qps.values())))
    return qps


def extension_qps(qps, absmax, extend=True):
    """Old rule: past the largest QP, add every 4th QP until step > 2*absmax (all coefficients zero)."""
    qps = sorted(set(qps))
    if not extend or not qps:
        return qps
    q = max(qps)
    while levels.step_size(q) <= 2 * absmax:
        q += 4
        qps.append(q)
    return qps


def rate_files(out_dir):
    out_dir = Path(out_dir)
    return [p for p in [out_dir / 'rate_units.jsonl'] if p.exists()] + sorted(out_dir.glob('rate_units.shard-*-of-*.jsonl'))


def read_rate_lines(dirs, tolerate_tail=True):
    """{unit: [lines]} over rate_units files of several dirs (a truncated tail line is skipped)."""
    out = {}
    for d in dirs:
        for p in rate_files(d):
            lines = p.read_text().splitlines()
            for n, line in enumerate(lines, 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    if tolerate_tail and n == len(lines):
                        print(json.dumps(dict(warning='skipped partial tail line', file=str(p))), file=sys.stderr)
                        continue
                    raise
                out.setdefault(r['unit'], []).append(r)
    return out


def heal_tail(path):
    """Before appending: a last line without its newline (writer killed, disk full, VM crash) is dropped, or closed
    with a newline when it is complete JSON. Otherwise the next appended line would be glued to it and the file
    would no longer read (read_rate_lines tolerates only a partial *last* line). Returns the bytes dropped."""
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return 0
    with open(p, 'rb+') as f:
        size = f.seek(0, 2)
        f.seek(size - 1)
        if f.read(1) == b'\n':
            return 0
        pos, tail = size, b''
        while pos > 0 and b'\n' not in tail:
            step = min(1 << 16, pos)
            pos -= step
            f.seek(pos)
            tail = f.read(step) + tail
        cut = pos + tail.rfind(b'\n') + 1 if b'\n' in tail else 0
        f.seek(cut)
        last = f.read()
        try:
            json.loads(last)
            f.seek(0, 2); f.write(b'\n')
            return 0
        except ValueError:
            f.truncate(cut)
    print(json.dumps(dict(warning='dropped partial tail line before appending', file=str(p), bytes=size - cut)), file=sys.stderr)
    return size - cut


def coded_pairs(lines):
    return {(row['stream'], row['qp']) for l in lines for row in l['rows']}


def merge_unit(lines):
    """One unit's rows from several passes: {(stream, qp): row}, stream order as first seen; stats/means
    from the first line that has them. Raises if the same (stream, qp) has different bytes."""
    rows, order, stats, means = {}, [], None, None
    for l in lines:
        stats = stats or l.get('stats')
        means = means or l.get('means_file')
        for r in l['rows']:
            k = (r['stream'], r['qp'])
            if k in rows:
                if rows[k]['bytes'] != r['bytes']:
                    raise ValueError(f"{l['unit']} {k}: bytes differ between passes ({rows[k]['bytes']} vs {r['bytes']})")
                continue
            rows[k] = r
            if r['stream'] not in order:
                order.append(r['stream'])
    return rows, order, stats, means


def ordered_rows(rows, order):
    out = []
    for s in order:
        mine = [r for (st, q), r in rows.items() if st == s]
        out += sorted(mine, key=lambda r: (r['qp'] != 'raw', r['qp'] if r['qp'] != 'raw' else 0))
    return out


# ---- coding --------------------------------------------------------------------------------------

def _init_worker():
    import torch
    torch.set_num_threads(1)


def code_unit(job):
    """Worker: code one unit's missing (stream, QP) pairs. job = dict(features, rec, qps, done, means_dir)."""
    import torch
    from cloud_prep.measurement_v19 import codec
    t_unit = time.time()
    rec, qps, done = job['rec'], job['qps'], {tuple(x) for x in job['done']}
    uid = fkey(rec)
    d = unit_dir(job['features'], uid)
    rows, stats, mvec = [], {}, {}
    for k in rec['streams']:
        if k in rec['inactive']:
            continue
        x = np.load(d / f'{k}.npy')
        x = x.reshape(-1, x.shape[-1])            # 2-D [tokens, hidden]; leading dims flattened into tokens
        x64 = x.astype(np.float64)
        nrm = float((x64 ** 2).sum())
        stats[k] = dict(n=int(x.size), sum=float(x64.sum()), sumsq=nrm, absmax=float(np.abs(x).max()),
                        tokens=int(x.shape[0]), hidden=int(x.shape[1]))
        if job['means_dir'] is not None:
            mvec[k] = x.mean(axis=0)
        if (k, 'raw') not in done:
            rows.append(dict(unit=uid, stream=k, qp='raw', bytes=2 * x.size, numel=x.size, tokens=x.shape[0]))
        xt = torch.from_numpy(x)
        for qp in qps_for(qps, k):
            if (k, qp) in done:
                continue
            t0 = time.perf_counter(); bs = codec.encode(x, qp)
            t1 = time.perf_counter(); y = codec.decode(bs)
            t2 = time.perf_counter(); rep = levels.quantize(xt, qp).numpy()
            eq = bool(np.array_equal(y, rep))
            t3 = time.perf_counter()
            step = levels.step_size(qp)
            mult = bool(np.all(np.mod(y.astype(np.float64) / step, 1.0) == 0))
            err = float(((x64 - y) ** 2).sum())
            rows.append(dict(unit=uid, stream=k, qp=qp, step=step, bytes=len(bs), numel=x.size, tokens=x.shape[0],
                             zero_frac=float((y == 0).mean()), rel_mse=err / nrm if nrm > 0 else None, replica_equal=eq,
                             multiples_of_step=mult, decoded_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                             encode_seconds=t1 - t0, decode_seconds=t2 - t1, replica_seconds=t3 - t2))
    means_file = None
    if job['means_dir'] is not None and mvec:
        means_file = str(Path(job['means_dir']) / (uid.replace('/', '_') + '.npz'))
        np.savez(means_file, **mvec)
    return dict(key=uid, unit=uid, unit_id=rec['unit_id'], rows=rows, stats=stats, means_file=means_file,
                qps=qps, started=t_unit, seconds=time.time() - t_unit, pid=os.getpid(), host=socket.gethostname())


def extract_finished(features):
    metas = list(Path(features).glob('run-extract-*.json'))
    if not metas:
        return False
    return all(json.loads(p.read_text()).get('status') not in ('loading', 'running') for p in metas)


def code_main(a):
    from multiprocessing import get_context
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    shard = parse_shard(a.shard)
    # Per-unit token-mean vectors are always kept (small; summary mean_vector_norm); means.npz is only
    # written by finalize without --skip-means, as before.
    means_dir = out / 'unit_means'
    means_dir.mkdir(exist_ok=True)
    recs = read_features(a.features, tolerate_tail=a.follow)
    configs = []
    for c in a.configs or []:
        configs.append(c)
    if a.configs_file:
        text = Path(a.configs_file).read_text()
        configs += json.loads(text) if text.lstrip().startswith('[') else \
            [l.split('#')[0].strip() for l in text.splitlines() if l.split('#')[0].strip()]
    offsets = json.loads(Path(a.offsets).read_text()) if a.offsets else None
    if configs and offsets is None and not recs:
        raise ValueError('--configs without --offsets needs at least one extracted unit to know the keys')
    main_key = a.main_key or (offsets or {}).get('main_key') or (default_main_key(recs) if recs else 'main')
    if configs:
        if offsets is not None:
            keys = [main_key] + list(levels.offset_map(offsets))
        else:
            ks = {k for r in recs for k in r['streams']}
            keys = [main_key] + levels.later_keys(ks | {main_key}, main_key)
    else:
        keys = None
    qps = needed_qps(a.qps or [], configs, keys, main_key, offsets)
    per = stream_qps(a.qps or [], configs, keys, main_key, offsets) if configs and not a.union_qps else None
    extend = not a.no_extend and not a.follow
    if extend:
        absmax = 0.0
        for r in recs:
            for k in r['streams']:
                if k not in r['inactive']:
                    absmax = max(absmax, float(np.abs(np.load(unit_dir(a.features, fkey(r)) / f'{k}.npy', mmap_mode='r')).max()))
        ext = extension_qps(qps, absmax)
        if per is not None:
            per = {k: sorted(set(v) | (set(ext) - set(qps))) for k, v in per.items()}
        qps = ext
    if not qps:
        raise ValueError('No QPs: give --qps and/or --configs')
    work = per if per is not None else qps
    out_path = out / (f'rate_units.shard-{shard[0]}-of-{shard[1]}.jsonl' if shard else 'rate_units.jsonl')
    plan = dict(event='plan', features=str(a.features), qps=qps, qps_by_stream=per, configs=len(configs), main_key=main_key,
                workers=a.workers, shard=shard, follow=a.follow, delete_features=a.delete_features)
    print(json.dumps(plan), flush=True)
    (out / f'code-plan-{time.strftime("%Y%m%dT%H%M%S")}.json').write_text(json.dumps({**plan, 'configs_list': configs}, indent=1))
    keep = {fkey(r) for r in recs[:a.keep_features]}
    ctx = get_context('spawn')
    t0 = time.time(); last = time.time(); n_done = 0; finishing = False
    pool = ctx.Pool(a.workers, initializer=_init_worker) if a.workers > 1 else None
    heal_tail(out_path)
    try:
        with open(out_path, 'a', buffering=1) as f:
            while True:
                recs = read_features(a.features, tolerate_tail=a.follow)
                keep |= {fkey(r) for r in recs[:a.keep_features]}
                have = read_rate_lines([out])
                jobs = []
                for r in recs:
                    if not unit_shard(r['unit_id'], shard) or (a.max_units is not None and len(jobs) + n_done >= a.max_units):
                        continue      # shards by document, so all pages of a document land on one machine
                    done = coded_pairs(have.get(fkey(r), []))
                    active = [k for k in r['streams'] if k not in r['inactive']]
                    if all((k, q) in done for k in active for q in qps_for(work, k)):
                        continue
                    if not unit_dir(a.features, fkey(r)).exists():
                        print(json.dumps(dict(warning='features missing (deleted?)', unit=fkey(r))), flush=True)
                        continue
                    jobs.append(dict(features=str(a.features), rec=r, qps=work, done=sorted(done, key=str),
                                     means_dir=None if (means_dir is None or have.get(fkey(r))) else str(means_dir)))
                if jobs:
                    it = pool.imap_unordered(code_unit, jobs) if pool else map(code_unit, jobs)
                    for line in it:
                        f.write(json.dumps(line) + '\n')
                        n_done += 1; last = time.time()
                        ok = all(r.get('replica_equal', True) for r in line['rows'])
                        d = unit_dir(a.features, line['unit'])
                        if a.delete_features and ok and line['unit'] not in keep and d.is_symlink():
                            print(json.dumps(dict(warning='unit dir is a symlink; not deleted', unit=line['unit'])), flush=True)
                        elif a.delete_features and ok and line['unit'] not in keep:
                            for p in d.glob('*.npy'):
                                p.unlink()
                            d.rmdir()
                            with open(out / 'deleted_features.txt', 'a') as g:
                                g.write(line['unit'] + '\n')
                        print(json.dumps(dict(unit=line['unit'], done=n_done, unit_seconds=round(line['seconds'], 2),
                                              replica_ok=ok, seconds=round(time.time() - t0, 1))), flush=True)
                    continue
                if not a.follow:
                    break
                if extract_finished(a.features):
                    # one more pass picks up units written just before the extract finished; a second pass
                    # without jobs ends the run (units left uncoded, e.g. --max-units or missing feature dirs,
                    # must not spin this loop forever)
                    if finishing:
                        break
                    finishing = True
                    continue
                if time.time() - last > a.idle_timeout:
                    print(json.dumps(dict(warning='no new features for idle timeout; stopping', seconds=a.idle_timeout)), flush=True)
                    break
                time.sleep(a.poll)
    finally:
        if pool:
            pool.close(); pool.join()
    print(json.dumps(dict(event='code_done', units_coded=n_done, seconds=round(time.time() - t0, 1))), flush=True)
    return n_done


# ---- finalize ------------------------------------------------------------------------------------

def finalize_main(a):
    from cloud_prep.measurement_v19 import codec
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    lines = read_rate_lines([out] + list(a.inputs or []))
    order = [fkey(r) for r in read_features(a.features)] if a.features else []
    units = [u for u in order if u in lines] + sorted(u for u in lines if u not in set(order))
    merged = {u: merge_unit(lines[u]) for u in units}
    rows = [r for u in units for r in ordered_rows(merged[u][0], merged[u][1])]
    with open(out / 'rate_per_unit.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    per_unit = {}
    for r in rows:
        per_unit.setdefault(r['unit'], []).append(dict(stream=r['stream'], qp=r['qp'], bytes=r['bytes']))
    # QPs complete for every unit (every active stream) vs partial.
    coded = {u: {(s, q) for (s, q) in merged[u][0] if q != 'raw'} for u in units}
    allq = sorted({q for u in units for (_, q) in coded[u]})
    full = [q for q in allq if all((s, q) in coded[u] for u in units for s in merged[u][1])]
    streams = []
    for u in units:
        streams += [k for k in merged[u][1] if k not in streams]
    has = {k: [u for u in units if k in merged[u][1]] for k in streams}
    by_stream = {k: [q for q in allq if all((k, q) in coded[u] for u in has[k])] for k in streams}
    (out / 'bytes.json').write_text(json.dumps(dict(nncodec=codec.nncodec_version(), settings=codec.SETTINGS, qps=full,
                                                     qps_by_stream=by_stream,
                                                     qps_partial=[q for q in allq if any(q not in v for v in by_stream.values())],
                                                     per_unit=per_unit)))
    st = {u: merged[u][2] for u in units}
    # Level M: per stream, image-weighted mean of per-image token means (calibration images only).
    means = {}
    um = {u: np.load(merged[u][3]) for u in units if merged[u][3] and Path(merged[u][3]).exists()}
    if len(um) == len(units):
        means = {k: np.mean([um[u][k] for u in has[k]], axis=0).astype(np.float32) for k in streams}
        if not a.skip_means:
            np.savez(out / 'means.npz', **means)
    else:
        print(json.dumps(dict(warning='per-unit means missing for some units; no means / mean_vector_norm',
                              have=len(um), units=len(units))), flush=True)
    shapes = sorted({(st[u][k]['tokens'], st[u][k]['hidden']) for u in units for k in merged[u][1]})
    header = {f'{s[0]}x{s[1]}': len(codec.encode(np.zeros(s, np.float32), 0)) for s in shapes[:1] + shapes[-1:]}
    stats = {}
    for k in streams:
        n = sum(st[u][k]['n'] for u in has[k]); s = sum(st[u][k]['sum'] for u in has[k]); ss = sum(st[u][k]['sumsq'] for u in has[k])
        m = s / n
        toks = [st[u][k]['tokens'] for u in has[k]]
        stats[display(k)] = dict(hidden=int(st[has[k][0]][k]['hidden']), active_images=len(has[k]),
                                 tokens_mean=float(np.mean(toks)), tokens_min=int(min(toks)), tokens_max=int(max(toks)),
                                 mean=m, std=float(np.sqrt(max(ss / n - m * m, 0.0))),
                                 absmax=max(st[u][k]['absmax'] for u in has[k]),
                                 mean_vector_norm=float(np.linalg.norm(means[k])) if k in means else None)
    n_units = len(units)
    qps = allq
    table = []
    bt = {(r['unit'], r['stream'], r['qp']): r for r in rows}
    raw = {k: np.mean([bt[(u, k, 'raw')]['bytes'] for u in has[k]]) for k in streams}
    rawtot = np.mean([sum(bt[(u, k, 'raw')]['bytes'] for k in merged[u][1]) for u in units])
    for qp in allq:
        tot = np.mean([sum(bt[(u, k, qp)]['bytes'] for k in merged[u][1]) for u in units]) if qp in full else None
        for k in streams:
            if qp not in by_stream[k]:
                continue
            rr = [bt[(u, k, qp)] for u in has[k]]
            b = float(np.mean([r['bytes'] for r in rr]))
            per_image = sum(r['bytes'] for r in rr) / n_units
            table.append(dict(stream=display(k), qp=qp, step=levels.step_size(qp), images=len(rr),
                              mean_bytes=round(b, 1), raw_bf16_bytes=round(raw[k], 1),
                              reduction_vs_bf16_pct=round(100 * (1 - b / raw[k]), 3),
                              zero_frac=round(float(np.mean([r['zero_frac'] for r in rr])), 6),
                              rel_mse=float(np.mean([r['rel_mse'] for r in rr if r['rel_mse'] is not None])),
                              share_of_total_bytes=round(per_image / tot, 4) if tot else None))
        if qp not in full:
            continue
        allr = [r for r in rows if r['qp'] == qp]
        table.append(dict(stream='all', qp=qp, step=levels.step_size(qp), images=n_units, mean_bytes=round(float(tot), 1),
                          raw_bf16_bytes=round(float(rawtot), 1), reduction_vs_bf16_pct=round(100 * (1 - tot / rawtot), 3),
                          zero_frac=round(float(np.mean([r['zero_frac'] for r in allr])), 6),
                          rel_mse=float(np.mean([r['rel_mse'] for r in allr if r['rel_mse'] is not None])),
                          share_of_total_bytes=1.0, preprint_reduction_pct=PREPRINT.get(qp) if a.preprint else None))
    if a.csv:
        Path(a.csv).parent.mkdir(parents=True, exist_ok=True)
        keys = ['stream', 'qp', 'step', 'images', 'mean_bytes', 'raw_bf16_bytes', 'reduction_vs_bf16_pct', 'zero_frac',
                'rel_mse', 'share_of_total_bytes', 'preprint_reduction_pct']
        with open(a.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in table:
                w.writerow({k: r.get(k) for k in keys})
    coded_rows = [r for r in rows if r['qp'] != 'raw']
    first = units[0] if units else None
    hashes = {f"{r['stream']}|{r['qp']}": r.get('decoded_sha256') for r in coded_rows if r['unit'] == first}
    all_lines = [l for u in units for l in lines[u]]
    summary = dict(nncodec=codec.nncodec_version(), settings=codec.SETTINGS, qps=qps, qps_complete=full, units=n_units,
                   replica_equal_all=all(r['replica_equal'] for r in coded_rows),
                   multiples_of_step_all=all(r['multiples_of_step'] for r in coded_rows),
                   header_bytes_all_zero=header, stats=stats,
                   active_streams_per_image={str(n): sum(1 for u in units if len(merged[u][1]) == n)
                                             for n in sorted({len(merged[u][1]) for u in units})},
                   first_unit_decoded_sha256=hashes, first_unit=first,
                   seconds=sum(l['seconds'] for l in all_lines),
                   wall_seconds_span=(max(l['started'] + l['seconds'] for l in all_lines) - min(l['started'] for l in all_lines))
                   if all_lines and all('started' in l for l in all_lines) else None,
                   hosts=sorted({l.get('host') for l in all_lines if l.get('host')}),
                   timing=timing_summary(coded_rows))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ('first_unit_decoded_sha256', 'timing')}, indent=1))
    return summary


def timing_summary(rows):
    """Per-QP encode+decode+replica seconds per million elements (rows with timing only)."""
    t = [r for r in rows if 'encode_seconds' in r]
    if not t:
        return None
    by = {}
    for r in t:
        e = by.setdefault(r['qp'], [0, 0.0, 0.0, 0])
        e[0] += r['numel']; e[1] += r['encode_seconds']; e[2] += r['encode_seconds'] + r['decode_seconds'] + r['replica_seconds']; e[3] += 1
    return dict(per_qp={str(q): dict(tensors=v[3], melem=round(v[0] / 1e6, 3), encode_s_per_melem=round(v[1] / v[0] * 1e6, 4),
                                     total_s_per_melem=round(v[2] / v[0] * 1e6, 4)) for q, v in sorted(by.items())},
                encode_seconds=sum(v[1] for v in by.values()), total_seconds=sum(v[2] for v in by.values()),
                melem=sum(v[0] for v in by.values()) / 1e6)


# ---- device check (model env) ----------------------------------------------------------------------

def device_check_main(a):
    import torch
    recs = {fkey(r): r for r in read_features(a.features)}
    lines = read_rate_lines(a.rate) if a.rate else {}
    units = [u for u in recs if unit_dir(a.features, u).exists()][:a.units]
    dev = torch.device(a.device)
    res, n_bad = [], 0
    for u in units:
        r = recs[u]
        rec_rows = merge_unit(lines[u])[0] if u in lines else {}
        for k in r['streams']:
            if k in r['inactive']:
                continue
            x = np.load(unit_dir(a.features, u) / f'{k}.npy')
            x = torch.from_numpy(x.reshape(-1, x.shape[-1]))
            qps = a.qps or sorted(q for (s, q) in rec_rows if s == k and q != 'raw')
            for qp in qps:
                c = levels.quantize(x, qp).contiguous()
                g = levels.quantize(x.to(dev), qp).to('cpu').contiguous()
                eq_dev = bool(torch.equal(c.view(torch.int32), g.view(torch.int32)))
                sha = hashlib.sha256(c.numpy().tobytes()).hexdigest()
                ref = rec_rows.get((k, qp), {}).get('decoded_sha256')
                eq_nnc = None if ref is None else sha == ref
                bad = (not eq_dev) or eq_nnc is False
                n_bad += bad
                res.append(dict(unit=u, stream=k, qp=qp, cpu_eq_device=eq_dev, cpu_eq_nncodec=eq_nnc,
                                mismatches=int((c.view(torch.int32) != g.view(torch.int32)).sum())))
    summary = dict(device=str(dev), torch=torch.__version__,
                   gpu=torch.cuda.get_device_name(0) if dev.type == 'cuda' else None,
                   units=len(units), checks=len(res), all_cpu_eq_device=all(r['cpu_eq_device'] for r in res),
                   nncodec_compared=sum(r['cpu_eq_nncodec'] is not None for r in res),
                   all_cpu_eq_nncodec=all(r['cpu_eq_nncodec'] is not False for r in res), failures=n_bad)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(dict(summary=summary, checks=res), indent=1))
    print(json.dumps(summary, indent=1))
    return 1 if n_bad else 0


# ---- estimate ----------------------------------------------------------------------------------------

def estimate_main(a):
    """hours = units * melem_per_unit * sum_q s_per_melem(q) / cores (per-QP when timed, else overall)."""
    per_qp, overall = {}, []
    for d in a.rate:
        s = json.loads((Path(d) / 'summary.json').read_text())
        if s.get('timing'):
            for q, v in s['timing']['per_qp'].items():
                per_qp.setdefault(int(q), []).append(v['total_s_per_melem'])
            overall.append(s['timing']['total_seconds'] / s['timing']['melem'])
        else:
            n = 0
            with open(Path(d) / 'rate_per_unit.jsonl') as f:
                for line in f:
                    r = json.loads(line)
                    if r['qp'] != 'raw':
                        n += r['numel']
            overall.append(s['seconds'] / (n / 1e6))
    flat = float(np.mean(overall))
    rates = [float(np.mean(per_qp[q])) if q in per_qp else flat for q in a.qps]
    cpu_s = a.units * a.melem_per_unit * sum(rates)
    out = dict(units=a.units, melem_per_unit=a.melem_per_unit, qps=a.qps, s_per_melem_used=dict(zip(map(str, a.qps), [round(r, 4) for r in rates])),
               overall_s_per_melem=round(flat, 4), cpu_hours=round(cpu_s / 3600, 2),
               wall_hours={str(c): round(cpu_s / 3600 / c, 2) for c in a.cores},
               note='measured on the machine that produced --rate; scale by the cloud CPU per-core speed')
    print(json.dumps(out, indent=1))
    return out


# ---- CLI -----------------------------------------------------------------------------------------------

def add_code_args(p):
    p.add_argument('--features', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--qps', type=int, nargs='*', default=None)
    p.add_argument('--configs', nargs='*')
    p.add_argument('--configs-file', type=Path)
    p.add_argument('--offsets', type=Path, help='offsets JSON for offset configs q<q>|k<k>')
    p.add_argument('--main-key', help='default: offsets main_key, else layer0 if present, else main')
    p.add_argument('--no-extend', action='store_true', help='code only the given QPs (no extension to all-zero)')
    p.add_argument('--union-qps', action='store_true', help='with --configs: code the union of QPs on every stream')
    p.add_argument('--skip-means', action='store_true', help='test images: never fit level-M means on them')
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--shard', help='i/n by sha256(unit id)')
    p.add_argument('--max-units', type=int)
    p.add_argument('--follow', action='store_true', help='keep polling the extract dir until its runs finish (implies --no-extend)')
    p.add_argument('--poll', type=float, default=30)
    p.add_argument('--idle-timeout', type=float, default=3600)
    p.add_argument('--delete-features', action='store_true', help='delete a unit\'s .npy after coding + replica check')
    p.add_argument('--keep-features', type=int, default=8, help='never delete the first N units (device-check)')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] not in SUBCOMMANDS and argv[0] not in ('-h', '--help'):
        # legacy: code (one process) + finalize
        p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        p.add_argument('--csv', type=Path, required=True)
        p.add_argument('--preprint', action='store_true', help='add the Qwen3-VL-8B preprint reference column')
        add_code_args(p)
        a = p.parse_args(argv)
        if a.qps is None and not a.configs and not a.configs_file:
            a.qps = BASE_QPS
        code_main(a)
        a.inputs = []
        finalize_main(a)
        return 0
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    add_code_args(sub.add_parser('code', help='code units (resumable, parallel)'))
    f = sub.add_parser('finalize', help='merge rate_units into bytes.json / summary / csv / means')
    f.add_argument('--out', type=Path, required=True)
    f.add_argument('--inputs', type=Path, nargs='*', help='more code output dirs to merge (other machines)')
    f.add_argument('--features', type=Path, help='extract dir: unit order as in features.jsonl')
    f.add_argument('--csv', type=Path)
    f.add_argument('--preprint', action='store_true')
    f.add_argument('--skip-means', action='store_true')
    d = sub.add_parser('device-check', help='CPU vs device replica vs NNCodec decode sha (model env)')
    d.add_argument('--features', type=Path, required=True)
    d.add_argument('--rate', type=Path, nargs='*', help='code output dirs with decoded_sha256')
    d.add_argument('--device', default='cuda')
    d.add_argument('--units', type=int, default=4)
    d.add_argument('--qps', type=int, nargs='*', help='default: every QP coded for the unit')
    d.add_argument('--out', type=Path)
    e = sub.add_parser('estimate', help='CPU hours for a planned audit')
    e.add_argument('--rate', type=Path, nargs='+', required=True)
    e.add_argument('--units', type=int, required=True)
    e.add_argument('--melem-per-unit', type=float, required=True, help='million float32 elements per unit, all active streams')
    e.add_argument('--qps', type=int, nargs='+', required=True)
    e.add_argument('--cores', type=int, nargs='+', default=[8, 16, 32, 64])
    a = p.parse_args(argv)
    if a.cmd == 'code':
        code_main(a)
    elif a.cmd == 'finalize':
        finalize_main(a)
    elif a.cmd == 'device-check':
        return device_check_main(a)
    else:
        estimate_main(a)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
