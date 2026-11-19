"""Merge per-unit byte tables (bytes.json) from several rate-audit runs over the same features."""
import json
import sys

if __name__ == '__main__':
    out, *ins = sys.argv[1:]
    docs = [json.load(open(p)) for p in ins]
    per_unit = {}
    for d in docs:
        for u, recs in d['per_unit'].items():
            seen = {(r['stream'], r['qp']) for r in per_unit.get(u, [])}
            per_unit.setdefault(u, []).extend(r for r in recs if (r['stream'], r['qp']) not in seen)
    qps = sorted({q for d in docs for q in d['qps']})
    json.dump(dict(nncodec=docs[0]['nncodec'], settings=docs[0]['settings'], qps=qps, per_unit=per_unit,
                   merged_from=ins), open(out, 'w'))
    print(out, qps, len(per_unit))
