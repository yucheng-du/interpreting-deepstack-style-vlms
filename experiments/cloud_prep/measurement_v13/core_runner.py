"""Run unchanged v12 common-core designs with repairable v13 provenance."""
import argparse
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sys

from cloud_prep.measurement_v12 import p1_runner, p2_runner
from cloud_prep.measurement_v13.runtime_provenance import repairable_runtime, scientific_identity

ARMS = ('p1', 'local', 'full', 'dose')


def split_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--execution-version', default='v13-core-expansion-v1')
    known, rest = parser.parse_known_args(argv)
    module = p1_runner if known.arm == 'p1' else p2_runner
    if known.arm != 'p1':
        rest = ['--arm', known.arm, *rest]
    return known, rest, module


def parse_args(argv=None):
    known, rest, module = split_args(argv)
    args = module.parse_args(rest)
    args.v13_core_arm = known.arm
    args.execution_version = known.execution_version
    return args


def decorate_plan(plan, version):
    plan['execution_version'] = version
    plan['scientific_identity'] = scientific_identity(plan)
    plan['v13_source_sha256'] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('core_runner.py', 'runtime_provenance.py', 'candidate_runner.py', 'model_extensions.py')
    }
    plan['implementation_policy'] = 'Repairable runtime; record edits and rerun only affected scientific units when needed. Completed compatible units remain valid.'
    return plan


def prepare(args):
    module = p1_runner if args.v13_core_arm == 'p1' else p2_runner
    with repairable_runtime():
        rows, plan = module.prepare(args)
    return rows, decorate_plan(plan, args.execution_version)


@contextmanager
def _record_version(module, version):
    original = module.prepare
    def wrapped(args):
        rows, plan = original(args)
        return rows, decorate_plan(plan, version)
    module.prepare = wrapped
    try:
        yield
    finally:
        module.prepare = original


def main(argv=None):
    known, rest, module = split_args(argv)
    with repairable_runtime(), _record_version(module, known.execution_version):
        return module.main(rest)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
