"""Run Qwen2 through the existing P1/P2 executors without editing old registries."""
import argparse
from contextlib import contextmanager
import sys
from cloud_prep.measurement_v13.model_extensions import candidate_registry


def dispatch(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--model', required=True, choices=('qwen2b',))
    parser.add_argument('--arm', required=True, choices=('p1', 'local', 'full', 'dose'))
    parser.add_argument('--execution-version', default='v13-qwen2-core-v1')
    args, rest = parser.parse_known_args(argv)
    forwarded = ['--model', args.model, *rest]
    if args.arm != 'p1':
        forwarded += ['--arm', args.arm]
    return args.arm, forwarded, args.execution_version


def parse_args(argv=None):
    arm, forwarded, version = dispatch(argv)
    with candidate_registry():
        if arm == 'p1':
            from cloud_prep.measurement_v12 import p1_runner as runner
        else:
            from cloud_prep.measurement_v12 import p2_runner as runner
        args = runner.parse_args(forwarded)
    args.candidate_arm = arm
    args.execution_version = version
    return args


def prepare(args):
    from cloud_prep.measurement_v13.runtime_provenance import repairable_runtime
    with candidate_registry(), repairable_runtime():
        if args.candidate_arm == 'p1':
            from cloud_prep.measurement_v12 import p1_runner as runner
        else:
            from cloud_prep.measurement_v12 import p2_runner as runner
        rows, plan = runner.prepare(args)
    plan['candidate_extension'] = dict(schema='candidate-qwen2-v13',
        CUDA_acceptance='pending_actual_smoke', reused_scientific_contract='v12_P1_P2',
        adapter='v10_QwenAdapter_and_v12_native_dose',
        candidate_arm=args.candidate_arm)
    from cloud_prep.measurement_v13.core_runner import decorate_plan
    decorate_plan(plan, args.execution_version)
    return rows, plan


@contextmanager
def _record_extension(runner, version):
    original = runner.prepare
    def wrapped(args):
        rows, plan = original(args)
        plan['candidate_extension'] = dict(schema='candidate-qwen2-v13',
            CUDA_acceptance='pending_actual_smoke', reused_scientific_contract='v12_P1_P2')
        from cloud_prep.measurement_v13.core_runner import decorate_plan
        decorate_plan(plan, version)
        return rows, plan
    runner.prepare = wrapped
    try:
        yield
    finally:
        runner.prepare = original


def main(argv=None):
    arm, forwarded, version = dispatch(argv)
    with candidate_registry():
        if arm == 'p1':
            from cloud_prep.measurement_v12 import p1_runner as runner
        else:
            from cloud_prep.measurement_v12 import p2_runner as runner
        from cloud_prep.measurement_v13.runtime_provenance import repairable_runtime
        with repairable_runtime(), _record_extension(runner, version):
            return runner.main(forwarded)


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
