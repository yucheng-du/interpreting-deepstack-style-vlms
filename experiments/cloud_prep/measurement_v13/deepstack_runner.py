"""Original DeepStack-L on the unchanged v12 common P1/P2 plans (CLI).

Planning reuses measurement_v12.p1_runner / p2_runner (conditions, queries,
readouts, shards, accounting) with DeepStack's own CPU geometry preflight and a
process-local deepstack7b registry entry; execution uses deepstack_executor,
never the Qwen/Granite factories. --dry-run imports no tensor library.
"""
import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import json
from pathlib import Path
import sys

from cloud_prep.measurement_v12 import p1_runner, p2_runner
from cloud_prep.measurement_v13.runtime_provenance import repairable_runtime, scientific_identity
from cloud_prep.measurement_v13 import deepstack_core as core

ARMS = ('p1', 'local', 'full', 'dose')
DEFAULT_VERSION = 'v13-deepstack-core-v1'
OWN_SOURCES = ('deepstack_runner.py', 'deepstack_executor.py', 'deepstack_core.py', 'deepstack_adapter.py',
               'deepstack_geometry.py', 'model_extensions.py', 'runtime_provenance.py')


def split_args(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--arm', required=True, choices=ARMS)
    parser.add_argument('--execution-version', default=DEFAULT_VERSION)
    known, rest = parser.parse_known_args(argv)
    module = p1_runner if known.arm == 'p1' else p2_runner
    if known.arm != 'p1':
        rest = ['--arm', known.arm, *rest]
    return known, rest, module


@contextmanager
def _replace(module, name, value):
    saved = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, saved)


@contextmanager
def planning_context():
    """Scoped, single-process: registry entry, provenance-only vendor record, native geometry."""
    with ExitStack() as stack:
        stack.enter_context(core.deepstack_registry())
        stack.enter_context(repairable_runtime())
        for module in (p1_runner, p2_runner):
            stack.enter_context(_replace(module, 'preflight_geometry', core.preflight_geometry))
        yield


def parse_args(argv=None):
    known, rest, module = split_args(argv)
    with planning_context():
        args = module.parse_args(rest)
    if args.model != core.MODEL_KEY:
        raise SystemExit('deepstack_runner executes only deepstack7b')
    args.deepstack_arm = known.arm
    args.execution_version = known.execution_version
    return args


def deepstack_contract():
    return {'schema': core.CORE_SCHEMA, 'model_key': core.MODEL_KEY,
        'geometry': 'main = 24x24 projected CLIP grid of the 336-squash global view at the 576 expanded <image> '
                    'positions (LM input); later0..3 = native 2D phase groups of the high-resolution crop grid added '
                    'before model.model.layers[1..4]; P1 384x384 -> 2x2 (four active); P2 384x256 -> 2x1 (two active, '
                    'two official zero-filled inactive_zero_padded_group, contribute zero, never missing)',
        'roi_rule': core.ROI_RULE, 'later_role': 'all active native phase groups; padded groups structurally inactive',
        'score_continuation_separator': core.SCORE_SEPARATOR, 'score_separator_rule': core.SCORE_SEPARATOR_RULE,
        'generation': 'official DeepstackLlamaForCausalLM.generate (inputs_embeds); returns only new tokens; greedy, '
                      'EOS=2, max_new_tokens from plan; truncated = no EOS within budget',
        'context': 'expanded length (prompt-1+576) + max(candidate, generation) <= min(max_sequence, tokenizer_model_max_length=2048)',
        'dose': 'v11-v2 resolve_scale/native_dose unchanged; DeepStack per-entry ROI tokens and native boundaries; '
                'zero/no-support dose -> unmatched (no arbitrary k, no zero denominator)'}


def decorate_plan(plan, args):
    plan['execution_version'] = args.execution_version
    plan['scientific_identity'] = scientific_identity(plan)
    here = Path(__file__).parent
    plan['v13_source_sha256'] = {name: hashlib.sha256((here / name).read_bytes()).hexdigest() for name in OWN_SOURCES}
    plan['executor'] = 'cloud_prep.measurement_v13.deepstack_executor'
    plan['deepstack_contract'] = deepstack_contract()
    plan['implementation_policy'] = ('Repairable runtime; record edits and rerun only affected scientific units when '
                                     'needed. Completed compatible units remain valid.')
    return plan


def prepare(args):
    module = p1_runner if args.deepstack_arm == 'p1' else p2_runner
    with planning_context():
        rows, plan = module.prepare(args)
    return rows, decorate_plan(plan, args)


def execute(args, rows, plan):
    from cloud_prep.measurement_v13 import deepstack_executor as executor
    from cloud_prep.measurement_v12 import p1_design, p2_design, v11_v2_dose_design
    arm = args.deepstack_arm
    if arm == 'p1':
        design, dose_design = p1_design, p1_design
        record_fields = lambda row, variant, condition, prediction: p1_design.record_fields(row, variant, condition, prediction)
    else:
        design, dose_design = p2_design.ExecutionDesign, v11_v2_dose_design
        record_fields = lambda row, variant, condition, prediction: p2_design.record_fields(row, variant, condition, prediction, arm)
    return executor.execute(args, rows, plan, design, mode=arm, record_fields=record_fields, dose_design=dose_design)


def main(argv=None):
    args = parse_args(argv)
    rows, plan = prepare(args)
    if args.plan_output:
        from cloud_prep.measurement_v8.runtime import write_json
        from cloud_prep.portable_data import _guard_path
        write_json(_guard_path(args.plan_output, write=True), plan)
    if args.dry_run:
        if 'torch' in sys.modules:
            raise RuntimeError('CPU planning imported torch')
        keep = ('schema', 'arm', 'panel', 'items', 'scene_templates', 'independent_source_photos',
                'unique_visual_interventions', 'expected_score_rows', 'expected_generation_rows',
                'expected_unique_score_evaluations', 'expected_unique_generation_evaluations',
                'expected_anchor_forwards', 'execution_schema', 'execution_version', 'manifest_sha256',
                'independence_unit', 'physical_work_accounting')
        print(json.dumps({k: plan[k] for k in keep if k in plan}
                         | {'model': plan['model']['key'], 'torch_imported': False, 'dry_run': True}, indent=2))
        return 0
    from cloud_prep.measurement_v8.runtime import prepare_runtime
    from cloud_prep.measurement_v4.content_intervention import cloud_gpu_lock
    prepare_runtime(args)
    try:
        with cloud_gpu_lock(args.campaign_root, args.gpu):
            return execute(args, rows, plan)
    except BlockingIOError:
        print(json.dumps({'event': 'gpu_busy', 'gpu': args.gpu}), flush=True)
        return 75


if __name__ == '__main__':
    raise SystemExit(main(sys.argv[1:]))
