"""NNCodec wrapper: one tensor per bitstream, exact QP, plain URQ + DeepCABAC.

Runs only in the separate venv for NNCodec (nncodec 2.1.3);
the model envs never import nncodec. Settings that would change the effective QP or the tensor
are switched off: opt_qp, ioq, bnf, lsa, fine_tune, sparsification (not called), codebook
(codebook_mode=0), dependent/trellis quantization (use_dq=False), RDOQ (lambda_scale=0),
integer bit-depth mode (int_quant_bw=False), differential/temporal coding (tca, approx_param_base).
qp_per_tensor and nonweight_qp are both set to the requested QP so the tensor is coded at exactly
that QP whatever type NNCodec infers. row_skipping and param_opt are lossless entropy-coding
tools and stay on (NNCodec tensor-API defaults).
"""
from __future__ import annotations

import numpy as np

TENSOR_ID = 't'
SETTINGS = dict(qp_density=2, use_dq=False, codebook_mode=0, scan_order=0, lambda_scale=0,
                param_opt=True, cabac_unary_length_minus1=10, opt_qp=False, ioq=False, bnf=False,
                lsa=False, fine_tune=False, row_skipping=True, tca=False, pre_signalling=False,
                int_quant_bw=False, compress_differences=False)


def encode(x, qp):
    """float32 ndarray (2-D: tokens x hidden) -> complete NNC bitstream bytes."""
    from nncodec import nnc
    x = np.ascontiguousarray(np.asarray(x, dtype=np.float32))
    if x.ndim != 2:
        raise ValueError('Code each feature tensor as a 2-D weight tensor')
    qp = int(qp)
    bs = nnc.compress({TENSOR_ID: x}, bitstream_path=None, qp=qp, nonweight_qp=qp,
                      qp_per_tensor={TENSOR_ID: qp}, return_bitstream=True, verbose=False, **SETTINGS)
    return bytes(bs)


def decode(bs):
    from nncodec import nnc
    out = nnc.decompress(bytearray(bs), verbose=False)
    return np.asarray(out[TENSOR_ID], dtype=np.float32)


def nncodec_version():
    import importlib.metadata
    return importlib.metadata.version('nncodec')
