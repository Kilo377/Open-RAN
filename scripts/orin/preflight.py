"""Exercise Orin CUDA kernels, independently of PyTorch's CUDA detection."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def check():
    import numpy as np
    import cupy as cp
    from phy.rt_sys.numerics import Algebra
    algebra=Algebra('cupy')
    matrix=algebra.array(np.eye(2)[None].repeat(4,axis=0))
    solved=algebra.solve(matrix,matrix)
    _,values,_=algebra.svd(matrix)
    cp.cuda.Stream.null.synchronize()
    np.testing.assert_allclose(cp.asnumpy(solved),np.eye(2)[None].repeat(4,axis=0),atol=1e-6)
    np.testing.assert_allclose(cp.asnumpy(values),1,atol=1e-6)
    return {'cupy':cp.__version__,'gpu_name':cp.cuda.runtime.getDeviceProperties(0)['name'].decode(),
        'compute_capability':cp.cuda.Device().compute_capability,
        'cuda_runtime_version':cp.cuda.runtime.runtimeGetVersion(),
        'actual_cuda_solve_and_svd':True,
        'execution_devices':{'rt':'cpu','channel_reconstruction':'cuda','linear_algebra':'cuda','phy_sys':'cpu'}}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true',help='Only print readiness; do not write the report')
    args=parser.parse_args()
    try:
        report=check()
    except Exception as exc:
        print(f'Orin CUDA preflight failed: {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(1)
    if args.quick:print(f'Orin CUDA ready: {report["gpu_name"]}; RT and Sionna NR remain on CPU.',file=sys.stderr)
    else:
        target=ROOT/'outputs/setup_orin_gpu/preflight.json'
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))
