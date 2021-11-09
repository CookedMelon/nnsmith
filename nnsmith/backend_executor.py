from nnsmith.backends import DiffTestBackend
from nnsmith.difftest import run_backend

# for testing
class CrashExecutor(DiffTestBackend):
    def predict(self, model, inputs):
        assert False

class HangExecutor(DiffTestBackend):
    def predict(self, model, inputs):
        while True:
            pass

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./tmp')
    parser.add_argument('--backend', type=str, required=True,
        help='One of ort, trt, tvm, and xla')
    parser.add_argument('--timeout', type=int, default=5*60, 
        help='timeout in seconds')
    # TODO: Add support for passing backend-specific options
    args = parser.parse_args()

    def get_backend(name):
        if name == 'ort':
            from nnsmith.backends.ort_graph import ORTExecutor
            return ORTExecutor()
        elif name == 'tvm':
            from nnsmith.backends.tvm_graph import TVMExecutor
            return TVMExecutor(target='cuda')
        elif name == 'xla':
            from nnsmith.backends.xla_graph import XLAExecutor
            return XLAExecutor(device='CUDA')
        elif name == 'crash':
            return CrashExecutor()
        elif name == 'hang':
            return HangExecutor()
        else:
            raise ValueError(f'unknown backend: {name}')

    bknd = get_backend(args.backend)
    run_backend(args.root, bknd, args.timeout)