"""
ONNX to TensorRT Conversion Script

Converts ONNX model to TensorRT engine with INT8 optimization.

Usage:
    python convert_tensorrt.py --onnx export/qvit_int8.onnx --output export/qvit_int8.engine
    python convert_tensorrt.py --onnx export/qvit_int8.onnx --output export/qvit_int8.engine --fp16
"""

import argparse
import os
from pathlib import Path


def convert_onnx_to_tensorrt(
    onnx_path: str,
    output_path: str,
    int8: bool = True,
    fp16: bool = False,
    workspace_size: int = 4,  # GB
    batch_size: int = 1,
    verbose: bool = False
):
    """
    Convert ONNX model to TensorRT engine.

    Args:
        onnx_path: Path to ONNX model
        output_path: Path to save TensorRT engine
        int8: Enable INT8 mode (uses Q/DQ nodes from ONNX)
        fp16: Enable FP16 mode (can be combined with INT8)
        workspace_size: GPU workspace size in GB
        batch_size: Batch size for optimization
        verbose: Print detailed logs
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("Error: TensorRT is not installed.")
        print("Install with: pip install tensorrt")
        print("Or use NVIDIA's TensorRT package for your CUDA version.")
        return None

    # Logger
    logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.INFO)

    print(f"[TensorRT] Converting ONNX to TensorRT")
    print(f"  - Input: {onnx_path}")
    print(f"  - Output: {output_path}")
    print(f"  - INT8: {int8}, FP16: {fp16}")
    print(f"  - Workspace: {workspace_size} GB")

    # 1. Create builder and network
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)

    # 2. Parse ONNX
    parser = trt.OnnxParser(network, logger)

    print(f"[TensorRT] Parsing ONNX model...")
    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            print("Error: Failed to parse ONNX model")
            for i in range(parser.num_errors):
                print(f"  - {parser.get_error(i)}")
            return None

    print(f"  - Network inputs: {network.num_inputs}")
    print(f"  - Network outputs: {network.num_outputs}")
    print(f"  - Network layers: {network.num_layers}")

    # 3. Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size * (1 << 30))

    # Enable INT8 mode (TensorRT will use Q/DQ nodes from ONNX)
    if int8:
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print(f"  - INT8 mode enabled")
        else:
            print(f"  - Warning: Platform does not support fast INT8")

    # Enable FP16 mode
    if fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print(f"  - FP16 mode enabled")
        else:
            print(f"  - Warning: Platform does not support fast FP16")

    # 4. Set optimization profile for dynamic batch
    profile = builder.create_optimization_profile()

    # Get input shape from network
    input_tensor = network.get_input(0)
    input_shape = input_tensor.shape

    # Set min/opt/max shapes (assuming batch is dim 0)
    min_shape = (1,) + tuple(input_shape[1:])
    opt_shape = (batch_size,) + tuple(input_shape[1:])
    max_shape = (batch_size * 4,) + tuple(input_shape[1:])  # Allow up to 4x batch

    profile.set_shape(input_tensor.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    print(f"  - Input shape: min={min_shape}, opt={opt_shape}, max={max_shape}")

    # 5. Build engine
    print(f"[TensorRT] Building engine (this may take a while)...")
    serialized_engine = builder.build_serialized_network(network, config)

    if serialized_engine is None:
        print("Error: Failed to build TensorRT engine")
        return None

    # 6. Save engine
    print(f"[TensorRT] Saving engine...")
    with open(output_path, 'wb') as f:
        f.write(serialized_engine)

    # Print size comparison
    onnx_size = os.path.getsize(onnx_path)
    engine_size = os.path.getsize(output_path)

    # Check for external data file
    onnx_data_path = onnx_path + '.data'
    if os.path.exists(onnx_data_path):
        onnx_size += os.path.getsize(onnx_data_path)

    print(f"\n[TensorRT] Conversion complete!")
    print(f"  - ONNX size: {onnx_size / 1024 / 1024:.2f} MB")
    print(f"  - Engine size: {engine_size / 1024 / 1024:.2f} MB")
    print(f"  - Compression ratio: {onnx_size / engine_size:.2f}x")

    return output_path


def verify_engine(engine_path: str, input_shape: tuple = (1, 3, 224, 224)):
    """
    Verify TensorRT engine by running inference.

    Args:
        engine_path: Path to TensorRT engine
        input_shape: Input tensor shape
    """
    try:
        import tensorrt as trt
        import numpy as np
    except ImportError:
        print("Error: TensorRT or NumPy not installed")
        return False

    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("Error: PyCUDA not installed")
        print("Install with: pip install pycuda")
        return False

    print(f"\n[TensorRT] Verifying engine...")

    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    if engine is None:
        print("Error: Failed to load engine")
        return False

    # Create execution context
    context = engine.create_execution_context()

    # Allocate buffers
    input_data = np.random.randn(*input_shape).astype(np.float32)
    output_shape = (input_shape[0], 1000)  # Assuming 1000 classes
    output_data = np.empty(output_shape, dtype=np.float32)

    # Allocate device memory
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_data.nbytes)

    # Copy input to device
    cuda.memcpy_htod(d_input, input_data)

    # Run inference
    context.execute_v2([int(d_input), int(d_output)])

    # Copy output to host
    cuda.memcpy_dtoh(output_data, d_output)

    print(f"  - Input shape: {input_shape}")
    print(f"  - Output shape: {output_data.shape}")
    print(f"  - Output range: [{output_data.min():.4f}, {output_data.max():.4f}]")
    print(f"  - Verification: PASSED")

    return True


def benchmark_engine(engine_path: str, input_shape: tuple = (1, 3, 224, 224), num_iterations: int = 100):
    """
    Benchmark TensorRT engine inference speed.

    Args:
        engine_path: Path to TensorRT engine
        input_shape: Input tensor shape
        num_iterations: Number of iterations for benchmarking
    """
    try:
        import tensorrt as trt
        import numpy as np
        import time
    except ImportError:
        print("Error: Required packages not installed")
        return

    try:
        import pycuda.driver as cuda
        import pycuda.autoinit
    except ImportError:
        print("Error: PyCUDA not installed")
        return

    print(f"\n[TensorRT] Benchmarking engine...")
    print(f"  - Iterations: {num_iterations}")

    # Load engine
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)

    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())

    context = engine.create_execution_context()

    # Allocate buffers
    input_data = np.random.randn(*input_shape).astype(np.float32)
    output_shape = (input_shape[0], 1000)
    output_data = np.empty(output_shape, dtype=np.float32)

    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_data.nbytes)

    cuda.memcpy_htod(d_input, input_data)

    # Warmup
    print(f"  - Warming up...")
    for _ in range(10):
        context.execute_v2([int(d_input), int(d_output)])

    # Benchmark
    print(f"  - Running benchmark...")
    start = time.perf_counter()
    for _ in range(num_iterations):
        context.execute_v2([int(d_input), int(d_output)])
    cuda.Context.synchronize()
    end = time.perf_counter()

    total_time = end - start
    avg_time = total_time / num_iterations * 1000  # ms
    throughput = num_iterations / total_time

    print(f"\n[TensorRT] Benchmark Results:")
    print(f"  - Total time: {total_time:.2f}s")
    print(f"  - Average latency: {avg_time:.2f}ms")
    print(f"  - Throughput: {throughput:.2f} images/sec")


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to TensorRT')
    parser.add_argument('--onnx', type=str, default='export/qvit_int8.onnx',
                        help='Path to ONNX model')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to output TensorRT engine (default: same as onnx with .engine)')
    parser.add_argument('--int8', action='store_true', default=True,
                        help='Enable INT8 mode (default: True)')
    parser.add_argument('--fp16', action='store_true', default=False,
                        help='Enable FP16 mode')
    parser.add_argument('--workspace', type=int, default=4,
                        help='Workspace size in GB (default: 4)')
    parser.add_argument('--batch', type=int, default=1,
                        help='Optimization batch size (default: 1)')
    parser.add_argument('--verify', action='store_true', default=False,
                        help='Verify engine after conversion')
    parser.add_argument('--benchmark', action='store_true', default=False,
                        help='Benchmark engine after conversion')
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Verbose logging')

    args = parser.parse_args()

    # Set default output path
    if args.output is None:
        args.output = str(Path(args.onnx).with_suffix('.engine'))

    # Check ONNX file exists
    if not os.path.exists(args.onnx):
        print(f"Error: ONNX file not found: {args.onnx}")
        return

    # Convert
    result = convert_onnx_to_tensorrt(
        onnx_path=args.onnx,
        output_path=args.output,
        int8=args.int8,
        fp16=args.fp16,
        workspace_size=args.workspace,
        batch_size=args.batch,
        verbose=args.verbose
    )

    if result is None:
        return

    # Verify
    if args.verify:
        verify_engine(args.output)

    # Benchmark
    if args.benchmark:
        benchmark_engine(args.output)


if __name__ == '__main__':
    main()