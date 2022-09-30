import params

import cudaprofile
import time
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(int(params.devname.split(':')[1]))
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
tf.random.set_random_seed(0)

def benchmark_wrapper(appname, roi, sess):
    print(f'Warmup with {params.nw} Iters')
    for i in range(params.nw): roi()

    print(f'Running {params.ni} Iters')
    tt0 = time.perf_counter()
    if params.mode in {'ncu', 'nsys'}:
        cudaprofile.start()
        for i in range(params.ni): roi()
        cudaprofile.stop()

    elif params.mode == 'prof':
        profiler = tf.profiler.Profiler(graph=sess.graph)
        for i in range(params.ni):
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_meta = tf.RunMetadata()
            roi(run_options=run_options, run_metadata=run_meta)
            profiler.add_step(i, run_meta)
            opts = tf.profiler.ProfileOptionBuilder.time_and_memory()
            profiler.profile_operations(options=opts)

    else: # bench mode
        t0 = time.perf_counter()
        for i in range(params.ni): roi()
        t1 = time.perf_counter()
        print(f'Throughput: {params.ni * params.bs / (t1 - t0)}')

    tt1 = time.perf_counter()
    print(f'Total Time: {tt1 - tt0}')

