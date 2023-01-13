https://stackoverflow.com/questions/63403203/terminology-used-in-nsight-compute
sm__throughput is max of a bunch of metrics


get_bench_times.sh <output directory>
    [uses get_bench_times.py for each application]
    - get's benchmark run times and creates a app/bench-times.csv with batch-name, execution-time
	- run from: /p/vertical/afs-huge3/casio-results/ [a100/, v100/, p100/]
	- provide as directory for output: /p/vertical/afs-huge3/casio-results/summaries/ [a100/, v100/, p100/]
	  (script creates bench mark subdirectories)
	  (bash script uses get_bench_times.py python script
	- env variable: CASIO_ROOT=/p/vertical/afs-huge3/casio
	  (location of casio git repo, needed for $CASIO_ROOT/scripts/get_bench_times.py]
	  

get_gemm_count_all.sh <output directory>
    [uses get_gemm.sh for each application]
    - uses egrep on gputrace.csv.gz (from nsys) to determine time spent in gemm and non-gemm kernels
	  for our purposes any kernel with reg-ex matches this is considered a gemm kernel
	  egrep -i "gemm|wgrad|nchw|nhwc|conv2d|dgrad2d|convolve"
	- run from: /p/vertical/afs-huge3/casio-results/ summaries
	- uses these semi-manually generated files:
	  [a/v,p]100-large-batch-list
	  [a/v,p]100-small-batch-list
	- writes results to all-hw/large-batch-times.csv, all-hw/small-batch-times.csv
	  (bash script uses get_gemm.sh bash script
	- env variable: CASIO_ROOT=/p/vertical/afs-huge3/casio
	  (location of casio git repo, needed for $CASIO_ROOT/scripts/get_gemm.sh]

get_nsys_reports.sh
   [uses process_nsys.sh for each application]
	- run from: /p/vertical/afs-huge3/casio-results/ [a100/, v100/, p100/]
	- provide as directory for output: /p/vertical/afs-huge3/casio-results/summaries/ [a100/, v100/, p100/]
	  (script creates bench mark subdirectories)
	- creates batch-<batch_size>_gputrace.csv.gz, _cudaapisum.csv, _gpukernsum.csv
	- env variable: CASIO_ROOT=/p/vertical/afs-huge3/casio
	  (location of casio git repo, needed for $CASIO_ROOT/scripts/get_gemm.sh]


get_ncu_reports.sh
   [uses histo_from_ncu.py for each application]
	- run from: /p/vertical/afs-huge3/casio-results/ [a100/, v100/, p100/]
	- provide as directory for output: /p/vertical/afs-huge3/casio-results/summaries/ [a100/, v100/, p100/]
	  (script creates bench mark subdirectories)
	- creates batch-<batch_size>_feature-avg.csv
	- creates batch-<batch_size>_flame.<hardware utilization>.csv
	  ['cumm-fraction-time', 'metric', 'fraction-time', 'Kernel Name', 'launch__block_dim_x', 'launch__block_dim_y', 'launch__block_dim_z', 'launch__block_size', 'launch__grid_dim_x', 'launch__grid_dim_y', 'launch__grid_dim_z', 'launch__grid_size', 'launch__occupancy_limit_blocks', 'launch__occupancy_limit_registers', 'launch__occupancy_limit_shared_mem', 'launch__occupancy_limit_warps', 'launch__occupancy_per_block_size', 'launch__occupancy_per_register_count', 'launch__occupancy_per_shared_mem_size', 'launch__registers_per_thread', 'launch__registers_per_thread_allocated', 'launch__shared_mem_config_size', 'launch__shared_mem_per_block', 'launch__shared_mem_per_block_allocated', 'launch__shared_mem_per_block_driver', 'launch__shared_mem_per_block_dynamic', 'launch__shared_mem_per_block_static', 'launch__thread_count', 'launch__waves_per_multiprocessor']
	- env variable: CASIO_ROOT=/p/vertical/afs-huge3/casio
	  (location of casio git repo, needed for $CASIO_ROOT/scripts/get_gemm.sh]
    - Details of histo_from_ncu.py
	  Originally written to get diff between two ncu reports.
	  We now use to by providing the same file name for 1st and 2nd are; 3rd arg is output directory.


get_sass_reports.sh
   [uses process_sass.py for each application]
	- run from: /p/vertical/afs-huge3/casio-results/ [a100/, v100/, p100/]
	- provide as directory for output: /p/vertical/afs-huge3/casio-results/summaries/ [a100/, v100/, p100/]
	  (script creates bench mark subdirectories)
	- creates app/sass/batch-<batchsize>_
			_opcode-dist.csv (opcode-name, fraction-time, cummulative-fraction-time]
			_kernelname-dist.csv
			_inst-total.csv
			_inst-dist.csv  (PC, i-count-as-cummulative-fraction, fraction-of-time, cummulative-fraction-of-time, #-dynamic-instructions)
	- env variable: CASIO_ROOT=/p/vertical/afs-huge3/casio
	  (location of casio git repo, needed for $CASIO_ROOT/scripts/get_gemm.sh]
    - Details of process_sass.py to be written


get_kernel_hw_similarity_all.sh
   [uses kernel_hw_similarity.py for each application]
	- run from: /p/vertical/afs-huge3/casio-results/summaries/[a100/, v100/, p100/]
	- provide as directory for output: /p/vertical/afs-huge3/casio-results/summaries/ [a100/, v100/, p100/]
	  (script creates bench mark subdirectories)
	- creates app/batch-<batch_size>_ncu_kernel_similarity.txt
	- env variable: CASIO_ROOT=/p/vertical/afs-huge3/casio
	  (location of casio git repo, needed for $CASIO_ROOT/scripts/get_gemm.sh]
    - heatmaps for each application can be created by uncommenting lines 54,55,56 in kernel_similarity.py
	

kernel_sass_similarity.py for each application]
	- Takes a sass directory as input (from running process_sass previously)
	- Looks for all -sass.csv files and does pairwise similarity and prints representative kernels
	- 1st arg: sass directory with prefix of filename
	- 2nd arg is number o files
[needs more work to write wrapper script for top level]

get_kernel_sass_similarity_all.sh is BROKEN	





