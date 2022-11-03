
RESULTS=./casio-results
SUMS=./casio-results/summaries
PPROC=./casio-results/postproc

OPTRACE_SCRIPTS= \
	./scripts/get_optrace \
	./scripts/tfprof.py \
	./scripts/torchprof.py

dirs:
	@mkdir -p $(RESULTS)
	@mkdir -p $(SUMS)
	@mkdir -p $(PPROC)

ALL_APPS= \
	meshgraphnets-cfd \
	meshgraphnets-cloth \
	muzero \
	nerf \
	pinn-ac \
	pinn-kdv \
	pinn-navier-stokes \
	pinn-schrodinger \
	qdtrack \
	swin-swinv2_base_patch4_window12_192_22k \
	swin-swinv2_base_patch4_window16_256 \
	swin-swinv2_large_patch4_window12_192_22k \
	swin-swinv2_large_patch4_window12to24_192to384_22kto1k_ft \
	tabnet \
	tacotron2 \
	wavenet

TRACE_APPS= \
	meshgraphnets-cfd \
	meshgraphnets-cloth \
	muzero \
	pinn-ac \
	pinn-kdv \
	pinn-navier-stokes \
	pinn-schrodinger \
	qdtrack \
	swin-swinv2_base_patch4_window12_192_22k \
	swin-swinv2_base_patch4_window16_256 \
	swin-swinv2_large_patch4_window12_192_22k \
	swin-swinv2_large_patch4_window12to24_192to384_22kto1k_ft \
	tabnet \
	tacotron2 \
	wavenet

$(PPROC)/a100/%/bench-times.csv: $(RESULTS)/a100/% scripts/get_bench_times.sh | dirs
	@mkdir -p $(dir $@)
	python3	$(CASIO)/scripts/get_bench_times.py $</bench* > $@

$(PPROC)/v100/%/bench-times.csv: $(RESULTS)/v100/% scripts/get_bench_times.sh | dirs
	@mkdir -p $(dir $@)
	python3	$(CASIO)/scripts/get_bench_times.py $</bench* > $@

$(PPROC)/p100/%/bench-times.csv: $(RESULTS)/p100/% scripts/get_bench_times.sh | dirs
	@mkdir -p $(dir $@)
	python3	$(CASIO)/scripts/get_bench_times.py $</bench* > $@

all-bench-times: \
	$(foreach app,$(ALL_APPS),$(PPROC)/a100/$(app)/bench-times.csv) \
	$(foreach app,$(ALL_APPS),$(PPROC)/v100/$(app)/bench-times.csv) \
	$(foreach app,$(ALL_APPS),$(PPROC)/p100/$(app)/bench-times.csv)

$(PPROC)/a100/%/op-trace-large-batch.csv: $(RESULTS)/a100/% $(OPTRACE_SCRIPTS) | dirs
	@mkdir -p $(dir $@)
	./scripts/get_optrace a100 $(notdir $<) $(shell ./scripts/largebatch a100 $(notdir $<)) > $@

$(PPROC)/v100/%/op-trace-large-batch.csv: $(RESULTS)/v100/% $(OPTRACE_SCRIPTS) | dirs
	@mkdir -p $(dir $@)
	./scripts/get_optrace v100 $(notdir $<) $(shell ./scripts/largebatch v100 $(notdir $<)) > $@

$(PPROC)/p100/%/op-trace-large-batch.csv: $(RESULTS)/p100/% $(OPTRACE_SCRIPTS) | dirs
	@mkdir -p $(dir $@)
	./scripts/get_optrace p100 $(notdir $<) $(shell ./scripts/largebatch p100 $(notdir $<)) > $@

all-op-traces: \
	$(foreach app,$(TRACE_APPS),$(PPROC)/a100/$(app)/op-trace-large-batch.csv) \
	$(foreach app,$(TRACE_APPS),$(PPROC)/v100/$(app)/op-trace-large-batch.csv) \
	$(foreach app,$(TRACE_APPS),$(PPROC)/p100/$(app)/op-trace-large-batch.csv)
