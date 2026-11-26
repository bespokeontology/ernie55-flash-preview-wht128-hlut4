# Pure-C++17 / gfx906 HIP / Blackhole C++20 TT-Metalium Build System
# Model: baidu/ERNIE-5.5-Flash (280B-A14B, 1M context, 100% uniform WHT128-HLUT4, 0 B CPU DDR4)
# Zero Python anywhere.

CXX      ?= g++
HIPCC    ?= /opt/rocm/bin/hipcc
GCC_DIR  ?= /usr/lib/gcc/x86_64-linux-gnu/11

PREFIX_MAP := -ffile-prefix-map=$(CURDIR)=. -fmacro-prefix-map=$(CURDIR)=.
CXXFLAGS := -std=c++17 -O3 -march=native -fopenmp -pthread -Wall -Wextra -Werror -Wno-unused-parameter $(PREFIX_MAP)
HIPFLAGS := --gcc-install-dir=$(GCC_DIR) -fPIC -std=c++17 -O3 --offload-arch=gfx906 -fno-gpu-rdc \
            -Wall -Wextra -Werror -Wno-unused-parameter -Wno-unused-command-line-argument $(PREFIX_MAP)
ROCM_INC := -I/opt/rocm/include -Isrc -Imi50 -Ip150
ROCM_LIB := -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lamdhip64 -lrocblas -lpcre2-8 -fopenmp -pthread -lrt

BUILD := build
BIN   := bin

CORE_OBJS := $(BUILD)/ernie55_hlut4_kernels.o \
             $(BUILD)/ernie55_core_kernels.o \
             $(BUILD)/ernie55_tokenizer.o \
             $(BUILD)/ernie55_runtime.o

BINS := $(BIN)/ernie55_fe \
        $(BIN)/ernie55_build_weights \
        $(BIN)/ernie55_build_weights_gpu \
        $(BIN)/ernie55_build_cal_corpus \
        $(BIN)/ernie55_swap_checkpoint \
        $(BIN)/ernie55_verify_unit \
        $(BIN)/ernie55_verify_gates \
        $(BIN)/ernie55_gate3_live \
        $(BIN)/ernie55_gate6_campaign \
        $(BIN)/ernie55_chart_gen \
        $(BIN)/ernie55_sanitize_verifier \
        $(BIN)/ernie55_sanitize_session \
        $(BIN)/ernie55_session_to_md \
        $(BIN)/ernie55_kvblk_verify \
        $(BIN)/ernie55_wave10_xray_verify \
        $(BIN)/bench_single_layer_gfx906

.PHONY: all p150 audit-gfx906 verify-unit clean

all: $(BINS)

$(BUILD) $(BIN):
	@mkdir -p $(BUILD) $(BIN)

$(BUILD)/ernie55_tokenizer.o: src/ernie55_tokenizer.cpp src/ernie55_tokenizer.hpp src/ernie55_json.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -Isrc -c $< -o $@

$(BUILD)/ernie55_hlut4_kernels.o: mi50/ernie55_hlut4_kernels.hip mi50/ernie55_hlut4_kernels.hpp src/ernie55_hlut4_pack.hpp src/ernie55_config.hpp | $(BUILD)
	$(HIPCC) $(HIPFLAGS) -Isrc -Imi50 -c $< -o $@

$(BUILD)/ernie55_core_kernels.o: mi50/ernie55_core_kernels.hip mi50/ernie55_core_kernels.hpp src/ernie55_config.hpp | $(BUILD)
	$(HIPCC) $(HIPFLAGS) -Isrc -Imi50 -c $< -o $@

$(BUILD)/ernie55_runtime.o: mi50/ernie55_runtime.cpp mi50/ernie55_runtime.hpp mi50/ernie55_core_kernels.hpp mi50/ernie55_hlut4_kernels.hpp mi50/ernie55_kvblk_cache.hpp src/ernie55_config.hpp src/ernie55_dspark.hpp src/ernie55_hlut4_pack.hpp src/ernie55_tokenizer.hpp p150/ernie55_p150_shm.hpp | $(BUILD)
	$(CXX) $(CXXFLAGS) -D__HIP_PLATFORM_AMD__ $(ROCM_INC) -c $< -o $@

$(BIN)/ernie55_fe: mi50/ernie55_fe.cpp $(CORE_OBJS) | $(BIN)
	$(CXX) $(CXXFLAGS) -D__HIP_PLATFORM_AMD__ $(ROCM_INC) $< $(CORE_OBJS) $(ROCM_LIB) -o $@

$(BIN)/ernie55_build_weights: tools/ernie55_build_weights.cpp $(BUILD)/ernie55_tokenizer.o src/ernie55_config.hpp src/ernie55_hlut4_pack.hpp src/ernie55_range_stream.hpp src/ernie55_safetensors_writer.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -Isrc $< $(BUILD)/ernie55_tokenizer.o -lpcre2-8 -fopenmp -pthread -o $@

$(BIN)/ernie55_build_weights_gpu: tools/ernie55_build_weights_gpu.cpp $(BUILD)/ernie55_hlut4_kernels.o $(BUILD)/ernie55_core_kernels.o src/ernie55_config.hpp src/ernie55_hlut4_pack.hpp src/ernie55_range_stream.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -D__HIP_PLATFORM_AMD__ $(ROCM_INC) $< $(BUILD)/ernie55_hlut4_kernels.o $(BUILD)/ernie55_core_kernels.o -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lamdhip64 -lrocblas -fopenmp -pthread -o $@

$(BIN)/ernie55_build_cal_corpus: tools/ernie55_build_cal_corpus.cpp $(BUILD)/ernie55_tokenizer.o src/ernie55_config.hpp src/ernie55_hlut4_pack.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -Isrc $< $(BUILD)/ernie55_tokenizer.o -lpcre2-8 -fopenmp -pthread -o $@

$(BIN)/ernie55_swap_checkpoint: tools/ernie55_swap_checkpoint.cpp $(BUILD)/ernie55_tokenizer.o src/ernie55_config.hpp src/ernie55_dspark.hpp src/ernie55_range_stream.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -Isrc $< $(BUILD)/ernie55_tokenizer.o -lpcre2-8 -fopenmp -pthread -o $@

$(BIN)/ernie55_verify_unit: tools/ernie55_verify_unit.cpp $(BUILD)/ernie55_tokenizer.o src/ernie55_config.hpp src/ernie55_dspark.hpp src/ernie55_hlut4_pack.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -Isrc $< $(BUILD)/ernie55_tokenizer.o -lpcre2-8 -fopenmp -pthread -o $@

$(BIN)/ernie55_verify_gates: tools/ernie55_verify_gates.cpp $(CORE_OBJS) | $(BIN)
	$(CXX) $(CXXFLAGS) -D__HIP_PLATFORM_AMD__ $(ROCM_INC) $< $(CORE_OBJS) $(ROCM_LIB) -o $@

$(BIN)/ernie55_gate3_live: tools/ernie55_gate3_live.cpp $(BUILD)/ernie55_tokenizer.o src/ernie55_json.hpp src/ernie55_tokenizer.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -Isrc $< $(BUILD)/ernie55_tokenizer.o -lpcre2-8 -fopenmp -pthread -o $@

$(BIN)/ernie55_gate6_campaign: tools/ernie55_gate6_campaign.cpp src/ernie55_json.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -Isrc $< -o $@

$(BIN)/ernie55_chart_gen: tools/ernie55_chart_gen.cpp src/ernie55_json.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -Isrc $< -o $@

$(BIN)/ernie55_sanitize_verifier: tools/ernie55_sanitize_verifier.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN)/ernie55_sanitize_session: tools/ernie55_sanitize_session.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN)/ernie55_session_to_md: tools/ernie55_session_to_md.cpp src/ernie55_json.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -Isrc $< -o $@

$(BIN)/ernie55_kvblk_verify: tools/ernie55_kvblk_verify.cpp mi50/ernie55_kvblk_cache.hpp src/ernie55_json.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -D__HIP_PLATFORM_AMD__ $(ROCM_INC) $< -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lamdhip64 -o $@

$(BIN)/ernie55_wave10_xray_verify: tools/ernie55_wave10_xray_verify.cpp | $(BIN)
	$(CXX) $(CXXFLAGS) $< -o $@

$(BIN)/bench_single_layer_gfx906: tools/bench_single_layer_gfx906.cpp $(BUILD)/ernie55_hlut4_kernels.o $(BUILD)/ernie55_core_kernels.o src/ernie55_config.hpp src/ernie55_hlut4_pack.hpp | $(BIN)
	$(CXX) $(CXXFLAGS) -D__HIP_PLATFORM_AMD__ $(ROCM_INC) $< $(BUILD)/ernie55_hlut4_kernels.o $(BUILD)/ernie55_core_kernels.o -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib -lamdhip64 -lrocblas -fopenmp -pthread -o $@

p150:
	bash p150/build.sh p150_ernie55_worker

audit-gfx906: | $(BUILD)
	$(HIPCC) $(HIPFLAGS) -Isrc -Imi50 --cuda-device-only -S mi50/ernie55_hlut4_kernels.hip -o $(BUILD)/ernie55_hlut4_kernels.s
	$(HIPCC) $(HIPFLAGS) -Isrc -Imi50 --cuda-device-only -S mi50/ernie55_core_kernels.hip -o $(BUILD)/ernie55_core_kernels.s
	@for f in $(BUILD)/ernie55_hlut4_kernels.s $(BUILD)/ernie55_core_kernels.s; do \
	  echo "=== ISA AUDIT $$f ==="; \
	  k=$$(grep -c '\.amdhsa_kernel' "$$f"); \
	  sc=$$(grep -E 'ScratchSize:|private_segment_fixed_size' "$$f" | grep -v -E '[: ]0$$' | wc -l); \
	  sp=$$(grep -E 'scratch_load|scratch_store' "$$f" | wc -l); \
	  tr=$$(grep -E 'v_sin_f32|v_cos_f32' "$$f" | wc -l); \
	  sl=$$(grep -E '\bslc\b' "$$f" | wc -l); \
	  fd=$$(grep -c 'v_dot2_f32_f16' "$$f" || true); \
	  echo "  kernels=$$k scratch>0=$$sc spills=$$sp trig=$$tr slc=$$sl fdot2=$$fd"; \
	  (test "$$sc" -eq 0 && test "$$sp" -eq 0 && test "$$tr" -eq 0 && test "$$sl" -eq 0) || exit 1; \
	done
	@test $$(find . -name '*.py' | wc -l) -eq 0
	@echo "=== GFX906 ISA AUDIT & ZERO-PYTHON CHECK PASSED 100% ==="

verify-unit: $(BIN)/ernie55_verify_unit
	./$(BIN)/ernie55_verify_unit

clean:
	rm -rf $(BUILD)/*.o $(BUILD)/*.s $(BINS)
