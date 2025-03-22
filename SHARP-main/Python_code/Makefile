# Fixed Configuration Variables
INPUT_DIR = ../input_files/
PROCESSED_PHASE_DIR = ./processed_phase/
DOPPLER_DIR = ./doppler_traces/
DATASET_DIR = ./datasets/
MODEL_DIR = ./models/
RESULTS_DIR = ./results/

NSS = 1
NCORES = 4
SAMPLE_LENGTH = 31
SLIDING = 1
NOISE_LEVEL = -1.2
WINDOW_LENGTH = 340
STRIDE = 30
FEATURE_LENGTH = 100
BATCH_SIZE = 32
ACTIVITIES = E,W,R,J,L,S,C,G
NUM_ANTENNAS = 4

# Domain Definitions
BEDROOM = AR1a,AR1b,AR1c,AR1d,AR1e,AR2a,AR3a,AR3b,AR4a
LIVING_ROOM = AR5a,AR5b
KITCHEN = AR6a
LABORATORY = AR7a
OFFICE = AR8a,AR8b
SEMI_ANECHOIC = AR9a,AR9b,AR9c

# Domain Splits for Source Variation Experiments
BEDROOM_SPLIT1 = AR1a,AR1b,AR1c
BEDROOM_SPLIT2 = AR1d,AR1e,AR2a
LIVING_SPLIT1 = AR5a
LIVING_SPLIT2 = AR5b
OFFICE_SPLIT1 = AR8a
OFFICE_SPLIT2 = AR8b
SEMI_SPLIT1 = AR9a,AR9b
SEMI_SPLIT2 = AR9c

# Add these variables at the top of the Makefile
TF_ENV_VARS = XLA_FLAGS="--xla_gpu_cuda_data_dir=/usr/lib/cuda --xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found" \
              TF_FORCE_GPU_ALLOW_GROWTH=true \
              TF_DEVICE_PLACEMENT_SILENT=1 \
			  PATH=/usr/local/cuda/bin:$$PATH \
			LD_LIBRARY_PATH=/usr/local/cuda/lib64:$$LD_LIBRARY_PATH

# Comma-separated experiment setups (no spaces)
AR_SETUPS = $(BEDROOM),$(LIVING_ROOM),$(KITCHEN),$(LABORATORY),$(OFFICE),$(SEMI_ANECHOIC)
.PHONY: all clean data_processing datasets train test metrics plots all_plots \
        rq1 rq4 leave_one_out varying_sources cross_domain

all: data_processing datasets train test metrics plots

# Data Processing Pipeline
data_processing:
	# Phase Sanitization
	@for setup in $(AR_SETUPS); do \
		echo "Processing $$setup..."; \
		python CSI_phase_sanitization_signal_preprocessing.py "$(INPUT_DIR)$$setup/" 1 - $(NSS) $(NCORES) 0; \
		python CSI_phase_sanitization_H_estimation.py "$(INPUT_DIR)$$setup/" 1 - $(NSS) $(NCORES) 0 -1; \
		python CSI_phase_sanitization_signal_reconstruction.py ./phase_processing/ "$(PROCESSED_PHASE_DIR)$$setup/" $(NSS) $(NCORES) 0 -1; \
	done
	
	# Doppler Computation
	@for setup in $(AR_SETUPS); do \
		echo "Calculating Doppler for $$setup..."; \
		python CSI_doppler_computation.py "$(PROCESSED_PHASE_DIR)" "$$setup" "$(DOPPLER_DIR)" 0 0 $(SAMPLE_LENGTH) $(SLIDING) $(NOISE_LEVEL) --bandwidth 80; \
	done

# Dataset Creation
datasets:
	# Create output directories first
	mkdir -p $(DATASET_DIR)
	
	# Create datasets for all domains
	python CSI_doppler_create_dataset_train.py \
		"$(DOPPLER_DIR)" \
		"$(AR_SETUPS)" \
		$(SAMPLE_LENGTH) \
		$(SLIDING) \
		$(WINDOW_LENGTH) \
		$(STRIDE) \
		"$(ACTIVITIES)" \
		$(NUM_ANTENNAS)
	
	python CSI_doppler_create_dataset_test.py \
		"$(DOPPLER_DIR)" \
		"$(AR_SETUPS)" \
		$(SAMPLE_LENGTH) \
		$(SLIDING) \
		$(WINDOW_LENGTH) \
		$(STRIDE) \
		"$(ACTIVITIES)" \
		$(NUM_ANTENNAS)

# Experiment Targets ----------------------------------------------------------

# Research Question Entry Points
rq1: leave_one_out
rq4: varying_sources

# Leave-One-Domain-Out Evaluation (RQ1)
leave_one_out: leave_one_bedroom leave_one_living leave_one_kitchen leave_one_lab leave_one_office leave_one_semi

leave_one_bedroom:
	$(MAKE) train TEST_DOMAINS="$(BEDROOM)" TRAIN_DOMAINS="$(LIVING_ROOM),$(KITCHEN),$(LABORATORY),$(OFFICE),$(SEMI_ANECHOIC)" MODEL_NAME=no_bedroom
	$(MAKE) test

leave_one_living:
	$(MAKE) train TEST_DOMAINS="$(LIVING_ROOM)" TRAIN_DOMAINS="$(BEDROOM),$(KITCHEN),$(LABORATORY),$(OFFICE),$(SEMI_ANECHOIC)" MODEL_NAME=no_living
	$(MAKE) test

leave_one_kitchen:
	$(MAKE) train TEST_DOMAINS="$(KITCHEN)" TRAIN_DOMAINS="$(BEDROOM),$(LIVING_ROOM),$(LABORATORY),$(OFFICE),$(SEMI_ANECHOIC)" MODEL_NAME=no_kitchen
	$(MAKE) test

leave_one_lab:
	$(MAKE) train TEST_DOMAINS="$(LABORATORY)" TRAIN_DOMAINS="$(BEDROOM),$(LIVING_ROOM),$(KITCHEN),$(OFFICE),$(SEMI_ANECHOIC)" MODEL_NAME=no_lab
	$(MAKE) test

leave_one_office:
	$(MAKE) train TEST_DOMAINS="$(OFFICE)" TRAIN_DOMAINS="$(BEDROOM),$(LIVING_ROOM),$(KITCHEN),$(LABORATORY),$(SEMI_ANECHOIC)" MODEL_NAME=no_office
	$(MAKE) test

leave_one_semi:
	$(MAKE) train TEST_DOMAINS="$(SEMI_ANECHOIC)" TRAIN_DOMAINS="$(BEDROOM),$(LIVING_ROOM),$(KITCHEN),$(LABORATORY),$(OFFICE)" MODEL_NAME=no_semi
	$(MAKE) test

# Varying Source Domain Sizes (RQ4)
varying_sources: source_1 source_2 source_3 source_4

source_1:
	$(MAKE) train TRAIN_DOMAINS="$(BEDROOM_SPLIT1)" MODEL_NAME=source1
	$(MAKE) test TEST_DOMAINS="$(BEDROOM_SPLIT2)"

source_2:
	$(MAKE) train TRAIN_DOMAINS="$(BEDROOM_SPLIT1),$(LIVING_SPLIT1)" MODEL_NAME=source2
	$(MAKE) test TEST_DOMAINS="$(BEDROOM_SPLIT2),$(LIVING_SPLIT2)"

source_3:
	$(MAKE) train TRAIN_DOMAINS="$(BEDROOM_SPLIT1),$(LIVING_SPLIT1),$(OFFICE_SPLIT1)" MODEL_NAME=source3
	$(MAKE) test TEST_DOMAINS="$(BEDROOM_SPLIT2),$(LIVING_SPLIT2),$(OFFICE_SPLIT2)"

source_4:
	$(MAKE) train TRAIN_DOMAINS="$(BEDROOM_SPLIT1),$(LIVING_SPLIT1),$(OFFICE_SPLIT1),$(SEMI_SPLIT1)" MODEL_NAME=source4
	$(MAKE) test TEST_DOMAINS="$(BEDROOM_SPLIT2),$(LIVING_SPLIT2),$(OFFICE_SPLIT2),$(SEMI_SPLIT2)"

# Base Targets with Dynamic Domains --------------------------------------------
train:
	$(TF_ENV_VARS) \
	python CSI_network.py "$(DOPPLER_DIR)" "$(TRAIN_DOMAINS)" \
	$(FEATURE_LENGTH) $(WINDOW_LENGTH) 1 $(BATCH_SIZE) $(NUM_ANTENNAS) $(MODEL_NAME) "$(ACTIVITIES)" \
	--bandwidth 80

test:
	$(TF_ENV_VARS) \
	python CSI_network_test.py "$(DOPPLER_DIR)" "$(TEST_DOMAINS)" \
	$(FEATURE_LENGTH) $(WINDOW_LENGTH) 1 $(BATCH_SIZE) $(NUM_ANTENNAS) $(MODEL_NAME) "$(ACTIVITIES)" \
	--bandwidth 80
	@mv ./outputs/complete_*.txt ./results/$(MODEL_NAME)_results.txt

# Enhanced Metrics for experiments
metrics:
	@echo "Calculating metrics for all experiments..."
	@for result in ./results/*_results.txt; do \
		python CSI_network_metrics.py "$$result" "$(ACTIVITIES)"; \
	done
	@echo "Generated metrics available in ./results/"

# Modified test target with optimized GPU settings (keeping existing content below this line)
test_original:
	$(TF_ENV_VARS) \
	python CSI_network_test.py "$(DOPPLER_DIR)" "$(KITCHEN)" \
	$(FEATURE_LENGTH) $(WINDOW_LENGTH) 1 $(BATCH_SIZE) $(NUM_ANTENNAS) baseline "$(ACTIVITIES)" --bandwidth 80


# Metrics Calculation - Fixed to use the updated scripts with improved error handling
metrics_original:
	@echo "Running metrics calculation..."
	@echo "Step 1: Processing metrics using CSI_network_metrics.py..."
	@if [ -f ./outputs/complete_different_E_W_R_J_L_S_C_G_$(KITCHEN)_band_80_subband_1.txt ]; then \
		python CSI_network_metrics.py ./outputs/complete_different_E_W_R_J_L_S_C_G_$(KITCHEN)_band_80_subband_1.txt "$(ACTIVITIES)" || true; \
	elif [ -f ./outputs/complete_different_E_W_R_J_L_S_C_G_AR6a_band_80_subband_1.txt ]; then \
		python CSI_network_metrics.py ./outputs/complete_different_E_W_R_J_L_S_C_G_AR6a_band_80_subband_1.txt "$(ACTIVITIES)" || true; \
	else \
		echo "No output file found for kitchen. Checking for other output files..."; \
		LATEST_FILE=$$(ls -t ./outputs/complete_different_E_W_R_J_L_S_C_G_*.txt 2>/dev/null | head -1); \
		if [ -n "$$LATEST_FILE" ]; then \
			echo "Using latest output file: $$LATEST_FILE"; \
			python CSI_network_metrics.py "$$LATEST_FILE" "$(ACTIVITIES)" || true; \
		else \
			echo "No output files found. Run 'make test' first."; \
			exit 0; \
		fi; \
	fi
	@echo "Step 2: Creating confusion matrix plots..."
	python -W ignore CSI_network_metrics_plot.py confusion_matrix_baseline || echo "Warning: Error generating confusion matrix plots"
	@echo "Step 3: Creating additional performance visualizations..."
	python -W ignore generate_plots.py || echo "Warning: Error generating additional plots"
	@echo "Metrics calculation and visualization completed"
	@echo "All plots can be found in: $(shell pwd)/plots/"

# Generate additional visualizations for Doppler activity data
plots:
	@echo "Generating Doppler activity visualizations..."
	@mkdir -p ./plots
	@if [ -d "$(DOPPLER_DIR)" ]; then \
		echo "Step 1: Generating Doppler activity plots..."; \
		chmod +x ./CSI_doppler_plot_activities.py; \
		for setup in AR1a AR1b AR1c AR6a; do \
			if [ -d "$(DOPPLER_DIR)$$setup" ]; then \
				echo "Creating Doppler plots for $$setup..."; \
				./CSI_doppler_plot_activities.py "$(DOPPLER_DIR)" "$$setup" $(FEATURE_LENGTH) $(SLIDING) "E,W,R,J,L" 0 340 || \
				echo "Notice: Some errors occurred but plot generation continued"; \
			fi; \
		done; \
		echo "All plots have been generated in the ./plots/ directory"; \
		if [ -d "./plots" ]; then \
			echo "Available plots:"; \
			ls -la ./plots/ | grep -E '\.pdf|\.png' | awk '{print "  - " $$9}'; \
		fi; \
	else \
		echo "Error: Doppler directory not found. Run 'make data_processing' first."; \
		exit 1; \
	fi

# Generate all types of visualizations using the comprehensive generator script
all_plots:
	@echo "Generating all types of visualizations..."
	@mkdir -p ./plots
	@chmod +x ./generate_all_plots.py
	@echo "Step 1: Running comprehensive visualization generator..."
	./generate_all_plots.py \
		--plots-dir "./plots" \
		--phase-dir "$(PROCESSED_PHASE_DIR)" \
		--doppler-dir "$(DOPPLER_DIR)" \
		--feature-length $(FEATURE_LENGTH) \
		--sliding $(SLIDING)
	@echo "Step 2: Generating traditional Doppler plots for specific setups..."
	$(MAKE) plots
	@echo "Step 3: Generating comparison plots..."
	$(MAKE) comparison_plots
	@echo "All visualizations have been generated in the ./plots/ directory"
	@if [ -d "./plots" ]; then \
		echo "Summary of available visualizations:"; \
		echo "  Signal Processing Visualizations:"; \
		ls -la ./plots/ | grep -E 'amplitude|phase|abs_comparison|angle_comparison|gridspec' | awk '{print "    - " $$9}'; \
		echo "  Doppler Spectrum Visualizations:"; \
		ls -la ./plots/ | grep -E 'doppler|antennas|fft' | awk '{print "    - " $$9}'; \
		echo "  Performance Visualizations:"; \
		ls -la ./plots/ | grep -E 'cm_|confusion|accuracy' | awk '{print "    - " $$9}'; \
	fi

# Generate comparison plots across different activities
comparison_plots:
	@echo "Generating comparison plots across activities..."
	@mkdir -p ./plots
	@chmod +x ./generate_doppler_comparison_plots.py
	@if [ -d "$(DOPPLER_DIR)" ]; then \
		for setup in AR1a AR1b AR1c AR6a; do \
			if [ -d "$(DOPPLER_DIR)$$setup" ]; then \
				echo "Creating comparison plots for $$setup..."; \
				./generate_doppler_comparison_plots.py "$(DOPPLER_DIR)" "$$setup" $(FEATURE_LENGTH) $(SLIDING) "E,W,R,J,L" || \
				echo "Notice: Some errors occurred but comparison plot generation continued"; \
			fi; \
		done; \
		echo "All comparison plots have been generated in the ./plots/ directory"; \
		if [ -d "./plots" ]; then \
			echo "Available comparison plots:"; \
			ls -la ./plots/ | grep -E 'comparison|compare|antennas' | awk '{print "  - " $$9}'; \
		fi; \
	else \
		echo "Error: Doppler directory not found. Run 'make data_processing' first."; \
		exit 1; \
	fi
