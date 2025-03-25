# Fixed Configuration Variables
INPUT_DIR = ../input_files/
PROCESSED_PHASE_DIR = ./processed_phase/
DOPPLER_DIR = ./doppler_traces/
DATASET_DIR = ./datasets/
MODEL_DIR = ./models/
RESULTS_DIR = ./results/

# CUDA Environment Configuration for TensorFlow
CUDA_HOME = /usr/lib/cuda
CUDNN_PATH = /usr/lib/x86_64-linux-gnu

# TensorFlow Environment Variables for proper GPU operation
TF_ENV_VARS = CUDA_HOME=$(CUDA_HOME) \
              CUDNN_PATH=$(CUDNN_PATH) \
              TF_CUDA_PATHS=$(CUDA_HOME) \
              XLA_FLAGS="--xla_gpu_cuda_data_dir=$(CUDA_HOME) --xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found" \
              TF_FORCE_GPU_ALLOW_GROWTH=true \
              TF_DEVICE_PLACEMENT_SILENT=1 \
              PATH=/usr/bin:$(CUDA_HOME)/bin:$$PATH \
              LD_LIBRARY_PATH=$(CUDNN_PATH):$(CUDA_HOME)/lib64:$$LD_LIBRARY_PATH

# Add this after existing variable definitions
RQ_NUMBER = $(shell \
    if echo "$(MODEL_NAME)" | grep -qE '^no_(bedroom|kitchen|lab|living|office|semi)$$'; then \
        echo "RQ1_generalization/leave_one_out/$(MODEL_NAME)"; \
    elif echo "$(MODEL_NAME)" | grep -qE '^source[1-4]$$'; then \
        num=$$(echo "$(MODEL_NAME)" | sed 's/source//'); \
        echo "RQ4_source_scaling/source_$$num"; \
    elif echo "$(MODEL_NAME)" | grep -qE '^no_(antenna_randomization|phase_sanitization)$$'; then \
        echo "RQ5_component_analysis/$(MODEL_NAME)"; \
    else \
        echo "RQ1_generalization/generalization_gap/$(MODEL_NAME)"; \
    fi)
RESULTS_PATH = $(RESULTS_DIR)$(RQ_NUMBER)/
EXPERIMENT_TYPE := $(shell echo "$(RQ_NUMBER)" | awk -F/ '{print $$2}')
RQ := $(shell echo "$(RQ_NUMBER)" | awk -F_ '{print $$1}')


# Add this target before existing targets
create_structure:
	@echo "Creating results directory structure..."
	@mkdir -p $(RQ_STRUCTURE)
	@find $(RESULTS_DIR) -type d -exec touch {}/.keep \;

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

# Comma-separated experiment setups (no spaces)
AR_SETUPS = $(BEDROOM),$(LIVING_ROOM),$(KITCHEN),$(LABORATORY),$(OFFICE),$(SEMI_ANECHOIC)
#AR_SETUPS = $(KITCHEN),$(LABORATORY)
.PHONY: all clean data_processing datasets train test metrics plots all_plots \
        rq1 rq4 leave_one_out varying_sources cross_domain check_cuda

# Check CUDA configuration and TensorFlow GPU setup
check_cuda:
	@echo "Checking CUDA and TensorFlow configuration..."
	@echo "CUDA_HOME: $(CUDA_HOME)"
	@echo "CUDNN_PATH: $(CUDNN_PATH)"
	@echo "Verifying ptxas installation..."
	@which ptxas || echo "WARNING: ptxas not found in PATH!"
	@echo "CUDA and TensorFlow environment variables are set up as follows:"
	@echo "TF_CUDA_PATHS=$(CUDA_HOME)"
	@echo "XLA_FLAGS=\"--xla_gpu_cuda_data_dir=$(CUDA_HOME) --xla_gpu_unsafe_fallback_to_driver_on_ptxas_not_found\""
	@echo "TF_FORCE_GPU_ALLOW_GROWTH=true"
	@echo "TF_DEVICE_PLACEMENT_SILENT=1"
	@echo "PATH includes $(CUDA_HOME)/bin"
	@echo "LD_LIBRARY_PATH includes $(CUDNN_PATH) and $(CUDA_HOME)/lib64"

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

test_training:
	make run_complete TEST_DOMAINS="$(LABORATORY)" TRAIN_DOMAINS="$(KITCHEN)" MODEL_NAME=no_bedroom
	@# Ensure environment variables are exported for metrics calculation
	@export TRAIN_DOMAINS="$(KITCHEN)" && \
	export TEST_DOMAINS="$(LABORATORY)" && \
	echo "Running metrics calculation with explicitly set domains..." && \
	find $(RESULTS_PATH)metrics -name 'complete_different_*.txt' -exec bash -c 'TRAIN_DOMAINS="$(KITCHEN)" TEST_DOMAINS="$(LABORATORY)" python CSI_network_metrics.py {} "$(ACTIVITIES)"' \;
	@echo "Metrics recalculated with correct domain values"

# Leave-One-Domain-Out Evaluation (RQ1)
leave_one_out: leave_one_bedroom leave_one_living leave_one_kitchen leave_one_lab leave_one_office leave_one_semi

leave_one_bedroom:
	make run_complete TEST_DOMAINS="$(BEDROOM)" TRAIN_DOMAINS="$(LIVING_ROOM),$(KITCHEN),$(LABORATORY),$(OFFICE),$(SEMI_ANECHOIC)" MODEL_NAME=no_bedroom

leave_one_living:
	make run_complete TEST_DOMAINS="$(LIVING_ROOM)" TRAIN_DOMAINS="$(BEDROOM),$(KITCHEN),$(LABORATORY),$(OFFICE),$(SEMI_ANECHOIC)" MODEL_NAME=no_living

leave_one_kitchen:
	make run_complete TEST_DOMAINS="$(KITCHEN)" TRAIN_DOMAINS="$(BEDROOM),$(LIVING_ROOM),$(LABORATORY),$(OFFICE),$(SEMI_ANECHOIC)" MODEL_NAME=no_kitchen

leave_one_lab:
	make run_complete TEST_DOMAINS="$(LABORATORY)" TRAIN_DOMAINS="$(BEDROOM),$(LIVING_ROOM),$(KITCHEN),$(OFFICE),$(SEMI_ANECHOIC)" MODEL_NAME=no_lab

leave_one_office:
	make run_complete TEST_DOMAINS="$(OFFICE)" TRAIN_DOMAINS="$(BEDROOM),$(LIVING_ROOM),$(KITCHEN),$(LABORATORY),$(SEMI_ANECHOIC)" MODEL_NAME=no_office

leave_one_semi:
	make run_complete TEST_DOMAINS="$(SEMI_ANECHOIC)" TRAIN_DOMAINS="$(BEDROOM),$(LIVING_ROOM),$(KITCHEN),$(LABORATORY),$(OFFICE)" MODEL_NAME=no_semi

# Varying Source Domain Sizes (RQ4)
varying_sources: source_1 source_2 source_3 source_4

source_1:
	make run_complete TRAIN_DOMAINS="$(BEDROOM_SPLIT1)" TEST_DOMAINS="$(BEDROOM_SPLIT2)" MODEL_NAME=source1

source_2:
	make run_complete TRAIN_DOMAINS="$(BEDROOM_SPLIT1),$(LIVING_SPLIT1)" TEST_DOMAINS="$(BEDROOM_SPLIT2),$(LIVING_SPLIT2)" MODEL_NAME=source2

source_3:
	make run_complete TRAIN_DOMAINS="$(BEDROOM_SPLIT1),$(LIVING_SPLIT1),$(OFFICE_SPLIT1)" TEST_DOMAINS="$(BEDROOM_SPLIT2),$(LIVING_SPLIT2),$(OFFICE_SPLIT2)" MODEL_NAME=source3

source_4:
	make run_complete TRAIN_DOMAINS="$(BEDROOM_SPLIT1),$(LIVING_SPLIT1),$(OFFICE_SPLIT1),$(SEMI_SPLIT1)" TEST_DOMAINS="$(BEDROOM_SPLIT2),$(LIVING_SPLIT2),$(OFFICE_SPLIT2),$(SEMI_SPLIT2)" MODEL_NAME=source4

# Base Targets with Dynamic Domains --------------------------------------------
train:
	$(TF_ENV_VARS) \
	python CSI_network.py "$(DOPPLER_DIR)" "$(TRAIN_DOMAINS)" \
	$(FEATURE_LENGTH) $(WINDOW_LENGTH) 1 $(BATCH_SIZE) $(NUM_ANTENNAS) $(MODEL_NAME) "$(ACTIVITIES)" \
	--bandwidth 80

# Automated Test Target
test:
	$(TF_ENV_VARS) \
	python CSI_network_test.py "$(DOPPLER_DIR)" "$(TEST_DOMAINS)" \
	$(FEATURE_LENGTH) $(WINDOW_LENGTH) 1 $(BATCH_SIZE) $(NUM_ANTENNAS) $(MODEL_NAME) "$(ACTIVITIES)" \
	--bandwidth 80
	
	@# Auto-create directory structure properly
	@mkdir -p $(RESULTS_PATH)metrics
	@mkdir -p $(RESULTS_PATH)model
	@mkdir -p $(RESULTS_PATH)plots
	
	@# Move model files to the results directory
	@echo "Moving model for MODEL_NAME=$(MODEL_NAME) to $(RESULTS_PATH)model/"
	@if [ -f "$(MODEL_NAME)_$(ACTIVITIES)_network.h5" ]; then \
		cp "$(MODEL_NAME)_$(ACTIVITIES)_network.h5" "$(RESULTS_PATH)model/" && \
		echo "Model file $(MODEL_NAME)_$(ACTIVITIES)_network.h5 copied to $(RESULTS_PATH)model/"; \
	else \
		echo "Warning: Model file $(MODEL_NAME)_$(ACTIVITIES)_network.h5 not found."; \
		echo "Looking for alternative model files:"; \
		find . -maxdepth 1 -name "$(MODEL_NAME)*_network.h5" -o -name "*$(MODEL_NAME)*_network.h5"; \
		for model_file in $$(find . -maxdepth 1 -name "$(MODEL_NAME)*_network.h5" -o -name "*$(MODEL_NAME)*_network.h5"); do \
			echo "Found model file: $$model_file"; \
			cp "$$model_file" "$(RESULTS_PATH)model/" && \
			echo "Model file $$model_file copied to $(RESULTS_PATH)model/"; \
		done; \
	fi
	
	@# Enhanced file matching and logging
	@echo "Moving results for MODEL_NAME=$(MODEL_NAME) to $(RESULTS_PATH)metrics/"
	@# Modified pattern to be more flexible in matching the output files
	@mv ./results/complete_different_$(MODEL_NAME)_*_band_*.txt $(RESULTS_PATH)metrics/ 2>/dev/null || \
	 mv ./results/complete_different_*_$(MODEL_NAME)_*_band_*.txt $(RESULTS_PATH)metrics/ 2>/dev/null || \
	 echo "Warning: No files found matching either pattern for $(MODEL_NAME). Check output file naming."
	@mv ./results/change_number_antennas_*_$(MODEL_NAME)_*_band_*.txt $(RESULTS_PATH)metrics/ 2>/dev/null || \
	 mv ./results/change_number_antennas_complete_different_*_band_*.txt $(RESULTS_PATH)metrics/ 2>/dev/null || true


		
	@# Generate metadata with auto-detected parameters
	@python generate_meta.py \
		--model $(MODEL_NAME) \
		--type $(EXPERIMENT_TYPE) \
		--rq $(RQ) \
		--path $(RESULTS_PATH)meta.json

# Add this new target before run_complete
clean_run_artifacts:
	@echo "Cleaning up artifacts from previous runs..."
	@rm -f train_activities.txt test_activities.txt common_activities.txt common_activities.csv
	@echo "Temporary activity files removed."
	@echo "Note: To fully clean datasets and models, use 'make clean_datasets' or 'make clean_all'"

# Add a more comprehensive cleanup target
# clean_datasets:
# 	@echo "Cleaning up dataset directories for all domains..."
# 	@for d in $(shell find $(DOPPLER_DIR) -mindepth 1 -maxdepth 1 -type d -exec basename {} \;); do \
# 		echo "Cleaning datasets for domain: $$d"; \
# 		rm -rf $(DOPPLER_DIR)$$d/train_antennas_* $(DOPPLER_DIR)$$d/test_antennas_* $(DOPPLER_DIR)$$d/val_antennas_*; \
# 		rm -f $(DOPPLER_DIR)$$d/files_train_antennas_*.txt $(DOPPLER_DIR)$$d/labels_train_antennas_*.txt $(DOPPLER_DIR)$$d/num_windows_train_antennas_*.txt; \
# 		rm -f $(DOPPLER_DIR)$$d/files_test_antennas_*.txt $(DOPPLER_DIR)$$d/labels_test_antennas_*.txt $(DOPPLER_DIR)$$d/num_windows_test_antennas_*.txt; \
# 		rm -f $(DOPPLER_DIR)$$d/files_val_antennas_*.txt $(DOPPLER_DIR)$$d/labels_val_antennas_*.txt $(DOPPLER_DIR)$$d/num_windows_val_antennas_*.txt; \
# 	done
# 	@echo "Dataset directories cleaned."

# Add comprehensive cleanup target
clean_all: clean_run_artifacts
	@echo "Removing models..."
	@rm -rf $(MODEL_DIR)*.h5
	@echo "Removing results..."
	@rm -rf ./results/*
	@echo "Removing plots..."
	@rm -rf ./plots/*
	@echo "All artifacts cleaned."

# Add a target to regenerate metrics with correct domain values
regenerate_metrics:
	@echo "Regenerating metrics with correct domain values..."
	@echo "Source domains: $(TRAIN_DOMAINS)"
	@echo "Target domains: $(TEST_DOMAINS)"
	@if [ -z "$(TRAIN_DOMAINS)" ] || [ -z "$(TEST_DOMAINS)" ]; then \
		echo "Error: TRAIN_DOMAINS and TEST_DOMAINS must be specified."; \
		echo "Usage: make regenerate_metrics TRAIN_DOMAINS=domain1,domain2 TEST_DOMAINS=domain3,domain4 MODEL_NAME=model_name"; \
		exit 1; \
	fi
	@if [ -z "$(MODEL_NAME)" ]; then \
		echo "Error: MODEL_NAME must be specified."; \
		echo "Usage: make regenerate_metrics TRAIN_DOMAINS=domain1,domain2 TEST_DOMAINS=domain3,domain4 MODEL_NAME=model_name"; \
		exit 1; \
	fi
	@if [ -z "$(ACTIVITIES)" ]; then \
		ACTIVITIES="E,W,R,J,L,S,C,G"; \
		echo "Using default activities: $$ACTIVITIES"; \
	else \
		ACTIVITIES="$(ACTIVITIES)"; \
		echo "Using specified activities: $$ACTIVITIES"; \
	fi
	@# Remove existing metrics file if it exists
	@if [ -f "$(RESULTS_DIR)experiment_metrics.csv" ]; then \
		rm "$(RESULTS_DIR)experiment_metrics.csv"; \
		echo "Removed existing experiment_metrics.csv file"; \
	fi
	@# Use export to ensure environment variables are passed to child processes
	@export TRAIN_DOMAINS="$(TRAIN_DOMAINS)" && \
	export TEST_DOMAINS="$(TEST_DOMAINS)" && \
	echo "Looking for metrics files in $(RESULTS_DIR)$(RQ_NUMBER)/metrics/..." && \
	METRICS_FILE=$$(find $(RESULTS_DIR)$(RQ_NUMBER)/metrics/ -name 'complete_different_*.txt' | head -n 1) && \
	if [ -n "$$METRICS_FILE" ]; then \
		echo "Processing metrics file: $$METRICS_FILE" && \
		TRAIN_DOMAINS="$(TRAIN_DOMAINS)" TEST_DOMAINS="$(TEST_DOMAINS)" python CSI_network_metrics.py "$$METRICS_FILE" "$$ACTIVITIES"; \
	else \
		echo "No metrics files found in $(RESULTS_DIR)$(RQ_NUMBER)/metrics/. Trying results directory..." && \
		METRICS_FILE=$$(find $(RESULTS_DIR) -name 'complete_different_*.txt' | head -n 1) && \
		if [ -n "$$METRICS_FILE" ]; then \
			echo "Processing metrics file: $$METRICS_FILE" && \
			TRAIN_DOMAINS="$(TRAIN_DOMAINS)" TEST_DOMAINS="$(TEST_DOMAINS)" python CSI_network_metrics.py "$$METRICS_FILE" "$$ACTIVITIES"; \
		else \
			echo "No metrics files found. Please run the experiment first."; \
			exit 1; \
		fi; \
	fi
	@echo "Metrics regenerated with correct domain values"
	@if [ -f "$(RESULTS_DIR)experiment_metrics.csv" ]; then \
		echo "New experiment_metrics.csv:"; \
		cat "$(RESULTS_DIR)experiment_metrics.csv"; \
	fi

# Modify the run_complete target to call clean_run_artifacts at the beginning
run_complete: clean_run_artifacts
	@echo "===== Starting complete workflow for MODEL_NAME=$(MODEL_NAME) ====="
	@echo "0. Determining common activities between training and test directories..."
	
	@# Check if directories exist
	@for d in $(TRAIN_DOMAINS); do \
		if [ ! -d "$(DOPPLER_DIR)$$d" ]; then \
			echo "Error: Training directory $(DOPPLER_DIR)$$d does not exist."; \
			exit 1; \
		fi; \
	done
	@for d in $(TEST_DOMAINS); do \
		if [ ! -d "$(DOPPLER_DIR)$$d" ]; then \
			echo "Error: Test directory $(DOPPLER_DIR)$$d does not exist."; \
			exit 1; \
		fi; \
	done
	
	@# Step 1: Extract activities from training directories
	@echo "Extracting activities from training directories: $(TRAIN_DOMAINS)"
	@echo "Looking for activity directories..."
	@for d in $(TRAIN_DOMAINS); do \
		echo "Checking domain: $$d"; \
		find $(DOPPLER_DIR)$$d -maxdepth 1 -type d -name "$$d"_* | \
		grep -v "train_antennas\|test_antennas\|val_antennas\|complete_antennas" || echo "No matching directories found"; \
	done
	@(for d in $(TRAIN_DOMAINS); do \
		find $(DOPPLER_DIR)$$d -maxdepth 1 -type d -name "$$d"_* | \
		grep -v "train_antennas\|test_antennas\|val_antennas\|complete_antennas" | \
		xargs -I{} basename {} | cut -d '_' -f 2 | cut -c1; \
	done) | sort -u > train_activities.txt
	@if [ ! -s train_activities.txt ]; then \
		echo "Error: No activity directories found in training domains."; \
		echo "Available directories in training domains:"; \
		for d in $(TRAIN_DOMAINS); do \
			echo "Content of $(DOPPLER_DIR)$$d/:"; \
			ls -la $(DOPPLER_DIR)$$d/; \
		done; \
		exit 1; \
	fi
	@echo "Found training activities: $$(cat train_activities.txt | tr '\n' ' ')"

	@# Step 2: Extract activities from test directories
	@echo "Extracting activities from test directories: $(TEST_DOMAINS)"
	@echo "Looking for activity directories..."
	@for d in $(TEST_DOMAINS); do \
		echo "Checking domain: $$d"; \
		find $(DOPPLER_DIR)$$d -maxdepth 1 -type d -name "$$d"_* | \
		grep -v "train_antennas\|test_antennas\|val_antennas\|complete_antennas" || echo "No matching directories found"; \
	done
	@(for d in $(TEST_DOMAINS); do \
		find $(DOPPLER_DIR)$$d -maxdepth 1 -type d -name "$$d"_* | \
		grep -v "train_antennas\|test_antennas\|val_antennas\|complete_antennas" | \
		xargs -I{} basename {} | cut -d '_' -f 2 | cut -c1; \
	done) | sort -u > test_activities.txt
	@if [ ! -s test_activities.txt ]; then \
		echo "Error: No activity directories found in test domains."; \
		echo "Available directories in test domains:"; \
		for d in $(TEST_DOMAINS); do \
			echo "Content of $(DOPPLER_DIR)$$d/:"; \
			ls -la $(DOPPLER_DIR)$$d/; \
		done; \
		exit 1; \
	fi
	@echo "Found test activities: $$(cat test_activities.txt | tr '\n' ' ')"
	
	@# Step 3: Compute intersection of activities
	@echo "Computing intersection of activities..."
	@echo "Training activities: $$(cat train_activities.txt | tr '\n' ' ')"
	@echo "Test activities: $$(cat test_activities.txt | tr '\n' ' ')"
	
	@# Use grep instead of comm for more robust intersection
	@(for a in $$(cat train_activities.txt); do \
		if grep -q "^$$a$$" test_activities.txt; then \
			echo $$a; \
		fi; \
	done) > common_activities.txt
	
	@echo "Common activities found: $$(cat common_activities.txt | tr '\n' ' ')"
	
	@# Check if common_activities.txt has content before proceeding
	@if [ ! -s common_activities.txt ]; then \
		echo "Error: No common activities found between training and test directories."; \
		echo "Training activities: $$(cat train_activities.txt | tr '\n' ' ')"; \
		echo "Test activities: $$(cat test_activities.txt | tr '\n' ' ')"; \
		exit 1; \
	fi
	
	@# Convert to comma-separated list and directly save to variable
	@ACTIVITIES="$$(cat common_activities.txt | tr '\n' ',' | sed 's/,$$//')" && \
	echo "Common activities: $$ACTIVITIES" && \
	if [ -z "$$ACTIVITIES" ]; then \
		echo "Error: No common activities found between training and test directories."; \
		exit 1; \
	fi && \
	echo "1. Creating datasets with activities: $$ACTIVITIES" && \
	python CSI_doppler_create_dataset_train.py \
		"$(DOPPLER_DIR)" \
		"$(TRAIN_DOMAINS)" \
		$(SAMPLE_LENGTH) \
		$(SLIDING) \
		$(WINDOW_LENGTH) \
		$(STRIDE) \
		"$$ACTIVITIES" \
		$(NUM_ANTENNAS) && \
	python CSI_doppler_create_dataset_test.py \
		"$(DOPPLER_DIR)" \
		"$(TEST_DOMAINS)" \
		$(SAMPLE_LENGTH) \
		$(SLIDING) \
		$(WINDOW_LENGTH) \
		$(STRIDE) \
		"$$ACTIVITIES" \
		$(NUM_ANTENNAS) && \
	echo "2. Training model with activities: $$ACTIVITIES..." && \
	$(MAKE) train ACTIVITIES="$$ACTIVITIES" && \
	echo "3. Testing model with activities: $$ACTIVITIES..." && \
	$(MAKE) test ACTIVITIES="$$ACTIVITIES" && \
	echo "4. Calculating metrics..." && \
	$(MAKE) model_metrics ACTIVITIES="$$ACTIVITIES" && \
	echo "5. Generating plots..." && \
	$(MAKE) model_plots ACTIVITIES="$$ACTIVITIES" && \
	echo "===== Complete workflow finished successfully for MODEL_NAME=$(MODEL_NAME) ====="

# Per-model metrics calculation
model_metrics:
	@echo "Calculating metrics for $(MODEL_NAME)..."
	@find $(RESULTS_PATH)metrics -name 'complete_different_*.txt' -exec python CSI_network_metrics.py {} "$(ACTIVITIES)" \;
	@echo "Metrics calculated and saved to $(RESULTS_PATH)metrics/"

# Per-model plots generation 
model_plots:
	@echo "Generating plots for $(MODEL_NAME)..."
	@mkdir -p $(RESULTS_PATH)plots
	@# Generate confusion matrix plots
	@find $(RESULTS_PATH)metrics -name 'confusion_matrix_*.txt' -exec python -W ignore CSI_network_metrics_plot.py {} \; 2>/dev/null || true
	@# Generate activity plots for specific environments in the test domains
	@for domain in $(TEST_DOMAINS); do \
		echo "Creating activity plots for domain: $$domain"; \
		python CSI_doppler_plot_activities.py "$(DOPPLER_DIR)" "$$domain" $(FEATURE_LENGTH) $(SLIDING) "$(ACTIVITIES)" 0 $(WINDOW_LENGTH) 2>/dev/null || true; \
	done
	@# Copy generated plots to the model's plots directory
	@find ./plots -type f -name "*.png" -o -name "*.pdf" -exec cp {} $(RESULTS_PATH)plots/ \; 2>/dev/null || true
	@echo "Plots generated and saved to $(RESULTS_PATH)plots/"


# Enhanced Metrics for experiments
metrics:
	@echo "Calculating metrics for all experiments..."
	@mkdir -p $(RESULTS_DIR)  # Ensure base directory exists
	@find $(RESULTS_DIR) -name 'complete_different_*.txt' -exec python CSI_network_metrics.py {} "$(ACTIVITIES)" \;
	@echo "Metrics processed in respective RQ directories."


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
	make plots
	@echo "Step 3: Generating comparison plots..."
	make comparison_plots
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

# Add a target to run a completely clean experiment
clean_experiment: clean_all
	@echo "Starting clean experiment with TEST_DOMAINS=$(TEST_DOMAINS) and TRAIN_DOMAINS=$(TRAIN_DOMAINS)..."
	@$(MAKE) run_complete TEST_DOMAINS="$(TEST_DOMAINS)" TRAIN_DOMAINS="$(TRAIN_DOMAINS)" MODEL_NAME="$(MODEL_NAME)"
