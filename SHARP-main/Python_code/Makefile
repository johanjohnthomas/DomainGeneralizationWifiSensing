# Define variables for directories and parameters
INPUT_DIR = ../input_files/
OUTPUT_DIR = ./processed_phase/
DOPPLER_DIR = ./doppler_traces/
NSS = 1  # Number of spatial streams
NCORE = 4  # Number of cores
START_IDX = 0  # Index to start processing
SAMPLE_LENGTH = 31  # Number of packets in a sample
SLIDING = 1  # Number of packets for sliding operations
NOISE_LEVEL = -1.2  # Noise level for thresholding
WINDOWS_LENGTH = 340  # Number of samples per window
STRIDE_LENGTHS = 30  # Number of samples for window sliding
FEATURE_LENGTH = 100  # Length along the feature dimension (height)
BATCH_SIZE = 32  # Number of samples in a batch
NAME_BASE = single_ant  # Name prefix for the files
ACTIVITIES = E,L,W,R,J  # Activities to be considered

# Define the commands for each step
PREPROCESS_CMD = for subdir in $(INPUT_DIR)AR-*; do \
    dir_name=$$(basename "$$subdir"); \
    python CSI_phase_sanitization_signal_preprocessing.py "$$subdir/" 1 - $(NSS) $(NCORE) $(START_IDX); \
done
H_ESTIMATION_CMD = for signal_file in ./phase_processing/signal_*.txt; do \
    name=$$(basename "$$signal_file" .txt | sed 's/signal_//'); \
    python CSI_phase_sanitization_H_estimation.py ./phase_processing/ 0 $$name $(NSS) $(NCORE) $(START_IDX) -1; \
done
RECONSTRUCTION_CMD = for subdir in $(INPUT_DIR)AR-*; do \
    dir_name=$$(basename "$$subdir"); \
    python CSI_phase_sanitization_signal_reconstruction.py ./phase_processing/ $(OUTPUT_DIR) $(NSS) $(NCORE) $(START_IDX) -1; \
done
DOPPLER_CMD = python CSI_doppler_computation.py $(OUTPUT_DIR) AR-1a,AR-1b,AR-1c,AR-2a,AR-2b,AR-3a,AR-4a,AR-4b,AR-5a,AR-6a,AR-6b,AR-7a $(DOPPLER_DIR) 800 800 $(SAMPLE_LENGTH) $(SLIDING) $(NOISE_LEVEL)
CREATE_DATASET_TRAIN_CMD = python CSI_doppler_create_dataset_train.py $(DOPPLER_DIR) $(SUBDIRS_TRAIN) $(SAMPLE_LENGTH) $(SLIDING) $(WINDOWS_LENGTH) $(STRIDE_LENGTHS) $(ACTIVITIES) 4
CREATE_DATASET_TEST_CMD = python CSI_doppler_create_dataset_test.py $(DOPPLER_DIR) $(SUBDIRS_TEST) $(SAMPLE_LENGTH) $(SLIDING) $(WINDOWS_LENGTH) $(STRIDE_LENGTHS) $(ACTIVITIES) 4
TRAIN_CMD = python CSI_network.py $(DOPPLER_DIR) $(SUBDIRS_TRAIN) $(FEATURE_LENGTH) $(WINDOWS_LENGTH) 1 $(BATCH_SIZE) $(NAME_BASE) $(ACTIVITIES)
TEST_CMD = python CSI_network_test.py $(DOPPLER_DIR) $(SUBDIRS_TEST) $(FEATURE_LENGTH) $(WINDOWS_LENGTH) 1 $(BATCH_SIZE) $(NAME_BASE) $(ACTIVITIES)

# Define the targets for each variation
no_unseen_domains:
	$(PREPROCESS_CMD)
	$(H_ESTIMATION_CMD)
	$(RECONSTRUCTION_CMD)
	$(DOPPLER_CMD)
	$(eval SUBDIRS_TRAIN = AR-1a,AR-1b,AR-1c,AR-3a,AR-4a,AR-5a,AR-6a,AR-7a,AR-8a,AR-9a,AR-9b)
	$(eval SUBDIRS_TEST = AR-1d,AR-1e,AR-2a,AR-5b,AR-8b,AR-9c)
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

one_unseen_domain_bedroom_hidden:
	$(PREPROCESS_CMD)
	$(H_ESTIMATION_CMD)
	$(RECONSTRUCTION_CMD)
	$(DOPPLER_CMD)
	$(eval SUBDIRS_TRAIN = AR-5a,AR-5b,AR-6a,AR-7a,AR-8a,AR-8b,AR-9a,AR-9b,AR-9c)
	$(eval SUBDIRS_TEST = AR-1a,AR-1b,AR-1c,AR-1d,AR-1e,AR-2a,AR-3a,AR-3b,AR-4a)
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

one_unseen_domain_living_room_hidden:
	$(PREPROCESS_CMD)
	$(H_ESTIMATION_CMD)
	$(RECONSTRUCTION_CMD)
	$(DOPPLER_CMD)
	$(eval SUBDIRS_TRAIN = AR-1a,AR-1b,AR-1c,AR-1d,AR-1e,AR-2a,AR-3a,AR-3b,AR-4a,AR-6a,AR-7a,AR-8a,AR-8b,AR-9a,AR-9b,AR-9c)
	$(eval SUBDIRS_TEST = AR-5a,AR-5b)
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

one_unseen_domain_kitchen_hidden:
	$(PREPROCESS_CMD)
	$(H_ESTIMATION_CMD)
	$(RECONSTRUCTION_CMD)
	$(DOPPLER_CMD)
	$(eval SUBDIRS_TRAIN = AR-1a,AR-1b,AR-1c,AR-1d,AR-1e,AR-2a,AR-3a,AR-3b,AR-4a,AR-5a,AR-5b,AR-7a,AR-8a,AR-8b,AR-9a,AR-9b,AR-9c)
	$(eval SUBDIRS_TEST = AR-6a)
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

one_unseen_domain_laboratory_hidden:
	$(PREPROCESS_CMD)
	$(H_ESTIMATION_CMD)
	$(RECONSTRUCTION_CMD)
	$(DOPPLER_CMD)
	$(eval SUBDIRS_TRAIN = AR-1a,AR-1b,AR-1c,AR-1d,AR-1e,AR-2a,AR-3a,AR-3b,AR-4a,AR-5a,AR-5b,AR-6a,AR-8a,AR-8b,AR-9a,AR-9b,AR-9c)
	$(eval SUBDIRS_TEST = AR-7a)
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

one_unseen_domain_office_hidden:
	$(PREPROCESS_CMD)
	$(H_ESTIMATION_CMD)
	$(RECONSTRUCTION_CMD)
	$(DOPPLER_CMD)
	$(eval SUBDIRS_TRAIN = AR-1a,AR-1b,AR-1c,AR-1d,AR-1e,AR-2a,AR-3a,AR-3b,AR-4a,AR-5a,AR-5b,AR-6a,AR-7a,AR-9a,AR-9b,AR-9c)
	$(eval SUBDIRS_TEST = AR-8a,AR-8b)
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

one_unseen_domain_semi_anechoic_chamber_hidden:
	$(PREPROCESS_CMD)
	$(H_ESTIMATION_CMD)
	$(RECONSTRUCTION_CMD)
	$(DOPPLER_CMD)
	$(eval SUBDIRS_TRAIN = AR-1a,AR-1b,AR-1c,AR-1d,AR-1e,AR-2a,AR-3a,AR-3b,AR-4a,AR-5a,AR-5b,AR-6a,AR-7a,AR-8a,AR-8b)
	$(eval SUBDIRS_TEST = AR-9a,AR-9b,AR-9c)
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

multiple_unseen_domains_bedroom_and_semi_anechoic_chamber_hidden:
	$(PREPROCESS_CMD)
	$(H_ESTIMATION_CMD)
	$(RECONSTRUCTION_CMD)
	$(DOPPLER_CMD)
	$(eval SUBDIRS_TRAIN = AR-5a,AR-5b,AR-6a,AR-7a,AR-8a,AR-8b)
	$(eval SUBDIRS_TEST = AR-1a,AR-1b,AR-1c,AR-1d,AR-1e,AR-2a,AR-3a,AR-3b,AR-4a,AR-9a,AR-9b,AR-9c)
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

multiple_unseen_domains_equivalent_data_kitchen_laboratory_hidden:
	$(PREPROCESS_CMD)
	$(H_ESTIMATION_CMD)
	$(RECONSTRUCTION_CMD)
	$(DOPPLER_CMD)
	$(eval SUBDIRS_TRAIN = AR-1a,AR-1b,AR-3a,AR-3b,AR-5a,AR-5b,AR-8a,AR-8b,AR-9a,AR-9b)
	$(eval SUBDIRS_TEST = AR-1c,AR-1d,AR-1e,AR-2a,AR-4a,AR-6a,AR-7a,AR-9c)
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

multiple_unseen_domains_equivalent_data_kitchen_laboratory_hidden_test2:
	$(PREPROCESS_CMD)
	$(H_ESTIMATION_CMD)
	$(RECONSTRUCTION_CMD)
	$(DOPPLER_CMD)
	$(eval SUBDIRS_TRAIN = AR-1a,AR-1b,AR-3a,AR-3b,AR-5a,AR-5b,AR-8a,AR-8b,AR-9a,AR-9b)
	$(eval SUBDIRS_TEST = AR-6a,AR-7a)
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

# Default target (runs all variations)
all: no_unseen_domains one_unseen_domain_bedroom_hidden one_unseen_domain_living_room_hidden \
	 one_unseen_domain_kitchen_hidden one_unseen_domain_laboratory_hidden one_unseen_domain_office_hidden \
	 one_unseen_domain_semi_anechoic_chamber_hidden multiple_unseen_domains_bedroom_and_semi_anechoic_chamber_hidden \
	 multiple_unseen_domains_equivalent_data_kitchen_laboratory_hidden \
	 multiple_unseen_domains_equivalent_data_kitchen_laboratory_hidden_test2