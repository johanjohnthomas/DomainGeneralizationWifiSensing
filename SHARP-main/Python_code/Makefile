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
H_ESTIMATION_CMD = python parallel_h_estimation.py
RECONSTRUCTION_CMD = for subdir in $(INPUT_DIR)AR-*; do \
    dir_name=$$(basename "$$subdir"); \
    python CSI_phase_sanitization_signal_reconstruction.py ./phase_processing/ $(OUTPUT_DIR) $(NSS) $(NCORE) $(START_IDX) -1; \
done
SUBDIRS_DOPPLER = AR7a_J3,AR8a_J2,AR9a_J2,AR3a_R,AR6a_W2,AR8b_L1,AR1a_S,AR8b_J2,AR1d_S,AR1a_E,AR2a_J1,AR9b_W2,AR5a_R,AR2a_J2,AR5a_J2,AR9c_J2,AR1c_S,AR1c_H,AR6a_R2,AR1e_J1,AR5a_L,AR1e_L,AR3a_S,AR9c_L1,AR6a_J1,AR8a_R2,AR4a_C1,AR9b_J1,AR8a_R1,AR7a_W,AR9a_J1,AR1c_W,AR5b_W,AR2a_H,AR9c_R2,AR4a_R,AR1b_E,AR1c_C,AR1b_J1,AR2a_L,AR9b_L3,AR8b_J1,AR2a_W,AR6a_L2,AR6a_W1,AR8b_W1,AR8b_W2,AR9a_L2,AR3b_R,AR3a_H1,AR9c_W1,AR7a_R,AR5b_J1,AR4a_J2,AR8b_E1,AR6a_E,AR1e_W,AR7a_L2,AR1d_C1,AR7a_S,AR8a_L1,AR3a_C2,AR1e_J2,AR1d_C2,AR4a_C2,AR8a_L3,AR7a_C,AR3b_E,AR3b_W,AR1d_E,AR9c_L3,AR3a_J1,AR1a_J1,AR3a_W,AR1d_J,AR1a_J2,AR4a_E,AR1a_C,AR1d_W,AR5a_H,AR6a_J2,AR9b_W1,AR3a_J2,AR7a_J2,AR3b_J1,AR1b_W,AR5a_C,AR8a_E1,AR2a_S,AR4a_W,AR5a_S,AR9c_J3,AR3b_L,AR8a_J1,AR5b_J2,AR1a_H,AR8a_W1,AR1c_E,AR1d_L,AR9a_L1,AR1a_W,AR9a_E,AR3b_J2,AR8a_W2,AR3a_L,AR1a_L,AR4a_J1,AR7a_H,AR4a_S,AR7a_J1,AR9a_W2,AR1c_J1,AR4a_H1,AR3a_C1,AR9b_L2,AR9b_J2,AR1b_S,AR5b_R,AR7a_L1,AR1c_L,AR9b_E,AR3a_H2,AR3a_E,AR8b_E2,AR8b_L2,AR6a_J3,AR9b_L1,AR5a_E,AR1b_L,AR2a_C,AR1a_R,AR9c_W2,AR8a_E2,AR9c_J1,AR6a_R1,AR9a_R1,AR9c_L2,AR9a_R2,AR7a_E,AR8a_L2,AR1d_H,AR2a_E,AR6a_L1,AR9a_W1,AR5b_L,AR1b_R,AR9b_R1,AR9c_E,AR1d_R,AR9c_R1,AR2a_R,AR1b_H,AR1c_R,AR1e_R,AR4a_L,AR5a_W,AR5a_J1,AR4a_H2,AR1c_J2,AR1e_E,AR1b_C,AR5b_E,AR9b_R2,AR1b_J2
DOPPLER_CMD = python CSI_doppler_computation.py $(OUTPUT_DIR) "$(SUBDIRS_DOPPLER)" $(DOPPLER_DIR) 800 800 $(SAMPLE_LENGTH) $(SLIDING) $(NOISE_LEVEL)
CREATE_DATASET_TRAIN_CMD = python CSI_doppler_create_dataset_train.py $(DOPPLER_DIR) $(SUBDIRS_TRAIN) $(SAMPLE_LENGTH) $(SLIDING) $(WINDOWS_LENGTH) $(STRIDE_LENGTHS) $(ACTIVITIES) 4
CREATE_DATASET_TEST_CMD = python CSI_doppler_create_dataset_test.py $(DOPPLER_DIR) $(SUBDIRS_TEST) $(SAMPLE_LENGTH) $(SLIDING) $(WINDOWS_LENGTH) $(STRIDE_LENGTHS) $(ACTIVITIES) 4
TRAIN_CMD = python CSI_network.py $(DOPPLER_DIR) $(SUBDIRS_TRAIN) $(FEATURE_LENGTH) $(WINDOWS_LENGTH) 1 $(BATCH_SIZE) $(NAME_BASE) $(ACTIVITIES)
TEST_CMD = python CSI_network_test.py $(DOPPLER_DIR) $(SUBDIRS_TEST) $(FEATURE_LENGTH) $(WINDOWS_LENGTH) 1 $(BATCH_SIZE) $(NAME_BASE) $(ACTIVITIES)

data_processing:
	$(PREPROCESS_CMD)
	$(H_ESTIMATION_CMD)
	$(RECONSTRUCTION_CMD)
	$(DOPPLER_CMD)

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