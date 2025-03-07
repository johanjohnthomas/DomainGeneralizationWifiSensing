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
BATCH_SIZE = 16  # Number of samples in a batch
NAME_BASE = single_ant.keras   # Name prefix for the files
ACTIVITIES = C,C1,C2,E,E1,E2,H,H1,H2,J,J1,J2,J3,L,L1,L2,L3,R,R1,R2,S,W,W1,W2  # Activities to be considered
NUM_TOT = 4
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
TRAIN_CMD = python CSI_network.py $(DOPPLER_DIR) $(SUBDIRS_TRAIN) $(FEATURE_LENGTH) $(WINDOWS_LENGTH) 1 $(BATCH_SIZE) $(NUM_TOT) $(NAME_BASE) $(ACTIVITIES)
TEST_CMD = python CSI_network_test.py $(DOPPLER_DIR) $(SUBDIRS_TEST) $(FEATURE_LENGTH) $(WINDOWS_LENGTH) 1 $(BATCH_SIZE) $(NUM_TOT) $(NAME_BASE) $(ACTIVITIES)

AR1d=AR1d_C1,AR1d_C2,AR1d_E,AR1d_H,AR1d_J,AR1d_L,AR1d_R,AR1d_S,AR1d_W

AR7a=AR7a_C,AR7a_E,AR7a_H,AR7a_J1,AR7a_J2,AR7a_J3,AR7a_L1,AR7a_L2,AR7a_R,AR7a_S,AR7a_W

AR3a=AR3a_C1,AR3a_C2,AR3a_E,AR3a_H1,AR3a_H2,AR3a_J1,AR3a_J2,AR3a_L,AR3a_R,AR3a_S,AR3a_W
AR4a=AR4a_C1,AR4a_C2,AR4a_E,AR4a_H1,AR4a_H2,AR4a_J1,AR4a_J2,AR4a_L,AR4a_R,AR4a_S,AR4a_W

AR9b=AR9b_E,AR9b_J1,AR9b_J2,AR9b_L1,AR9b_L2,AR9b_L3,AR9b_R1,AR9b_R2,AR9b_W1,AR9b_W2
AR6a=AR6a_E,AR6a_J1,AR6a_J2,AR6a_J3,AR6a_L1,AR6a_L2,AR6a_R1,AR6a_R2,AR6a_W1,AR6a_W2

AR8a=AR8a_E1,AR8a_E2,AR8a_J1,AR8a_J2,AR8a_L1,AR8a_L2,AR8a_L3,AR8a_R1,AR8a_R2,AR8a_W1,AR8a_W2
AR9a=AR9a_E,AR9a_J1,AR9a_J2,AR9a_L1,AR9a_L2,AR9a_R1,AR9a_R2,AR9a_W1,AR9a_W2

AR8b=AR8b_E1,AR8b_E2,AR8b_J1,AR8b_J2,AR8b_L1,AR8b_L2,AR8b_W1,AR8b_W2

AR1a=AR1a_C,AR1a_E,AR1a_H,AR1a_J1,AR1a_J2,AR1a_L,AR1a_R,AR1a_S,AR1a_W

AR2a=AR2a_C,AR2a_E,AR2a_H,AR2a_J1,AR2a_J2,AR2a_L,AR2a_R,AR2a_S,AR2a_W
AR5a=AR5a_C,AR5a_E,AR5a_H,AR5a_J1,AR5a_J2,AR5a_L,AR5a_R,AR5a_S,AR5a_W
AR1c=AR1c_C,AR1c_E,AR1c_H,AR1c_J1,AR1c_J2,AR1c_L,AR1c_R,AR1c_S,AR1c_W
AR1b=AR1b_C,AR1b_E,AR1b_H,AR1b_J1,AR1b_J2,AR1b_L,AR1b_R,AR1b_S,AR1b_W

AR9c=AR9c_E,AR9c_J1,AR9c_J2,AR9c_J3,AR9c_L1,AR9c_L2,AR9c_L3,AR9c_R1,AR9c_R2,AR9c_W1,AR9c_W2

AR1e=AR1e_E,AR1e_J1,AR1e_J2,AR1e_L,AR1e_R,AR1e_W
AR5b=AR5b_E,AR5b_J1,AR5b_J2,AR5b_L,AR5b_R,AR5b_W
AR3b=AR3b_E,AR3b_J1,AR3b_J2,AR3b_L,AR3b_R,AR3b_W


bedroom = $(AR1a),$(AR1b),$(AR1c),$(AR1d),$(AR1e),$(AR2a),$(AR3a),$(AR3b),$(AR4a)
living_room = $(AR5a),$(AR5b)
kitchen = $(AR6a)
laboratory = $(AR7a)
office = $(AR8a),$(AR8b)
semi_anechoic_chamber = $(AR9a),$(AR9b),$(AR9c)
data_processing:
	$(PREPROCESS_CMD)
	$(H_ESTIMATION_CMD)
	$(RECONSTRUCTION_CMD)
	$(DOPPLER_CMD)
run_nn:
	$(TRAIN_CMD)
	$(TEST_CMD)
# Define the targets for each variation
no_unseen_domains:
	$(eval SUBDIRS_TRAIN = AR1a_S,AR1b_E,AR1c_C,AR3a_R,AR4a_R,AR5a_J,AR6a_W,AR7a_J,AR8a_E,AR9a_J,AR9b_W)
	$(eval SUBDIRS_TEST = AR1d_S,AR1e_E,AR2a_J,AR5b_R,AR8b_W,AR9c_J)
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

one_unseen_domain_bedroom_hidden:
	$(eval SUBDIRS_TRAIN = $(living_room),$(kitchen),$(laboratory),$(office),$(semi_anechoic_chamber))
	$(eval SUBDIRS_TEST = $(bedroom))
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

one_unseen_domain_living_room_hidden:
	$(eval SUBDIRS_TRAIN = $(bedroom),$(kitchen),$(laboratory),$(office),$(semi_anechoic_chamber))
	$(eval SUBDIRS_TEST = $(living_room))
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

one_unseen_domain_kitchen_hidden:
	$(eval SUBDIRS_TRAIN = $(bedroom),$(living_room),$(laboratory),$(office),$(semi_anechoic_chamber))
	$(eval SUBDIRS_TEST = $(kitchen))
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

one_unseen_domain_laboratory_hidden:
	$(eval SUBDIRS_TRAIN = $(bedroom),$(kitchen),$(living_room),$(office),$(semi_anechoic_chamber))
	$(eval SUBDIRS_TEST = $(laboratory))
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

one_unseen_domain_office_hidden:
	$(eval SUBDIRS_TRAIN = $(bedroom),$(kitchen),$(laboratory),$(living_room),$(semi_anechoic_chamber))
	$(eval SUBDIRS_TEST = $(office))
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

one_unseen_domain_semi_anechoic_chamber_hidden:
	$(eval SUBDIRS_TRAIN = $(bedroom),$(kitchen),$(laboratory),$(office),$(living_room))
	$(eval SUBDIRS_TEST = $(semi_anechoic_chamber))
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

multiple_unseen_domains_bedroom_and_semi_anechoic_chamber_hidden:
	$(eval SUBDIRS_TRAIN = $(living_room),$(kitchen),$(laboratory),$(office),)
	$(eval SUBDIRS_TEST = $(bedroom),$(semi_anechoic_chamber))
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

multiple_unseen_domains_kitchen_and_laboratory_hidden:
	$(eval SUBDIRS_TRAIN = $(bedroom),$(living_room),$(office),$(semi_anechoic_chamber))
	$(eval SUBDIRS_TEST = $(kitchen),$(laboratory))
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

multiple_unseen_domains_living_room_and_office_hidden:
	$(eval SUBDIRS_TRAIN = $(bedroom),$(kitchen),$(laboratory),$(semi_anechoic_chamber))
	$(eval SUBDIRS_TEST = $(living_room),$(office))
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

multiple_unseen_domains_office_bedroom_and_semi_anechoic_chamber_hidden:
	$(eval SUBDIRS_TRAIN = $(living_room),$(kitchen),$(laboratory))
	$(eval SUBDIRS_TEST = $(office),$(bedroom),$(semi_anechoic_chamber))
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)

multiple_unseen_domains_living_room_kitchen_and_laboratory_hidden:
	$(eval SUBDIRS_TRAIN = $(bedroom),$(office),$(semi_anechoic_chamber))
	$(eval SUBDIRS_TEST = $(living_room),$(kitchen),$(laboratory))
	$(CREATE_DATASET_TRAIN_CMD)
	$(CREATE_DATASET_TEST_CMD)
	$(TRAIN_CMD)
	$(TEST_CMD)


all: data_processing all_tests
all_tests: 
	 no_unseen_domains one_unseen_domain_bedroom_hidden one_unseen_domain_living_room_hidden \
	 one_unseen_domain_kitchen_hidden one_unseen_domain_laboratory_hidden one_unseen_domain_office_hidden \
	 one_unseen_domain_semi_anechoic_chamber_hidden multiple_unseen_domains_bedroom_and_semi_anechoic_chamber_hidden \
	 multiple_unseen_domains_equivalent_data_kitchen_laboratory_hidden \
	 multiple_unseen_domains_equivalent_data_kitchen_laboratory_hidden_test2