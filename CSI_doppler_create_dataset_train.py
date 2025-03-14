def create_windows_antennas(csi_list, labels_list, sample_length, stride_length):
    csi_matrix_stride = []
    labels_stride = []
    for i in range(len(labels_list)):
        csi_i = csi_list[i]  # Shape: (antennas, features, time)
        label_i = labels_list[i]
        len_csi = csi_i.shape[2]
        
        for ii in range(0, len_csi - sample_length + 1, stride_length):
            # Extract window with all antennas
            window = csi_i[:, :, ii:ii + sample_length]  # (antennas, features, time)
            csi_matrix_stride.append(window)
            labels_stride.append(label_i)
    
    return csi_matrix_stride, labels_stride 