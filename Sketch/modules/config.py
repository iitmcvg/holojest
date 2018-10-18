'''
Config params
'''

main_dir = 'data_char'
name_list_path = 'data_char/list.txt'
sketch_dir = 'data_char/sketch'
dnfs_dir = 'data_char/dnfs'

# Loss tuning

# Train configs
training_iter = 2
batch_size = 2
learning_rate = .0001

# Data
prefetch_buffer_size = 64
num_parallel_batches = 8
