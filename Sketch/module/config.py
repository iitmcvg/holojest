

'''
Config params
'''
import os

# main_dir = os.path.join(os.getcwd(),'data_char')
# name_list_path = os.path.join(os.getcwd(),'data_char/list.txt')
# sketch_dir = os.path.join(os.getcwd(),'data_char/sketch')
# dnfs_dir = os.path.join(os.getcwd(),'data_char/dnfs')

home=os.path.expanduser('~')
main_dir=os.path.join(home,'TrainingData','Character')
sketch_dir=os.path.join(main_dir,'sketch')
dnfs_dir=os.path.join(main_dir,'dn')
name_list_path=os.path.join(main_dir,'train-list.txt')
#name_list_path=os.path.join(main_dir,'temp_list.txt')

#is training
is_training=True

# Logging
log_dir=os.path.join(home,'holojest','Sketch','logs')
train_log_dir=os.path.join(log_dir,'train')
if not os.path.exists(train_log_dir):
    os.mkdir(train_log_dir)

eval_log_dir=os.path.join(log_dir,'eval')
if not os.path.exists(eval_log_dir):
    os.mkdir(eval_log_dir)

checkpoints_dir=os.path.join(home,'holojest','Sketch','checkpoints')

is_adversial= True
# Loss tuning
loss_normalize=True
mask_threshold=0.9
lambda_pixel=1
lambda_adv=0.01
# Train configs
training_iter = 4
batch_size = 2
learning_rate = .0001

# Data
prefetch_buffer_size = 2
num_parallel_batches = 2
