import tensorflow as tf
import numpy as np
import os
import time
import sys

import module.data as data
import module.loss as loss
import module.model as model
import module.adversial as adversial
import module.config as config
from module.memory_saving_gradients import gradients

main_dir = config.main_dir
training_iter = config.training_iter
batch_size = config.batch_size
learning_rate = config.learning_rate
name_list_path = config.name_list_path

name_list = data.file_to_list(name_list_path)
source_iterator, target_iterator = data.load_data(name_list)
source = source_iterator.get_next()
target = target_iterator.get_next()

#predictions
pred = model.encoderNdecoder(source)
total_pixel_loss=loss.total_loss(pred,target)

if(config.is_adversial):
    #probabilities
    view_pred=tf.transpose(pred,[1,0,2,3,4])
    view_truth=tf.transpose(target[0],[1,0,2,3,4])# so that views are in the first dimension
    view_truth=tf.reshape(view_truth,[12,-1,256,256,5])
    #[12,?,256,256,5]
    split_preds=tf.split(view_pred,12,axis=0)
    stacked_pred=[tf.squeeze(x,axis=0) for x in split_preds]
    stacked_pred=tf.concat(stacked_pred,axis=0)
    #[?,256,256,5]
    split_truths=tf.split(view_truth,12,axis=0)
    stacked_truth=[tf.squeeze(x,axis=0) for x in split_truths]
    stacked_truth=tf.concat(stacked_truth,axis=0)
    #[?,256,256,5]
    probs_input=tf.concat([stacked_pred,stacked_truth],axis=0)
    print(probs_input.shape)
    probs=adversial.discriminate(probs_input)
    prob_pred,prob_truth=tf.split(probs,2,axis=0)

    loss_gen,loss_adv=loss.get_adversial_loss(prob_pred,prob_truth,total_pixel_loss)

accuracy, _ = tf.metrics.accuracy(labels=target, predictions=pred)

if(config.is_adversial):
    print("Using adversarial network")
    all_variables=tf.trainable_variables()
    generator_vars=[var for var in all_variables if 'coder' in var.name]
    disc_vars=[var for var in all_variables if 'discri' in var.name]
    print(len(disc_vars))
    print(len(generator_vars))

    # Encoder decoder
    optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate)
    grads=gradients(loss_gen,generator_vars, checkpoints= "memory")
    grads_and_vars=list(zip(grads,generator_vars))
    optimizer1 = optimizer1.apply_gradients(grads_and_vars)

    # Discriminator
    grads2=gradients(loss_adv,disc_vars, checkpoints= "memory")
    grads_and_vars2 =list(zip(grads2,disc_vars))
    optimizer2=tf.train.AdamOptimizer(learning_rate=learning_rate)
    optimizer2 = optimizer2.apply_gradients(grads_and_vars2) 
else:
    optimizer1 = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(total_pixel_loss)

init = tf.global_variables_initializer()
linit = tf.local_variables_initializer()

n_batches=name_list.shape[0]//batch_size

with tf.Session() as sess:
    sess.run(init)
    sess.run(linit)
    sess.run(source_iterator.initializer)
    sess.run(target_iterator.initializer)

    for epoch in range(training_iter):
        tic = time.clock()
        print("Starting epoch {}".format(epoch + 1))
        sess.run(source_iterator.initializer)
        sess.run(target_iterator.initializer)

        for batch in range((name_list.shape[0] // batch_size) + 1):
            try:
                sys.stdout.write('\r')
                print("\t {} completed .....".format((batch+1)*100/n_batches),end=' ')
                if(config.is_adversial):
                    opt1 = sess.run(optimizer1)
                    opt2=sess.run(optimizer2)
                    l=sess.run(loss_gen)
                else:
                    opt1 = sess.run(optimizer1)
                    l=sess.run(total_pixel_loss)
                    
                acc = sess.run(accuracy)
            except tf.errors.OutOfRangeError:
                print()
                print("\t Epoch {} summary".format(epoch + 1))
                print("\t loss = {} ".format(l))
                print("\t accuracy = {}".format(acc))
                toc = time.clock()
                print("\t Time taken :{}".format((toc - tic) / 60))
                break
