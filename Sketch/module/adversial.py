import tensorflow as tf
import modules.config
import numpy as np

import tf.contrib.layers.conv2d as conv2d
import tf.contrib.framework as framework
import tf.contrib.layers.fully_connected as fully_connected
import tf.keras.layers.Dense as dense

def discriminator(images):
    """
    input:[n,h,w,5]
    """
    with tf.name_scope("discriminator"):
        with framework.arg_scope([conv2d],kernel_size=4,strides=2,activation_fn=tf.nn.leaky_relu,
                normalizer_fn=tf.contrib.layers.batch_norm,padding="same"):
            net=conv2d(images,num_outputs=64)
            net=conv2d(net,num_outputs=128)
            net=conv2d(net,num_outputs=256)
            net=conv2d(net,num_outputs=512)
            net=conv2d(net,num_outputs=512)
            net=conv2d(net,num_outputs=512)
            net=conv2d(net,num_outputs=512)
        
        net=fully_connected(net,num_outputs=2048,activation_fn=tf.nn.leaky_relu)
        net=dense(net,units=1,activation=sigmoid)
    
    return net

def get_view_predictions(images):
    """
    images:[n,12,256,256,5] #output of decoder
    
    return 12xnx1
    """
    images=tf.transpose(images,[1,0,2,3,4])
    vp=[]
    for view in range(12):
        prediction=discriminator.discriminator(images[view,:,:,:,:]) #nx1
        vp.append(prediction)
    vp=tf.stack(vp,axis=0)
    return vp
