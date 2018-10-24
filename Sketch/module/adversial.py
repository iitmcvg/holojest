import tensorflow as tf
import module.config
import numpy as np
import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as framework

def discriminate(images):
    """
    input:[n,h,w,5]
    """
    with tf.name_scope("discriminator"):
        with framework.arg_scope([layers.conv2d],kernel_size=4,stride=2,activation_fn=tf.nn.leaky_relu,
                normalizer_fn=tf.contrib.layers.batch_norm,padding="same"):
            net=layers.conv2d(images,num_outputs=64)
            net=layers.conv2d(net,num_outputs=128)
            net=layers.conv2d(net,num_outputs=256)
            net=layers.conv2d(net,num_outputs=512)
            net=layers.conv2d(net,num_outputs=512)
            net=layers.conv2d(net,num_outputs=512)
            net=layers.conv2d(net,num_outputs=512)
        
        net=layers.fully_connected(net,num_outputs=2048,activation_fn=tf.nn.leaky_relu)
        net=layers.Dense(net,units=1,activation=sigmoid)
    
    return net

def get_view_predictions(images):
    """
    images:[n,12,256,256,5] #output of decoder
    
    return 12xnx1
    """
    images=tf.transpose(images,[1,0,2,3,4])
    vp=[]
    for view in range(12):
        prediction=adversial.discriminate(images[view,:,:,:,:]) #nx1
        vp.append(prediction)
    vp=tf.stack(vp,axis=0)
    return vp
