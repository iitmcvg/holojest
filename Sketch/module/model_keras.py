import tensorflow as tf 
import module.config

import tf.keras.layers.CONV2D as conv2d
import tf.keras.models.Model as Model
import tf.keras.layers.LeakyReLU as leaky_relu
import tf.keras.layers.BatchNormalization as batch_norm
import tf.keras.layers.concatenate  as concat
import tf.keras.layers.Conv2DTranspose as deconv
import tf.keras.layers.Dropout as dropout


def decoder(images,view=12):
    inputs=tf.keras.layers.Input(shape=(256,256,2))
    with tf.name_scope("encoder"):
        
        net=conv2d(filters=64,kernel_size=4,
                strides=2,padding='same')(inputs) #256,256,64
        net=leaky_relu()(net)
        net=batch_norm(name='e1')(net)
        
        net=conv2d(filters=128,kernel_size=4,
                strides=2,padding='same')(net) #256,256,128
        net=leaky_relu()(net)
        net=batch_norm(name='e2')(net)
        
        net=conv2d(filters=256,kernel_size=4,
                strides=2,padding='same')(net) #256,256,256
        net=leaky_relu()(net)
        net=batch_norm(name='e3')(net)
        
        net=conv2d(filters=512,kernel_size=4,
                strides=2,padding='same')(net) #256,256,512
        net=leaky_relu()(net)
        net=batch_norm(name='e4')(net)
        
        net=conv2d(filters=512,kernel_size=4,
                strides=2,padding='same')(net) #256,256,512
        net=leaky_relu()(net)
        net=batch_norm(name='e5')(net)
        
        net=conv2d(filters=512,kernel_size=4,
                strides=2,padding='same')(net) #256,256,512
        net=leaky_relu()(net)
        net=batch_norm(name='e6')(net)
        
        net=conv2d(filters=512,kernel_size=4,
                strides=2,padding='same')(net) #256,256,512
        net=leaky_relu()(net)
        encoder_out=batch_norm(name='encoded')(net)
        
    encoder=Model(input=inputs,outputs=enoder_out)
    
    view_array=[]
    
    e6=encoder.get_layer('e6').output
    e5=encoder.get_layer('e5').output
    e4=encoder.get_layer('e4').output
    e3=encoder.get_layer('e3').output
    e2=encoder.get_layer('e2').output
    e1=encoder.get_layer('e1').output
    
    for view in range(views):
        with tf.name_scope("decoder_{}".format(view+1)):
            
            d6=deconv(filters=512,kernel_size=4,strides=1)(net)
            d6=leaky_relu()(d6)
            d6=batch_norm()(d6)
            d6=dropout(rate=0.5,)(d6)
        
            d5=concat(inputs=[d6,e6],axis=-1)
            d5=deconv(filters=512,kernel_size=4,strides=1)(d5)
            d5=leaky_relu()(d5)
            d5=batch_norm()(d5)
            d5=dropout(rate=0.5)(d5)
            
            d4=concat(inputs=[d5,e5],axis=-1)
            d4=deconv(filters=512,kernel_size=4,strides=1)(d4)
            d4=leaky_relu()(d4)
            d4=batch_norm()(d4)
            
            d3=concat(inputs=[d4,e4],axis=-1)
            d3=deconv(filters=256,kernel_size=4,strides=1)(d3)
            d3=leaky_relu()(d3)
            d3=batch_norm()(d3)
            
            d2=concat(inputs=[d3,e3],axis=-1)
            d2=deconv(filters=128,kernel_size=4,strides=1)(d2)
            d2=leaky_relu()(d2)
            d2=batch_norm()(d2)
            
            d1=concat(inputs=[d2,e2],axis=-1)
            d1=deconv(filters=64,kernel_size=4,strides=1)(d4)
            d1=leaky_relu()(d4)
            d1=batch_norm()(d4)
            
            decoded=concat(inputs=[d1,e1],axis=-1)
            decoded=deconv(filters=5,kernel_size=4,strides=1,
                    activation=tf.keras.activations.tanh)(decoded)
            decoded=tf.keras.backend.l2_normalize (decoded,axis=[1,2,3])

            view_array.append(decoded)
            
    

            
            
        
        
        
        
