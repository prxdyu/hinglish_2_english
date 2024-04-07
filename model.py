# importing the required libraries
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os
import re
import string
import tensorflow_datasets as tfds
import time
from utils import *



# =====================================================================================================================================================#
 
""" POSITIONAL ENCODING """


# definfing the function to create angle_matrix which contains angles for sin and cos
def get_angles(pos,i,d_model):
  """
  pos     : column vector (pos,1) having position values 0 to pos-1
  i       : row vector (1,d_model) having values from 0 to d_models-1
  d_model : embedding_dim

  returns
  angle_matrix: a matrix of dim (pos,d_model)

  """
  angles= 1/ np.power (10000,(2*(i//np.float32(d_model))))
  # assume pos=5 and d_model=512 then (5,1)*(1,512) => (5,512) dimensions of angles will get broadcasted to match the dim of pos
  angle_matrix=pos*angles
  return angle_matrix


def positional_encodings(pos,d_model):
  # creating a column vector (pos,1) from 0 to pos-1
  pos=np.arange(pos)[:,np.newaxis]
  # creating a row vector (1,d_model) from 0 to d_model-1
  i=np.arange(d_model)[np.newaxis,:]
  # passing the two vectors and d_model scalar to the get_angle fuction which returns a 2d matrix of dim (pos,d_model)
  angle_matrix = get_angles(pos,i,d_model)
  # applying sin function to even col indices in the matrix
  angle_matrix[:,0::2]=np.sin(angle_matrix[:,0::2])
  # applying the cos function to odd col indices in the matrix
  angle_matrix[:,1::2]=np.cos(angle_matrix[:,1::2])
  # after we apply sin and cos to the angle matrix it becomes positional encodings and we convert this matrix into a tensor
  pos_encodings=angle_matrix[np.newaxis,...]
  pos_encodings = tf.convert_to_tensor(pos_encodings, dtype=tf.float32)
  return pos_encodings

#=======================================================================================================================================================================#


""" PADDING  """

# defining a function which creates padding mask ie it returns a binary vecotor where 1 represent corresponding token is a padding token ie "0"
def create_padding_mask(seq):
  mask=tf.cast(tf.math.equal(seq,0),tf.float32)
  # making this as a tensor
  mask=mask[:,tf.newaxis,tf.newaxis,:]
  return mask

# defining a function for look ahead mask
def create_look_ahead_mask(size):
  mask=1-tf.linalg.band_part(tf.ones((size,size)),-1,0)
  return mask

# defining functions which creates aboves two masks
def create_masks(inp,target):
  # creating encoder padding mask
  enc_padding_mask= create_padding_mask(inp)
  # creating decoder padding mask which will be used in the 2nd attention in the decoder layer
  dec_padding_mask= create_padding_mask(inp)
  # creating look_ahead_mask for masking future tokens in the decoder layer which will be used in the 1st attention in the decoder
  look_ahead_mask= create_look_ahead_mask(tf.shape(target)[1])
  # creating the padding mask for decoder
  dec_target_padding_mask= create_padding_mask(target)
  # creating a combined mask
  combined_mask= tf.maximum(dec_target_padding_mask,look_ahead_mask)
  return enc_padding_mask,combined_mask,dec_padding_mask



#=============================================================================================================================================================================================#

""" ATTENTION """


def scaled_dot_product_attention(k,v,q,mask):
  """
  This is a self attention so q,k and v is build from the datamatrix
  q: data tensor after passing through by linear layer Wq , (batch_size,n_heads,seq_len,depth)
  k: data tensor after passing through by linear layer Wk , (batch_size,n_heads,seq_len,depth)
  v: data tensor after passing through by linear layer Wv , (batch_size,n_heads,seq_len,depth)
  k_transpose   : (batch_size,n_heads,depth_seq_len)
  q.k_transpose : (batch_size,n_heads,seq_len,depth) * (batch_size,n_heads,depth,seq_len) ==> (batch_size,n_heads,seq_len,seq_len)
  """
  # matrix multiplication of Q and K.transpose
  matmul_qk=tf.matmul(q,k,transpose_b=True) # now this tensor has logits
  # computing dk(embedding_dim) and casting it to float
  dk=tf.cast(tf.shape(k)[-1],tf.float32)
  # scaling the logits in the matmul_qk using dk
  scaled_logits=matmul_qk/tf.math.sqrt(dk)
  # since padding tokens contribute nothing to the attention we ignore them padding by adding a large negative numbers to the logits of the padding tokens
  if mask is not None:
    scaled_logits+=(mask * -1e9)
  # applying the softmax function
  attention_weights=tf.nn.softmax(scaled_logits,axis=-1)
  # multiplying the attention weights with the v (batch_size,n_heads,seq_len,seq_len)*(batch_size,n_heads,seq_len,depth)
  output=tf.matmul(attention_weights,v)
  """output            : (batch_size,n_heads,seq_len,depth)
     attention_weights : (batch_size,n_heads,seq_len,seq_len) """
  return output,attention_weights



# implementing the Multi Head attention layer
class MultiHeadAttention(tf.keras.layers.Layer):

  def __init__(self,d_model,n_heads):
    """
    d_model: embedding_dim or no of hidden_units
    n_heads: no of heads of self attention

    """
    super(MultiHeadAttention,self).__init__()
    self.n_heads=n_heads
    self.d_model=d_model
    # asserting if the d_model is divisible by the number of heads
    assert d_model % self.n_heads == 0
    # depth is the split of d_model for each head
    self.depth = self.d_model // self.n_heads
    # defining the linear layers for weight matrices Wk,Wq,Wv
    self.wk=tf.keras.layers.Dense(d_model)
    self.wq=tf.keras.layers.Dense(d_model)
    self.wv=tf.keras.layers.Dense(d_model)
    # defining the linear layer for the last weight matrix W0
    self.w0=tf.keras.layers.Dense(d_model)


  def split_heads(self,x,batch_size):
    """
    x: tensor of dim (batch_size,seq_len,embedding)
    splits the tensors along last (embeddings) dim to pass that slice for each head
    """
    # splitting the last dimension of x into (n_heads,depth)
    x=tf.reshape(x,(batch_size,-1,self.n_heads,self.depth))
    # after reshaping the shape of x is (batch_size,seq_len,n_heads,depth) but we want to change the dim order to (batch_size,n_heads,seq_len,depth) so permuting it
    x=tf.transpose(x,perm=[0,2,1,3])
    return x


  def call(self,k,v,q,mask):
    """
    k: tensor of shape (batch_size,seq_len,embedding)
    q: tensor of shape (batch_size,seq_len,embedding)
    v: tensor of shape (batch_size,seq_len,embedding)
    """
    # getting the batch size
    batch_size=tf.shape(q)[0]
    # passing the k,v,q to the linear layers wk,wv,wq respectively
    k=self.wk(k)  # (batch_size, seq_len, d_model)
    v=self.wv(v)  # (batch_size, seq_len, d_model)
    q=self.wq(q)  # (batch_size, seq_len, d_model)
    # splitting the last dimension of k,v,q
    k=self.split_heads(k,batch_size)  # (batch_size, n_heads, seq_len, depth)
    v=self.split_heads(v,batch_size)  # (batch_size, n_heads, seq_len, depth)
    q=self.split_heads(q,batch_size)  # (batch_size, n_heads, seq_len, depth)
    # passing the k,v,q to the scaled attention function to compute attention weights
    scaled_attention, attention_weights = scaled_dot_product_attention(k,v,q,mask)
    """
    shape of scaled_attention : (batch_size,n_heads,seq_len,depth)
    shape of attention_weights: (batch_size,n_heads,seq_len,seq_len)
    """
    # transposing the scaled attention to make it prepare for reshaping
    scaled_attention= tf.transpose(scaled_attention,perm=[0,2,1,3]) # (batch_size, seq_len, n_heads, depth)
    # reshaping the scaled_attention back to the (batch_size,seq_len,d_model) d_model=n_heads*depth
    concat_attention= tf.reshape(scaled_attention, (batch_size,-1,self.d_model))
    # passing the concat_attention to the final linear layer which consists of w0
    output=self.w0(concat_attention) # (batch_size, seq_len, d_model)
    return output,attention_weights


#=============================================================================================================================================================================================#


def point_wise_fc_layer(d_model,dff):
  """
  d_model: dimensionality of the input
  dff    : hidden_units in the dense_layer
  it returns a model consisting 2 FC layers. 1st layers reduces the dimensionality of input to dff, the 2nd layer converts the dimensionality of input back to d_model
  """
  return tf.keras.Sequential([tf.keras.layers.Dense(dff,activation='relu'), # (batch_size,seq_len,dff)
                              tf.keras.layers.Dense(d_model)])              # (batch_size,seq_len,d_model)



# =======================================================================================================================================================================================#

""" ENCODER """

class EncoderLayer(tf.keras.layers.Layer):

  def __init__(self,d_model,num_heads,dff,rate=0.1):
    super(EncoderLayer,self).__init__()
    # defining the multi head attention layer
    self.mha=MultiHeadAttention(d_model,num_heads)
    # defining the point wise feed forward layer
    self.fc=point_wise_fc_layer(d_model,dff)
    # defining the layer norm 1 and 2
    self.layer_norm1=tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layer_norm2=tf.keras.layers.LayerNormalization(epsilon=1e-6)
    # defining the dropouts
    self.dropout1=tf.keras.layers.Dropout(rate)
    self.dropout2=tf.keras.layers.Dropout(rate)

  def call(self,x,training,mask):
    # since it is a self attention we pass the same input x as key,query,value to the multi-head-attention
    attn_output,_= self.mha(k=x,v=x,q=x,mask=mask) # (batch_size,seq_len,d_model)
    # passing the attn_output through the dropout
    attn_output= self.dropout1(attn_output,training=training)
    # passing the output throught the layer norm
    out1= self.layer_norm1(x+attn_output)
    # passing the output from the layer norm to the FC layer
    fc_output=self.fc(out1)
    # passing throught the dropout
    fc_output=self.dropout2(fc_output)
    # passing the ouput to the layer norm2
    out2=self.layer_norm2(fc_output) # (batch_size, seq_len, d_model)
    return out2




class Encoder(tf.keras.layers.Layer):

  def __init__(self,num_layers,d_model,num_heads,dff,input_vocab_size,max_pos_encoding,rate=0.1):
    """
    num_layers        : no of encoder layers
    d_model           : dimensionality of input embeddings
    num_heads         : no of heads in multi-head attention
    input_vocab_size  : no of words in input language vocab
    max_pos_encoding  :
    dff               :
    max_pos_encoding  :
    """
    super(Encoder,self).__init__()
    self.d_model=d_model
    self.num_heads=num_heads
    self.num_layers=num_layers
    # defining the embedding layer
    self.embeddings=tf.keras.layers.Embedding(input_vocab_size,d_model)
    # defining the positional encoding layer
    self.pos_encodings=positional_encodings(max_pos_encoding,self.d_model)
    # defining the encoding layers
    self.enc_layers=[EncoderLayer(d_model,num_heads,dff,rate) for _ in range(num_layers)]
    self.dropout=tf.keras.layers.Dropout(rate)


  def call(self,x,training,mask):
    """
    x        : input tensor of shape (batch_size, seq_len)
    training : bool variable to indicate whether we are in training
    mask     : masks for padding

    """
    # getitng the seq_len of the inputs
    seq_len=tf.cast(tf.shape(x)[1], tf.int32)
    # passing the input throught the emebdding layer
    x=self.embeddings(x)  # x:(batch_size, seq_len, d_model)
    # normalization
    x*=tf.math.sqrt(tf.cast(self.d_model,tf.float32))
    # passing the embedding to the positional encodng
    x+=self.pos_encodings[:,:seq_len,:]
    # adding dropout
    x=self.dropout(x,training=training)
    # adding encoder layers
    for i in range(self.num_layers):
      x=self.enc_layers[i](x, training, mask)
    return x
  


#================================================================================================================================#

"""  DECODER  """

class DecoderLayer(tf.keras.layers.Layer):

  def __init__(self,d_model,num_heads,dff,rate=0.1):
    super(DecoderLayer,self).__init__()
    # defining the multi head attention layer1
    self.mha1=MultiHeadAttention(d_model,num_heads)
    # definfing the multi head attention layer2 (encoder-decoder attention)
    self.mha2=MultiHeadAttention(d_model,num_heads)
    # defining the point wise feed forward layer
    self.fc=point_wise_fc_layer(d_model,dff)
    # defining the layer norm 1,2 and 3
    self.layer_norm1=tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layer_norm2=tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layer_norm3=tf.keras.layers.LayerNormalization(epsilon=1e-6)
    # defining the dropouts
    self.dropout1=tf.keras.layers.Dropout(rate)
    self.dropout2=tf.keras.layers.Dropout(rate)
    self.dropout3=tf.keras.layers.Dropout(rate)


  def call(self,x,enc_output,training,look_ahead_mask,padding_mask):
    """
    x          : input at the current timestep t
    enc_output : output of the encoder with shape (batch_size, seq_len, d_model)
    """
    # passing the target as the inputs to the multi-head-attention
    attn1,attn_weights1,=self.mha1(x,x,x,look_ahead_mask)
    # passing throught dropouts
    attn1=self.dropout1(attn1,training=training)
    # passing through layer norm
    out1=self.layer_norm1(attn1 + x)
    # passing the encoders output as v and k and passing out1 as q to the multi-head-attention(encoder_decoder_attention)
    attn2,attn_weights2=self.mha2(enc_output,enc_output,out1,padding_mask)
    # passing through dropout
    attn2=self.dropout2(attn2,training=training)
    # passing through selfnorm
    out2=self.layer_norm2(attn2 + out1)
    # passing through the fc layer
    fc_output=self.fc(out2)
    # passing through dropout
    fc_output=self.dropout3(fc_output)
    # passing through layer norm
    out3=self.layer_norm3(fc_output+out2)
    return out3,attn_weights1,attn_weights2
  


  
class Decoder(tf.keras.layers.Layer):
  
  def __init__(self,num_layers,d_model,num_heads,dff,target_vocab_size,max_pos_encoding,rate=0.1):
    """
    num_layers        : number of decoder layers
    d_model           : dimensionality of target embedding
    num_heads         : number of heads in decoder multi-head attention
    dff               : hidden units in the decoder fc layer
    target_vocab_size : vocab size of target
    """
    super(Decoder,self).__init__()
    self.d_model=d_model
    self.num_layers=num_layers
    # defining the embedding layer in the decoder
    self.embeddings=tf.keras.layers.Embedding(target_vocab_size,d_model)
    # defining the positional encoding layer
    self.pos_encodings=positional_encodings(max_pos_encoding,self.d_model)
    # defining the encoding layers
    self.dec_layers=[DecoderLayer(d_model,num_heads,dff,rate) for _ in range(num_layers)]
    self.dropout=tf.keras.layers.Dropout(rate)


  def call(self,x,enc_output,training,look_ahead_mask,padding_mask):
    """
    x          : input tensor of target words with shape (batch_size,seq_len)
    enc_output : output from the encoding layer with shape (batch_size, seq_len, d_model)
    training   : boolean variable to indicate whether we are training

    """
    # getting the seq_len of the target inputs
    seq_len=tf.cast(tf.shape(x)[1],tf.float32)
    attn_weights={}
    # passing the input target words to the embedding layer
    x=self.embeddings(x)  # x: (batch_size, seq_len, d_model)
    # normalization
    x*=tf.math.sqrt(tf.cast(self.d_model,tf.float32))
    # positional encodings
    x += self.pos_encodings[:, :tf.cast(seq_len, tf.int32), :]
    # adding dropouts
    x=self.dropout(x,training=training)
    # passing through the decoder layers
    for i in range(self.num_layers):
      x,block1,block2=self.dec_layers[i](x,enc_output,training,look_ahead_mask,padding_mask)
      attn_weights[f'decoder_layer{i+1}_block1']=block1
      attn_weights[f'decoder_layer{i+1}_block2']=block2
    return x,attn_weights # x: (batch_size,seq_len,d_model)
  


#===========================================================================================================#

"""  DECODER """

class Transformer(tf.keras.Model):

  def __init__(self,num_layers,d_model,num_heads,dff,input_vocab_size,target_vocab_size,pe_input,pe_target,rate=0.1):
    """
    num_layers        : number of encoder/decoder layers
    d_model           : dimensionality of target embedding
    num_heads         : number of heads in decoder multi-head attention
    dff               : hidden units in the decoder fc layer
    input_vocab_size  : no of words in input language vocab
    target_vocab_size : vocab size of target
    pe_iput           : positional encodings of input embeddings
    pe_output         : positional encodings of target embeddings

    """
    super(Transformer,self).__init__()
    # defining the encoder
    self.encoder= Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
    # defining the decoder
    self.decoder= Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
    # defining the final classfier layer
    self.final_layer=tf.keras.layers.Dense(target_vocab_size)


  def call(self,inp,target,training,enc_padding_mask,look_ahead_mask,dec_padding_mask):
    """
    inp    : input tesor of shape (batch_size, input_seq_len)
    target : target tensor of shape (batch_size, target_seq_len)

    """
    # passing the input to the encoder
    enc_output= self.encoder(inp,training,enc_padding_mask)  #(batch_size,input_seq_len,d_model)
    # passing the target to the decoder
    dec_output,attention_weights=self.decoder(target,enc_output,training,look_ahead_mask,dec_padding_mask)
    # passing the decoder output to the final layer
    final_output=self.final_layer(dec_output)  # (batch_size,target_seq_len,target_vocab_size)
    return final_output,attention_weights
  

#=============================================================================================================================================================================================#

"""  CUSTOM SCHEDULER  """

# creating a custom scheduler which changes the learning rate according the formula in the research paper
class CustomScheduler(tf.keras.optimizers.schedules.LearningRateSchedule):

  def __init__(self,d_model,warmup_steps=4000):
    self.d_model=d_model
    self.d_model=tf.cast(self.d_model,tf.float32)
    self.warmup_steps=warmup_steps

  def __call__(self,step):
    step = tf.cast(step, tf.float32)
    arg1= tf.math.rsqrt(step)
    arg2= step * (self.warmup_steps ** -1.5)
    return tf.math.rsqrt(self.d_model)*tf.math.minimum(arg1,arg2)
  

#=============================================================================================================================================================================================#

""" LOSS FUNCTION """

# creating the loss object
loss_obj=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,reduction='none')

# creating a loss function
def loss_function(real,pred):
  # creating a mask because we need to ignore padding tokens while calculating the loss
  mask=tf.math.logical_not(tf.math.equal(real,0)) # this is a tensor which contains 0 if the corresponding token is a padding token ie 0 else 1
  # creating the loss obj
  loss_=loss_obj(real,pred)
  # casting the mask
  mask=tf.cast(mask,dtype=loss_.dtype)
  # we need to multiply the loss with the mask to ignore the losses corresponding to the token
  loss_*=mask
  # calculating the avg loss
  total_loss= tf.reduce_sum(loss_) # total loss for the non padding tokens
  non_padding_tokens= tf.reduce_sum(mask) # total number of non padding tokens
  avg_loss= total_loss/non_padding_tokens
  return avg_loss
  

#=============================================================================================================================================================================================#



""" ACCURACY FUNCTION """

def accuracy_func(real, pred):
    """
    real : ground truth matrix of shape (batch_size, seq_len)
    pred : predicted tensor of shape (batch_size, seq_len, target_voc_size)
    """
    # computing the correctly predicted words
    correctly_predicted = tf.equal(real, tf.argmax(pred, axis=2))
    # creating a mask for padding tokens in the real
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    # calculating the correctly predicted words excluding the padding tokens
    accuracies = tf.math.logical_and(correctly_predicted, mask)
    # calculating the accuracy
    accuracy = tf.reduce_sum(tf.cast(accuracies, dtype=tf.float32)) / tf.reduce_sum(tf.cast(mask, dtype=tf.float32))
    return accuracy


#=============================================================================================================================================================================================#

""" CUSTOM TRAINING FUNCTION """

# creating metrics for train
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]

# wrapping the training function using tf.function wrapper to compile the steps into a TF Graph for faster execution
@tf.function(input_signature=train_step_signature)
def train_step(inp,target,transformer,optimizer):
  """
  inp         : input tensor of shape (batch_size, inp_seq_len)
  target      : target tensor of shape (batch_size, target_seq_len)
  transformer : transformer object
  optimizer   : adam optimizer object 

  """
  # slicing the target by excluding the last token of all sequences in the batch to pass it to the decoder as input
  tar_inp= target[:,:-1]
  # slicing the target by excluding the first token of all sequences in the batch to pass it as the label for the decoder to compute loss
  tar_real= target[:,1:]
  # creating masks
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  # Gradient Calculation
  with tf.GradientTape() as tape:
    # making predictions
    predictions, _ = transformer(inp,
                                 tar_inp,
                                 True,
                                 enc_padding_mask,
                                 combined_mask,
                                 dec_padding_mask)
    # computing the loss
    loss = loss_function(tar_real,predictions)
    # computing the gradients
    gradients = tape.gradient(loss, transformer.trainable_variables)
    # updating the params by applying the gradients
    optimizer.apply_gradients(zip(gradients,transformer.trainable_variables))
    # making the loss and accuracies as the metrics
    train_loss(loss)
    train_accuracy(accuracy_func(tar_real,predictions))


#=============================================================================================================================================================================================#

""" CUSTOM VALIDATION FUNCTION """

# creating metrics for validation
val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.Mean(name='val_accuracy')


val_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


# wrapping the val_step function using tf.function wrapper to compile the steps into a TF Graph for faster execution
@tf.function(input_signature=val_step_signature)
def val_step(inp,target,transformer):
  """
  inp         : input tensor of shape (batch_size, inp_seq_len)
  target      : target tensor of shape (batch_size, target_seq_len)
  transformer : transformer object

  """
  # slicing the target by excluding the last token of all sequences in the batch to pass it to the decoder as input
  tar_inp= target[:,:-1]
  # slicing the target by excluding the first token of all sequences in the batch to pass it as the label for the decoder to compute loss
  tar_real= target[:,1:]
  # creating masks
  enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)
  # making predictions
  predictions, _ = transformer(inp,
                                 tar_inp,
                                 False,
                                 enc_padding_mask,
                                 combined_mask,
                                 dec_padding_mask)
  # computing the loss
  loss = loss_function(tar_real,predictions)
  # computing the accuracy
  accuracy = accuracy_func(tar_real, predictions)
  # making the loss and accuracies as the metrics
  val_loss(loss)
  val_accuracy(accuracy)


#=============================================================================================================================================================================================#



""" MODEL INFERENCE """

# defining a function to evaluate query at runtime
def translate(sentence,hindi_tokenizer,en_tokenizer,transformer,maxlen=10):

  """sentence : hinglish sentence"""

  # getting the index for the <EOS> token
  EOS_INDEX=en_tokenizer.encode('<EOS>')[0]
  # adding <SOS> and <EOS> tokens to the sentence
  sentence= add_special_tokens(sentence)
  target="<SOS>"
  # tokenization (converting the sequence to indices)
  sentence=hindi_tokenizer.encode(sentence)
  target=en_tokenizer.encode(target)
  # converting the sentence to tf tensor
  sentence=tf.convert_to_tensor(sentence)      # (,inp_len)
  # adding a new dim to the sentence tensor
  sentence = tf.expand_dims(sentence, axis=0)
  encoder_input=sentence                       # (1, inp_len)
  # converting the target to a tensor
  target= tf.convert_to_tensor(target)         # (1, )
  # adding a new dimension to the target tensor
  target = tf.expand_dims(target, axis=0)      # (1, 1)

  """
  Initially @ 1st timestep we pass <EOS> token to the transformer,
  it generates maxlen ie 40 words, but we take the last word as the output of the 1st timestep
  now we concatenate this ouput to the <EOS> and pass it to the transformer as the input @ 2nd timestep and
  and again it generates 40 words we take the last word and concatenate it and pass it as the input @ 3rd timestep
  """

  for i in range(maxlen):
    # creating masks
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(encoder_input, target)
    """
    enccoder_input : (1, inp_len)
    target        : (1,1)
    """
    # making predictions of shape (batch_size,target_seq_len,target_vocab_size)
    predictions,attn_weights=transformer(encoder_input,
                                         target,
                                         False,
                                         enc_padding_mask,
                                         combined_mask,
                                         dec_padding_mask)
    # selecting the last word from predictions @ curr timestep
    last_word = predictions[:,-1,:]                         #(batch_size,1,target_vocab_size)
    # computing the index of the highest probability and last_word is a eager tensor containing the index
    last_word = tf.argmax(last_word,axis=-1)
    # converting the eager tensor to a int for if condition
    last_word_idx = last_word.numpy().item()
    # casting the last_word to int32
    last_word_id  = last_word
    last_word_id = tf.cast(last_word_id,tf.int32)
    # reshaping last_word_id to match the dimension of target to concat both of them
    last_word_id = tf.reshape(last_word_id, [1, 1])
    # adding the last predicted word(index) @ curr timestep to the input for the transformer @ next timestep
    target= tf.concat([target,last_word_id],axis=-1)
    # if the last predicted word is <EOS> break the loop and stop generating
    if last_word_idx==EOS_INDEX:
      break
  # decoding the indices in targets to texts
  target=list(target.numpy()[0])
  decoded=en_tokenizer.decode(target)
  # removing the special tokens
  decoded_lst= decoded.split(" ")
  decoded=[i for i in decoded_lst if i not in ('<SOS>','<EOS>') ]
  decoded=" ".join(decoded)

  return decoded


#=============================================================================================================================================================================================#



