import collections
import logging
import os
import pathlib
import re
import string
import sys
import time
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt

import tensorflow_datasets as tfds
import tensorflow_text as text
import tensorflow as tf
import trans_func as trans

logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

data = pd.read_csv("../jena_climate_2009_2016.csv")
data = data[5::6]
date_time = pd.to_datetime(data.pop('Date Time'), format='%d.%m.%Y %H:%M:%S')

wv = data['wv (m/s)']
bad_wv = wv == -9999.0
wv[bad_wv] = 0

max_wv = data['max. wv (m/s)']
bad_max_wv = max_wv == -9999.0
max_wv[bad_max_wv] = 0.0

wv = data.pop('wv (m/s)')
max_wv = data.pop('max. wv (m/s)')

# Convert to radians.
wd_rad = data.pop('wd (deg)') * np.pi / 180

# Calculate the wind x and y components.
data['Wx'] = wv * np.cos(wd_rad)
data['Wy'] = wv * np.sin(wd_rad)

# Calculate the max wind x and y components.
data['max Wx'] = max_wv * np.cos(wd_rad)
data['max Wy'] = max_wv * np.sin(wd_rad)

timestamp_s = date_time.map(pd.Timestamp.timestamp)

day = 24 * 60 * 60
year = (365.2425) * day

data['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
data['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
data['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
data['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

# Split data

n = len(data)
train_df = data[0:int(n*0.7)]
val_df = data[int(n*0.7):int(n*0.9)]
test_df = data[int(n*0.9):]
train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std

#Move data to sets
data_features = train_df.copy()
data_labels = np.array(data_features.pop('T (degC)'))  # remove temperature column and return data to data_labels
data_features = np.array(train_df)


def chunk(data, target, seq_len, pred_len):  # slices 1d array of data points/labels into multiple windows
    buffer_data = []
    buffer_target = []
    # jumping window
    # for i in range(len(data) // seq_len - 1):
    #     buffer_data.append(data[0 + i * seq_len:(i + 1) * seq_len])
    #     buffer_target.append(target[(i + 1) * seq_len:(i + 1) * seq_len + pred_len])
    #sliding window
    for i in range(len(data)-(seq_len+pred_len)):
        buffer_data.append(data[i:i+seq_len])
        buffer_target.append(target[i+seq_len:i+seq_len+pred_len])
    return tf.convert_to_tensor(np.array(buffer_data)), tf.convert_to_tensor(np.array(buffer_target))


# data_features, data_labels = chunk(data_features,data_labels,10,5)

dataset = tf.data.Dataset.from_tensor_slices(chunk(data_features, data_labels, 24, 1)) #<-change desired input and output len
print(dataset)


BUFFER_SIZE = 20000
BATCH_SIZE = 32

def make_batches(ds):
    return (
        ds
            .cache()
            .shuffle(BUFFER_SIZE)
            .batch(BATCH_SIZE)
            .prefetch(tf.data.AUTOTUNE))


train_batches = make_batches(dataset)
print(train_batches)
num_layers = 4
d_model = 128
dff = 512
num_heads = 8
dropout_rate = 0.1

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

learning_rate = CustomSchedule(d_model)

optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
temp_learning_rate_schedule = CustomSchedule(d_model)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')
train_diff = tf.keras.metrics.Mean(name='train_diff')
train_MAE = tf.keras.metrics.MeanAbsoluteError()
transformer = trans.Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    pe_input=1000,
    pe_target=1000,
    rate=dropout_rate)

# for input_example, target_example in dataset.take(1): #32,10,19 -> 32,5
#     input_example = input_example[np.newaxis,:]
#     input = tf.keras.Input(input_example.get_shape().as_list()[1],1) #should have shape (10,1) but works out to (1,10)???
#     output = tf.keras.Input(1,1)
#
#     enc_padding_mask, combined_mask, dec_padding_mask = trans.create_masks(
#         input, output)
#
#     # predictions.shape == (batch_size, seq_len, vocab_size)
#     predictions, attention_weights = transformer.call(input_example,
#                                                  output,
#                                                  False,
#                                                  enc_padding_mask,
#                                                  combined_mask,
#                                                  dec_padding_mask)



checkpoint_path = "./checkpoints_weather/train"

ckpt = tf.train.Checkpoint(transformer=transformer,
                           optimizer=optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!')
EPOCHS = 20
# The @tf.function trace-compiles train_step into a TF graph for faster
# execution. The function specializes to the precise shape of the argument
# tensors. To avoid re-tracing due to the variable sequence lengths or variable
# batch sizes (the last batch is smaller), use input_signature to specify
# more generic shapes.

train_step_signature = [
    tf.TensorSpec(shape=(None, None, 19), dtype=tf.float64),
    tf.TensorSpec(shape=(None, None), dtype=tf.float64),
]

@tf.function(input_signature=train_step_signature)
def train_step(inp, tar):
    tar_inp = tar[:, :-1] #remove last element, meant for prediction
    tar_real = tar[:, 1:] #remove first element, redudant var cuz no [start] token

    enc_padding_mask, combined_mask, dec_padding_mask = trans.create_masks(inp[:,:-1,1], tar)

    with tf.GradientTape() as tape:
        predictions, _ = transformer.call(inp, tar_inp,
                                     True,
                                     enc_padding_mask,
                                     combined_mask,
                                     dec_padding_mask)
        loss = trans.loss_function(tar, predictions)
        diff = trans.diff_function(tar, predictions)

    gradients = tape.gradient(loss, transformer.trainable_variables)
    optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

    train_loss(loss)
    #train_accuracy(trans.accuracy_function(tar_real, predictions))
    train_diff(diff)
    #train_MAE(tar_real,predictions)


for epoch in range(EPOCHS):
    start = time.time()

    train_loss.reset_states()
    train_accuracy.reset_states()
    train_diff.reset_states() #avg diff between pred and real value
    #train_MAE.reset_states()

    # inp -> portuguese, tar -> english
    for (batch, (inp, tar)) in enumerate(train_batches):
        train_step(inp, tar)
        #print("Batch " + str(batch) + " complete")
        if batch % 50 == 0:
            print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f} '
                  f'Avg_diff {train_diff.result():.4f}')

    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

    print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f} Avg_diff {train_diff.result():.4f}')

    print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')

def evaluate(weather,target): #inp should be (1,10,19) #maybe make (1,11,19) take last entry out into output
    encoder_input = weather
    output = encoder_input[:,-1,1,np.newaxis] #temperature of last input
    encoder_input = encoder_input[:,:-1,:] #remove last inp from encoder_input
    pred_length = tf.shape(target)[1]
    for i in range(pred_length):
        enc_padding_mask, combined_mask, dec_padding_mask = trans.create_masks(
            encoder_input[:,:,0], output)

        # predictions.shape == (batch_size, seq_len, vocab_size)
        predictions, attention_weights = transformer(encoder_input,
                                                     output,
                                                     False,
                                                     enc_padding_mask,
                                                     combined_mask,
                                                     dec_padding_mask)

        # select the last temperature from the seq_len dimension
        predictions = predictions[:, -1:, 0]  # (batch_size, 1, 1)

        # concatenate the predicted_id to the output which is given to the decoder
        # as its input.
        output = tf.concat([output, tf.cast(predictions,dtype=tf.double)], axis=-1)
    return output, attention_weights

for input_example, target_example in dataset.take(1):
    input_example = tf.expand_dims(input_example, axis=0)
    target_example = tf.expand_dims(target_example, axis=0)
    #train_step(input_example,target_example)
    print("Input Shape:", tf.shape(input_example))
    print("Input T deg(C) Values:", input_example[:,:,1])
    print("Target Values T deg(C):", target_example)
    #output
#   Input Shape: tf.Tensor([ 1 10 19], shape=(3,), dtype=int32)
#   Input T deg(C) Values: tf.Tensor([[-8.05 -8.88 -8.81 -9.05 -9.63 -9.67 -9.17 -8.1  -7.66 -7.04]], shape=(1, 10), dtype=float64)
#   Target Values T deg(C): tf.Tensor([[-7.41 -6.87 -5.89 -5.94 -5.69]], shape=(1, 5), dtype=float64)
    predicted,_ = evaluate(input_example,target_example)
    print("Predicted:",predicted)
    plt.plot(predicted[0,:])
    plt.plot(target_example[0,:])
    plt.ylabel("Temperature(C)")
    plt.show()
    plt.plot(tf.concat([input_example[0,:,1],predicted[0,1:]],axis=0))
    plt.plot(tf.concat([input_example[0,:,1],target_example[0,:]],axis=0))
    plt.show()

