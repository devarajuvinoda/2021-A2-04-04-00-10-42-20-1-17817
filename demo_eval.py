import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from sklearn.model_selection import train_test_split
import ast
import unicodedata
import re
import numpy as np
import os
import io
import time
import pandas as pd
import csv
import sys
import collections, os
import regex as re
from abc import abstractmethod
from tokenizer.c_tokenizer import C_Tokenizer

def normalize(tokenized_program, name_dict):
    cnt = 0
    itr = -1
    build_dict = {}
    for i,j in zip(tokenized_program, name_dict):
        itr += 1
        new_name = ('var_'+str(cnt))
        if((j == 'name' or j == 'number' or j== 'string') and (i in build_dict.keys())):
            tokenized_program[itr] = build_dict[i]
        elif(j == 'name' or j == 'number' or j== 'string'):
            tokenized_program[itr] = new_name
            cnt += 1
            build_dict[i] = new_name
    return tokenized_program 

def normalize_test(tokenized_program, name_dict):
    cnt = 0
    itr = -1
    build_dict = {}
    reverse_dict = {}
    for i,j in zip(tokenized_program, name_dict):
        itr += 1
        new_name = ('var_'+str(cnt))
        if((j == 'name' or j == 'number' or j== 'string') and (i in build_dict.keys())):
            tokenized_program[itr] = build_dict[i]
        elif(j == 'name' or j == 'number' or j== 'string'):
            tokenized_program[itr] = new_name
            cnt += 1
            build_dict[i] = new_name
            reverse_dict[new_name] = i

    return tokenized_program, reverse_dict

def reverse_normalize(token_list, reverse_dict):
    itr = -1
    for i in token_list:
        itr += 1
        if(i in reverse_dict.keys()):
            token_list[itr] = reverse_dict[i]
    return token_list

def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<oov>')
    lang_tokenizer.fit_on_texts(lang)

    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

    return tensor, lang_tokenizer

def shrink_vocabulary(input_tokenizer):
    num_words = 600
    input_tokenizer.word_index = {e:i for e,i in input_tokenizer.word_index.items() if i <= num_words}
    input_tokenizer.word_index[input_tokenizer.oov_token] = num_words 
    return input_tokenizer


def text_to_seq(tokenizer, lang):
    tensor = tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                     padding='post')
    return tensor

def load_dataset(ip_data):
    inp_lang = []
    targ_lang = []
    obj = C_Tokenizer()
    for w,k in zip(ip_data['sourceLineText'], ip_data['targetLineText']):
        w_tok, w_dict = obj.tokenize(w)
        k_tok, k_dict = obj.tokenize(k)
        
        w_tok = normalize(w_tok, w_dict)
        k_tok = normalize(k_tok, k_dict)
        w_tok.append('<end>')
        w_tok.insert(0, '<start>')
        k_tok.append('<end>')
        k_tok.insert(0, '<start>')

        if(len(w_tok)<=50 and len(k_tok)<=50):
            inp_lang.append(w_tok)
            targ_lang.append(k_tok)

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

#     inp_lang_tokenizer = shrink_vocabulary(inp_lang_tokenizer)
#     targ_lang_tokenizer = shrink_vocabulary(targ_lang_tokenizer)
    
    input_tensor = text_to_seq(inp_lang_tokenizer, inp_lang)
    target_tensor = text_to_seq(targ_lang_tokenizer, targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, x, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self):
        return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # we are doing this to broadcast addition along the time axis to calculate the score
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(
            self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc = tf.keras.layers.Dense(vocab_size)

        # used for attention
        self.attention = BahdanauAttention(self.dec_units)

    def call(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)

        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        output = tf.reshape(output, (-1, output.shape[2]))

        x = self.fc(output)

        return x, state, attention_weights


def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

@tf.function
def train_step(inp, targ, enc_hidden):
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_hidden = encoder(inp, enc_hidden)

        dec_hidden = enc_hidden

        dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

        # Teacher forcing - feeding the target as the next input
        for t in range(1, targ.shape[1]):
            # passing enc_output to the decoder
            predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

            loss += loss_function(targ[:, t], predictions)

            # using teacher forcing
            dec_input = tf.expand_dims(targ[:, t], 1)

    batch_loss = (loss / int(targ.shape[1]))

    variables = encoder.trainable_variables + decoder.trainable_variables

    gradients = tape.gradient(loss, variables)

    optimizer.apply_gradients(zip(gradients, variables))

    return batch_loss

def start_training():
    EPOCHS = 60

    for epoch in range(EPOCHS):
        start = time.time()

        enc_hidden = encoder.initialize_hidden_state()
        total_loss = 0

        for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inp, targ, enc_hidden)
            total_loss += batch_loss

            if batch % 100 == 0:
                print(f'Epoch {epoch+1} Batch {batch} Loss {batch_loss.numpy():.4f}')
      # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(f'Epoch {epoch+1} Loss {total_loss/steps_per_epoch:.4f}')
        print(f'Time taken for 1 epoch {time.time()-start:.2f} sec\n')

        
def evaluate(sentence):
    attention_plot = np.zeros((max_length_targ, max_length_inp))
    inputs = [inp_lang.word_index[i] for i in sentence if i in inp_lang.word_index]
    inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
    inputs = tf.convert_to_tensor(inputs)

    result = ''

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

    for t in range(max_length_targ):
        predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                             dec_hidden,
                                                             enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += targ_lang.index_word[predicted_id] + ' '

        if targ_lang.index_word[predicted_id] == '<end>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

def translate(sentence):
    result, sentence, attention_plot = evaluate(sentence)
    return result

def save_result(input_file, output_file):
    df_test = pd.read_csv(input_file)
    out = []
    for w in df_test['sourceLineText']:
        get_token = C_Tokenizer()
        res,som = get_token.tokenize(w)
        res, rev_dict = normalize_test(res, som)
        
        res = translate(w)
        res,som = get_token.tokenize(res[:-6])
        res = reverse_normalize(res, rev_dict)
        out.append(res)
    new_df = df_test.copy();
    new_df['fixedTokens'] = out
    new_df.to_csv(output_file)

if __name__ == '__main__':
    if(len(sys.argv) < 3):
        print('please give input and output csv file name')
    else:
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
        tf.random.set_seed(100)
        
        ip_data = pd.read_csv('train.csv')
        input_tensor, target_tensor, inp_lang, targ_lang = load_dataset(ip_data)
        # Calculate max_length of the target tensors
        max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
        # Creating training and validation sets using an 80-20 split
        input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)

        BUFFER_SIZE = len(input_tensor_train)
        BATCH_SIZE = 64
        steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
        embedding_dim = 256
        units = 1024
        vocab_inp_size = len(inp_lang.word_index)+1
        vocab_tar_size = len(targ_lang.word_index)+1

        dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

        example_input_batch, example_target_batch = next(iter(dataset))

        encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)

        # sample input
        sample_hidden = encoder.initialize_hidden_state()
        sample_output, sample_hidden = encoder(example_input_batch, sample_hidden)

        attention_layer = BahdanauAttention(10)
        attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

        decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
        sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                              sample_hidden, sample_output)
        optimizer = tf.keras.optimizers.Adam()
        loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                                    reduction='none')

        checkpoint_dir = './training_checkpoints_10_may_line_28percent'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(optimizer=optimizer,
                                         encoder=encoder,
                                         decoder=decoder)
        
#         start_training()

        # restoring the latest checkpoint in checkpoint_dir
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

        save_result(input_file, output_file)