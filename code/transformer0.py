import re, os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import LambdaCallback

# print(tf.__version__)   # 确认tf的版本

# ====== Transformer Encoder ======
class TransformerEncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        attn_output = self.mha(x, x, x, attention_mask=mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

# ====== Transformer Decoder ======
class TransformerDecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(TransformerDecoderLayer, self).__init__()
        self.mha1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.mha2 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(dff, activation='relu'),
            tf.keras.layers.Dense(d_model)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        attn1 = self.mha1(x, x, x, attention_mask=look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)
        attn2 = self.mha2(out1, enc_output, enc_output, attention_mask=padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)
        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output, training=training)
        return self.layernorm3(out2 + ffn_output)

# ====== Transformer Model ======
class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target,
                 rate=0.1):
        super(TransformerModel, self).__init__()
        self.encoder_embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.decoder_embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.encoder_layers = [TransformerEncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.decoder_layers = [TransformerDecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs, training=False):
        inp, tar = inputs
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
        enc_output = self.encoder_embedding(inp)
        for i in range(len(self.encoder_layers)):
            enc_output = self.encoder_layers[i](enc_output, training, enc_padding_mask)
        dec_output = self.decoder_embedding(tar)
        for i in range(len(self.decoder_layers)):
            dec_output = self.decoder_layers[i](dec_output, enc_output, training, look_ahead_mask, dec_padding_mask)
        final_output = self.final_layer(dec_output)
        return final_output

    def create_masks(self, inp, tar):
        enc_padding_mask = self.create_padding_mask(inp)
        dec_padding_mask = self.create_padding_mask(inp)
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
        return enc_padding_mask, combined_mask, dec_padding_mask
    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        return seq[:, tf.newaxis, tf.newaxis, :]
    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask

# ====== Data Preprocessing ======
def is_uchar(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    if uchar in ('，', '。', '：', '？', '“', '”', '！', '；', '、', '《', '》', '——'):
        return True
    return False

# ====== Train Set Generation ======
def data_generator(data, batch_size, time_steps):
    num_batches = len(data) // (batch_size * time_steps)
    data = data[:num_batches * batch_size * time_steps]
    data = np.array(data).reshape((batch_size, -1))
    while True:
        for i in range(0, data.shape[1], time_steps):
            x = data[:, i:i + time_steps]
            y = np.roll(x, -1, axis=1)
            yield [x, y], y

# ====== Loss (of Model Training) ======
class LossHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))

# ====== Text Generation (of Model Testing) ======
# 生成100字的下文
def generate_text(model, start_string, num_generate=300):
    input_eval = [char2id[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    decoder_input = tf.expand_dims([char2id['。']], 0)  # 初始decoder input (即上一句话的末尾）

    for _ in range(num_generate):
        predictions = model([input_eval, decoder_input], training=False)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy() # 受限于电脑性能，此处只测试一个样本
        input_eval = tf.concat([input_eval, tf.expand_dims([predicted_id], 0)], axis=-1)    # concatenate
        decoder_input = tf.concat([decoder_input, tf.expand_dims([predicted_id], 0)], axis=-1)
        text_generated.append(id2char[predicted_id])

    return start_string + ''.join(text_generated)

if __name__ == '__main__':
    # ====== Initialize ======
    num_layers = 4
    d_model = 256  # 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    batch_size = 32
    time_steps = 100
    epochs = 50
    learning_rate = 0.05

    # ====== Data Processing ======
    # load data
    with open("E:/Deep_NLP/Homework4/天龙八部.txt", encoding='gbk', errors='ignore') as f:
    # with open("/content/NLP_h4_copy/天龙八部.txt", encoding='gbk', errors='ignore') as f: # when run in colab
        data = f.readlines()
    # filter
    pattern = re.compile(r'\(.*\)')
    data = [pattern.sub('', lines) for lines in data]
    data = [line.replace('……', '。') for line in data if len(line) > 1]
    data = ''.join(data)
    data = [char for char in data if is_uchar(char)]
    data = ''.join(data)

    # ====== Build Dictionary ======
    vocab = list(set(data))
    char2id = {c: i for i, c in enumerate(vocab)}
    id2char = {i: c for i, c in enumerate(vocab)}
    numdata = [char2id[char] for char in data]
    input_vocab_size = len(vocab) + 2
    target_vocab_size = len(vocab) + 2

    # ====== Build Model ======
    model = TransformerModel(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, time_steps,
                             time_steps, dropout_rate)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True))
    history = LossHistory()

    # ====== Train Model ======
    # generate batch data for train
    train_data = data_generator(numdata, batch_size, time_steps)
    model.fit(train_data, epochs=epochs, steps_per_epoch=len(numdata) // (batch_size * time_steps), callbacks=[history])

    # ====== Test Model ======
    # generate text
    print(generate_text(model, start_string="段誉这才明白，乔峰所以详详细细的说这段铁事，旨在叙述风波恶的性格，心想此人面貌丑陋，爱闹喜斗，原来天性却极善良，真是人不可以貌相了；刚才王语嫣关心而失碧双姝相顾微笑，自因朱碧二女熟知风波恶的性情，既知莫名其妙与人斗气者必是此君，而此君又决不会滥杀无辜。"))
    # # plot
    # plt.plot(history.losses)
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('transformer_model Training Loss')
    # plt.show()