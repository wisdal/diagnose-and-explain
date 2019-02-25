# Copyright 2018 Wisdom D'Almeida
# Licensed under the Apache License, Version 2.0 (the "License")

import tensorflow as tf
import numpy as np

def gru(units):
    return tf.keras.layers.GRU(units,
                             return_sequences=True,
                             return_state=True,
                             recurrent_activation='sigmoid',
                             recurrent_initializer='glorot_uniform')

class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, key, query):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)
        #print("Key Shape:", key.shape)
        #print("Query Shape:", query.shape)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        #hidden_with_time_axis = tf.expand_dims(hidden, 1)

        #print("W1:", self.W1(key).shape)
        #print("W2:", self.W2(query).shape)
        score = tf.nn.tanh(self.W1(key) + self.W2(query))
        #print("Score:", score.shape)
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights*key
        #print("Context Vector:", context_vector.shape)
        context_vector = tf.reduce_sum(context_vector, axis=1)
        #print("Context Vector Reduced:", context_vector.shape)

        return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    # Since we have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class Sentence_Encoder(tf.keras.Model):
    # Since we have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, units):
        super(Sentence_Encoder, self).__init__()
        self.attention = BahdanauAttention(units)
        self.fc = tf.keras.layers.Dense(units)

    def call(self, hidden_states, features):
        # hidden_states: (batch_size, max_sentence_length, units + units)
        # features: (batch_size, 64, embedding_dim)
        features = tf.expand_dims(features, 1)
        features = tf.reshape(features, (features.shape[0], features.shape[1], -1))
        # context_vector: (batch_size, units + units)
        # word_weights: (batch_size, max_sentence_length)
        context_vector, word_weights = self.attention(hidden_states, features)
        # encoded_sentence: (batch_size, units)
        encoded_sentence = self.fc(context_vector)

        return encoded_sentence, word_weights

class Paragraph_Encoder(tf.keras.Model):
    # Since we have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, units):
        super(Paragraph_Encoder, self).__init__()
        self.attention = BahdanauAttention(units)

    def call(self, encoded_sentences, features):
        # encoded_sentences: (batch_size, MAX_PARAGRAPH_LENGTH, units)
        # features: (batch_size, 64, embedding_dim)
        features = tf.expand_dims(features, 1)
        features = tf.reshape(features, (features.shape[0], features.shape[1], -1))
        # encoded_paragraph: (batch_size, units)
        # sentence_weights: (batch_size, max_paragraph_length)
        encoded_paragraph, sentence_weights = self.attention(encoded_sentences, features)
        return encoded_paragraph, sentence_weights

class Word_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(Word_Decoder, self).__init__()

        self.attention = BahdanauAttention(units)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = gru(units)
        self.fc1 = tf.keras.layers.Dense(units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

    def call(self, x, features, prev_sentence, hidden):
        # x: (batch_size, 1)
        # features: (batch_size, 64, embedding_dim)
        # prev_sentence: (batch_size, units)
        # hidden: (batch_size, units)

        #print("X shape:", x.shape)
        # visual_context: (batch_size, embedding)
        # visual_weights: (batch_size, 64)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        visual_context, visual_weights = self.attention(features, hidden_with_time_axis)

        #print("X before embedding:", x.shape)
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)
        #print("X after embedding:", x.shape)

        # x shape after concatenation == (batch_size, 1, embedding_dim + embedding_dim + units)
        x = tf.concat([tf.expand_dims(visual_context, 1), tf.expand_dims(prev_sentence, 1), x], axis=-1)
        #print("X after concatenation", x.shape)

        # passing the concatenated vector to the GRU
        # output: batch_size, 1, units
        output, state = self.gru(x)
        #print("Gru Output:", output.shape)

        # shape == (batch_size, 1, units)
        x = self.fc1(output)
        #print("FC1 Output:", x.shape)

        # x shape == (batch_size * 1, units)
        x = tf.reshape(x, (-1, x.shape[2]))
        #print("X reshaped:", x.shape)

        # output shape == (batch_size * 1, vocab_size)
        x = self.fc2(x)
        #print("Final X:", x.shape)

        return x, state, visual_weights

class Trainer():
    def __init__(self, tokenizer, embedding_dim, units):
        self.tokenizer = tokenizer
        self.units = units

        self.image_encoder = CNN_Encoder(embedding_dim)
        self.sentence_encoder = Sentence_Encoder(units)
        self.paragraph_encoder = Paragraph_Encoder(units)
        self.fwd_decoder = Word_Decoder(embedding_dim, units, len(tokenizer.word_index))
        self.bwd_decoder = Word_Decoder(embedding_dim, units, len(tokenizer.word_index))

    def loss_function(real, pred):
        mask = 1 - np.equal(real, 0)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_)

    def tensors_are_same(a, b):
        r = str(tf.reduce_all(tf.equal(a, b))) # In a perfect world, I would just compare tf.reduce_all(tf.equal(a, b)).numpy()
        return r[10] == 'T'

    def train_word_decoder(self, batch_size, loss, features, findings, i, prev_sentence):
        fwd_hidden = tf.zeros((batch_size, self.units))
        bwd_hidden = tf.zeros((batch_size, self.units))
        fwd_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * batch_size, 1)
        bwd_input = tf.expand_dims([self.tokenizer.word_index['<pad>']] * batch_size, 1)
        hidden_states = tf.zeros((batch_size, 1, self.units + self.units)) # concatenated fwd and bwd hidden states

        for j in range(findings.shape[2]): # generate each word (each sentence has a fixed # of words)
        #print("j", j)
            predictions, fwd_hidden, _ = self.fwd_decoder(fwd_input, features, prev_sentence, fwd_hidden)
            loss += self.loss_function(findings[:, i, j], predictions)
            fwd_input = tf.expand_dims(findings[:, i, j], 1)

            predictions, bwd_hidden, _ = self.bwd_decoder(bwd_input, features, prev_sentence, bwd_hidden)
            loss += self.loss_function(findings[:, i, -(j+1)], predictions)
            bwd_input = tf.expand_dims(findings[:, i, -(j+1)], 1)

            # Concat the bwd anf fwd hidden states
            # batch_size, 1, units + units
            hidden = tf.concat([tf.expand_dims(fwd_hidden, 1), tf.expand_dims(bwd_hidden, 1)], axis=-1)
            if self.tensors_are_same(hidden_states, tf.zeros((batch_size, 1, self.units + self.units))) is True:
              hidden_states = hidden
            else:
              hidden_states = tf.concat([hidden_states, hidden], axis = 1)

        prev_sentence, _ = self.sentence_encoder(hidden_states, features)
        print(hidden_states.shape, prev_sentence.shape)
        return loss, prev_sentence

    """ Training

    * We extract the features stored in the respective `.npy` files and then pass those features through the encoder.
    * The encoder output, hidden state(initialized to 0) and the decoder input (which is the start token) is passed to the decoder.
    * The decoder returns the predictions and the decoder hidden state.
    * The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
    * Use teacher forcing to decide the next input to the decoder.
    * Teacher forcing is the technique where the target word is passed as the next input to the decoder.
    * The final step is to calculate the gradients and apply it to the optimizer and backpropagate.
    """
    def train_fn(self, batch_size, img_tensor, findings):
        loss = 0
        with tf.GradientTape() as tape:
            features = self.image_encoder(img_tensor)
            encoded_sentences = tf.zeros((batch_size, 1, self.units))
            prev_sentence = tf.zeros((batch_size, self.units))

            # Generate Findings
            for i in range(int(findings.shape[1]/2)): # for each sentence in "findings" (each batch has a fixed # of sentences)
                print("-------------------------------------i:", i)
                loss, prev_sentence = self.train_word_decoder(batch_size, loss, features, findings, i, prev_sentence)
                if self.tensors_are_same(encoded_sentences, tf.zeros((batch_size, 1, self.units))) is True:
                    encoded_sentences = tf.expand_dims(prev_sentence, 1)
                else:
                    encoded_sentences = tf.concat([encoded_sentences, tf.expand_dims(prev_sentence, 1)], axis = 1)

            encoded_paragraph, _ = self.paragraph_encoder(encoded_sentences, features)

            # Generate Impressions
            prev_sentence = encoded_paragraph
            for i in range(int(findings.shape[1]/2), findings.shape[1]): # for each sentence in "impressions" (each batch has a fixed # of sentences)
                print("-------------------------------------i:", i)
                loss, _ = self.train_word_decoder(batch_size, loss, features, findings, i, prev_sentence)

            # Outside of "With tf.GradientTape()"
            variables = self.image_encoder.variables + self.sentence_encoder.variables + self.paragraph_encoder.variables + \
            self.fwd_decoder.variables + self.bwd_decoder.variables

        gradients = tape.gradient(loss, variables)
        return loss, gradients, variables
        #optimizer.apply_gradients(zip(gradients, variables), tf.train.get_or_create_global_step())