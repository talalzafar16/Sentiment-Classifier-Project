import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import json

# Importing Data
df = pd.read_json('./data/Sarcasm_Headlines_Dataset.json', lines=True)
df_v2 = pd.read_json('./data/Sarcasm_Headlines_Dataset_v2.json', lines=True)

# Splitting Data sets
testing_sentences = df['headline'].tolist()
training_sentences = df_v2['headline'].tolist()
testing_labels = df['is_sarcastic'].tolist()
training_labels = df_v2['is_sarcastic'].tolist()

# Tokenizer
tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index

# Save the tokenizer's word index
with open("./modalConfig/tokenizer_word_index.json", 'w') as f:
    json.dump(word_index, f)

# Calculate sentence lengths for both training and testing data
training_sentence_lengths = [len(sentence.split()) for sentence in training_sentences]
testing_sentence_lengths = [len(sentence.split()) for sentence in testing_sentences]

# Combine and calculate max length based on both datasets
all_sentence_lengths = training_sentence_lengths + testing_sentence_lengths
max_length = max(max(training_sentence_lengths),max(testing_sentence_lengths))

# Save the max sequence length
with open("./modalConfig/max_sequence_length.json", 'w') as f:
    json.dump({"max_sequence_length": max_length}, f)

# Sequencing and Padding on training data
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding='post')

# Sequencing and Padding on testing data
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding='post')

# Convert to NumPy arrays
training_padded = np.array(training_padded)
testing_padded = np.array(testing_padded)
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels)

# Model
vocab_size =len(word_index) + 1 # Ensure vocab_size doesn't exceed num_words
print(vocab_size,"lol")
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),  # Dropout layer to reduce overfitting
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Callbacks
# callbacks = [
#     tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
#     tf.keras.callbacks.ModelCheckpoint('./modalConfig/sarcasm_model_best.keras', save_best_only=True)
# ]

# Training model
num_epochs = 30
history = model.fit(
    training_padded, training_labels,
    epochs=num_epochs,
    validation_data=(testing_padded, testing_labels),
    verbose=2,
    # callbacks=callbacks
)

# Save the final model
model.save('./modalConfig/sarcasm_model.keras')
