import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding
import matplotlib.pyplot as plt
import tokenizer
import embeddings

#callbacks to be used during training at the end of each epoch
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                              patience=3, min_delta=0.0001)
#LSTM_model
tf.keras.backend.clear_session()

model = keras.models.Sequential([

    Embedding(embeddings.num_tokens,
              embeddings.embedding_dim,
              embeddings_initializer=keras.initializers.Constant(embeddings.embedding_matrix),
              mask_zero=True, input_shape=[None], trainable=False),
    keras.layers.Bidirectional(keras.layers.LSTM(256, dropout=0.4)),
    keras.layers.Dense(12, activation="softmax")

])

print("MODEL SUMMARY :- ",model.summary())

optimizer = keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
history = model.fit(tokenizer.train_set, tokenizer.train_label,
                     batch_size = 32,
                     steps_per_epoch=len(tokenizer.X_train) // 32,
                     validation_data = (tokenizer.val_set, tokenizer.val_label),
                     validation_steps = len(tokenizer.val_set)//32, epochs=20,
                     callbacks=early_stop)

#Plot loss curve

fig = plt.figure(figsize=(10,10))

# Plot accuracy
plt.subplot(221)
plt.plot(history.history['accuracy'],'bo-', label = "acc")
plt.plot(history.history['val_accuracy'], 'ro-', label = "val_acc")
plt.title("train_accuracy vs val_accuracy")
plt.ylabel("accuracy")
plt.xlabel("epochs")
plt.grid(True)
plt.legend()

# Plot loss function
plt.subplot(222)
plt.plot(history.history['loss'],'bo-', label = "loss")
plt.plot(history.history['val_loss'], 'ro-', label = "val_loss")
plt.title("train_loss vs val_loss")
plt.ylabel("loss")
plt.xlabel("epochs")
plt.grid(True)
plt.legend()



