
# Ага, теперь понятно, о чём ты спрашиваешь 🙂

# Если мы используем signature="default" в ELMo v3, тогда токенизация делается внутри самого TFHub-модуля ELMo, полностью скрыто от пользователя:

# Вход: батч предложений (batch,), каждый элемент — строка.

# Внутри ELMo происходит:

# Разбиение предложения на токены (внутренний токенизатор).

# Прохождение токенов через Character-CNN → BiLSTM → Contextual embeddings.

# На выходе:

# "default" → усреднённый 1024-вектор для предложения.

# "elmo" (если использовать signature="tokens") → эмбеддинги для каждого токена.


import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import Layer


# ---- ELMo ----
class ElmoEmbeddingLayer(Layer):
    def __init__(self, trainable=False, **kwargs):
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)
        self.dimensions = 1024
        self.trainable = trainable

    def build(self, input_shape):
        # Загружаем ELMo v3 как KerasLayer
        self.elmo = hub.KerasLayer(
            "https://tfhub.dev/google/elmo/3",
            #input_shape=[],  # Вход - один токен или предложение
            dtype=tf.string,
            trainable=self.trainable,
            #name="{}_module".format(self.name)
            signature="default",
            output_key="elmo",  # pooled embedding
        )
        super(ElmoEmbeddingLayer, self).build(input_shape)


    def call(self, x, mask=None):
        x = tf.squeeze(tf.cast(x, tf.string), axis=1)
        result = self.elmo(x)   # (batch, 1024) or (batch, seq_len, 1024): depend on output_key
        tf.print(">>> result:", result)
        return result


    def compute_mask(self, inputs, mask=None):
        # Все токены включены в маску
        mask_tensor = tf.ones_like(inputs, dtype=tf.bool)
        tf.print(">>> compute_mask:", mask_tensor)
        return mask_tensor


if __name__ == "__main__":

    sentences = [
        "I love machine learning very much.",
        "ELMo embeddings are really useful.",
        "TF2 makes life easier."
    ]

    # Transform list from shape [batch, 1] into tf.Tensor
    sentences_tensor = tf.constant(sentences)[:, tf.newaxis]

    # Create embedding layer
    elmo_layer = ElmoEmbeddingLayer(trainable=False)

    # Get embeddings
    embeddings = elmo_layer(sentences_tensor)

    print("Shape:", embeddings.shape)
    # Shape: [3, 1024]
