
import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import Layer


# ---- ELMo ----
class ElmoEmbeddingLayer(Layer):
    def __init__(self, trainable=True, **kwargs):
        super().__init__(**kwargs)
        self.dimensions = 1024
        self.trainable = trainable
        self.elmo = hub.KerasLayer(
            "https://tfhub.dev/google/elmo/3",
            trainable=self.trainable,
            name=f"{self.name}_elmo"
        )

    def call(self, x):
        x = tf.squeeze(tf.cast(x, tf.string), axis=1)
        return self.elmo(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


if __name__ == "__main__":

    sentences = [
        "I love machine learning.",
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
