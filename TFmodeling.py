import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from transformers import TFCamembertModel

INPUT_IDS_KEY = 'input_ids'
ATTENTION_MASK_KEY = 'attention_mask'
OUTPUT_KEY = 'classifier'


class CamemBertMultilabelClassification(tf.keras.Model):
    def __init__(self,
                 nb_class,
                 name="CamemBertMultilabelClassification",
                 **kwargs):
        super(CamemBertMultilabelClassification, self).__init__(name=name, **kwargs)
        self.nb_class = nb_class
        self.l1 = TFCamembertModel.from_pretrained("camembert-base")
        self.pre_classifier = Dense(768)
        self.dropout = Dropout(0.1)
        self.classifier = Dense(self.nb_class)

    def call(self, inputs):
        output_1 = self.l1(input_ids=inputs[0], attention_mask=inputs[1])
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = tf.keras.activations.tanh(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


def build_camembert_model(nb_class, seq_length, learning_rate):
    """
    Build camemBert model for multilabel classification
    :param nb_class: Number of class
    :param seq_length: The sequence length
    :param learning_rate: Learning rate for training
    :return: The model
    """
    # Input
    input_ids = tf.keras.layers.Input(
        shape=(seq_length,),
        name=INPUT_IDS_KEY,
        dtype=tf.int32

    )

    attention_mask = tf.keras.layers.Input(
        shape=(seq_length,),
        name=ATTENTION_MASK_KEY,
        dtype=tf.int32
    )

    camemBert_model = CamemBertMultilabelClassification(nb_class=nb_class)

    # Output
    camemBert_output = camemBert_model(
        inputs=[input_ids, attention_mask]
    )

    # Initialize the model
    model = tf.keras.models.Model(
        name="CamemBert",
        inputs=[input_ids, attention_mask],
        outputs=[camemBert_output]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.losses.BinaryCrossentropy(from_logits=True),
        metrics='acc'
    )
    model.summary(line_length=130)

    return model
