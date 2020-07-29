import streamlit as st
import tensorflow as tf
import numpy as np
from transformers import AutoConfig, AutoTokenizer
from transformers import TFBertForSequenceClassification
from transformers import TFBertForTokenClassification
from transformers import TFBertForMaskedLM
from .cleaning import cleanize


@st.cache(allow_output_mutation=True)
def load_config(_id):
    """ Load the model configuration """
    return AutoConfig.from_pretrained(_id)


@st.cache(allow_output_mutation=True)
def load_tokenizer(_id):
    """ Load the model tokenizer """
    return AutoTokenizer.from_pretrained(_id)


@st.cache(allow_output_mutation=True)
def load_model(_id, _type):
    """ Load the models """

    models = {
        'TFBertForSequenceClassification': TFBertForSequenceClassification,
        'TFBertForTokenClassification': TFBertForTokenClassification,
        'TFBertForMaskedLM': TFBertForMaskedLM
    }
    model = models[_type].from_pretrained(_id)
    return model


def sequence_predicting(model, tokenizer, text, labels):
    """ Sequence-prediction for tasks like text classification, ... """
    text = cleanize(text)
    inputs = tokenizer.encode(text, return_tensors="tf", max_length=128)

    logits = model(inputs)[0]
    outputs = tf.keras.backend.softmax(logits)
    prediction = tf.argmax(outputs, axis=1)
    prediction = prediction[0].numpy()
    scores = outputs[0].numpy()
    return scores, labels[prediction]


def token_predicting(model, tokenizer, text, labels):
    """ Token-prediction for tasks like named entity recognition, ... """
    text = cleanize(text)

    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(text)))
    inputs = tokenizer.encode(text, return_tensors="tf", max_length=128)

    outputs = model(inputs)[0]
    predictions = tf.argmax(outputs, axis=2)
    predictions = [(token, labels[prediction]) for token, prediction in
                   zip(tokens, predictions[0].numpy())]

    return predictions


def text_generation(model, tokenizer, text):
    """ Text-generation for tasks like filling the MASK """
    text = cleanize(text)

    words = np.array(text.split())
    masks = np.where(words == tokenizer.mask_token)[0]

    inputs = tokenizer.batch_encode_plus(
        [text],
        add_special_tokens=True,
        return_tensors='tf',
        pad_to_max_length=True)

    outputs = model(inputs.data, training=False)[0]

    batch_size = outputs.shape[0]
    masked_words = {}
    masked_colors = ['#0096C7', '#00B4D8', '#48CAE4', '#90E0EF', '#ADE8F4']

    for i in range(batch_size):
        input_ids = inputs["input_ids"][i]

        masked_indices = tf.where(input_ids == tokenizer.mask_token_id).numpy()
        for masked_index, mask in zip(masked_indices, masks):
            masked_index = masked_index[0]
            masked_words[mask] = []

            logits = outputs[i, masked_index, :]
            probs = tf.nn.softmax(logits)
            topk = tf.math.top_k(probs, k=5)
            values, predictions = topk.values.numpy(), topk.indices.numpy()

            for j, (v, p) in enumerate(zip(values.tolist(), predictions.tolist())):
                tokens = input_ids.numpy()
                tokens[masked_index] = p
                masked_words[mask].append({
                    'score': v,
                    'token': p,
                    'color': masked_colors[j],
                    'token_str': tokenizer.convert_ids_to_tokens(p)
                })

    return masked_words, words
