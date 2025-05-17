import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.layers import Input, Dense, LSTM, Concatenate, MultiHeadAttention, LayerNormalization
from tensorflow.keras.models import Model
from sklearn.preprocessing import StandardScaler
from utils import get_logger


logger = get_logger()


def create_hybrid_model(feature_dim, seq_length, n_features, embedding_dim=64):
    # Static (aggregated) features input
    feature_input = Input(shape=(feature_dim,), name='static_input')
    f = Dense(128, activation='relu')(feature_input)
    f = Dense(64, activation='relu')(f)

    # Sequential (time-series) input
    seq_input = Input(shape=(seq_length, n_features), name='sequence_input')
    attn = MultiHeadAttention(num_heads=4, key_dim=8)(seq_input, seq_input)
    norm1 = LayerNormalization()(attn + seq_input)
    lstm_out = LSTM(64, return_sequences=False)(norm1)

    # Merge
    merged = Concatenate()([f, lstm_out])
    merged = Dense(128, activation='relu')(merged)
    embedding = Dense(embedding_dim, activation='linear',
                      name='embedding')(merged)

    model = Model(inputs=[feature_input, seq_input], outputs=embedding)
    return model


def load_model(features, sequences,
               model_path='./model/trading_model.keras',
               checkpoint_path='./checkpoints/model_checkpoint.h5',
               max_seq_length=30, embedding_dim=64):

    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    logger.info("Scaling static features...")

    logger.info("Creating model...")
    model = create_hybrid_model(
        feature_dim=features.shape[1],
        seq_length=max_seq_length,
        n_features=sequences.shape[2],
        embedding_dim=embedding_dim
    )

    if os.path.exists(model_path):
        logger.info(f"Loading model from {model_path}...")
        model.load_weights(model_path)
    else:
        raise FileNotFoundError(
            f"""Model not found at {model_path}! Please train the model first
            & place in the suggest folder."""
        )

    return model
