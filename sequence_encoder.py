import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder


def encode_roles(df):
    if 'role' in df.columns:
        le = LabelEncoder()
        df['role_encoded'] = le.fit_transform(df['role'].astype(str))
    else:
        df['role_encoded'] = 0
    return df


def pad_sequence(seq, max_seq_length):
    if len(seq) >= max_seq_length:
        return seq[-max_seq_length:]  # take last N trades
    padding = np.zeros((max_seq_length - len(seq), seq.shape[1]))
    return np.vstack([padding, seq])  # pre-pad with zeros


def generate_sequences(df, max_seq_length=150):
    """
    Generate padded trade sequences and static feature vectors per
    client-symbol pair.

    Returns:
    --------
    features: np.ndarray
        Static (final trade) feature vectors
    sequences: np.ndarray
        Padded sequences (price, volume, etc.)
    ids: list of tuples
        List of (client_id, symbol_id)
    """
    df = encode_roles(df)

    sequence_cols = ['role_encoded', 'price',
                     'volume', 'price_change_pct', 'cum_net_vol']
    feature_cols = [
        'price', 'volume', 'secs_since_prev_trade', 'price_change_pct',
        'cum_net_vol', 'is_role_switch', 'role_encoded', 'amount'
    ]

    sequences = []
    features = []
    ids = []

    df = df.sort_values(['client_id', 'symbol_id', 'entry_datetime'])
    df = df.dropna(subset=sequence_cols + feature_cols)

    for (client_id, symbol_id), group in df.groupby(['client_id', 'symbol_id']):
        if len(group) < 2:
            continue  # skip too-short sequences

        seq_array = group[sequence_cols].values.astype(float)
        padded_seq = pad_sequence(seq_array, max_seq_length)

        static_vector = group.iloc[-1][feature_cols].values.astype(float)

        sequences.append(padded_seq)
        features.append(static_vector)
        ids.append((client_id, symbol_id))

    return np.array(features), np.array(sequences), feature_cols, sequence_cols, ids
