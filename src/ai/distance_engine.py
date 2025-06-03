from sklearn.metrics.pairwise import cosine_similarity
from fastdtw import fastdtw
from scipy.stats import wasserstein_distance
from utils import get_logger
from ai.sequence_encoder import pad_sequence
from tqdm import tqdm
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import faiss
from concurrent.futures import ThreadPoolExecutor, as_completed


logger = get_logger()


def cosine_similarity_score(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings, scaled to [0, 1]."""
    raw_score = cosine_similarity([embedding1], [embedding2])[0][0]
    scaled_score = (raw_score + 1) / 2
    return scaled_score


def dtw_similarity_score(sequence1, sequence2):
    """Calculate Dynamic Time Warping (DTW) distance between two sequences."""
    distance, _ = fastdtw(sequence1, sequence2)
    return 1 / (1 + distance)


def wasserstein_similarity_score(sequence1, sequence2):
    """Calculate Wasserstein distance between two distributions."""
    # Select a specific feature for comparison (e.g., volume)
    feature_index = 2  # Assuming volume is the 3rd column in the sequence

    # Extract the feature from the sequences
    dist1 = sequence1[:, feature_index]
    dist2 = sequence2[:, feature_index]

    return 1 / (1 + wasserstein_distance(dist1, dist2))


def calculate_similarity(
    benchmark_embedding,
    benchmark_sequence,
    candidate_embedding,
    candidate_sequence,
    weights=[0.4, 0.3, 0.3],
):
    """Calculate weighted similarity score from cosine, DTW, and Wasserstein.
    """

    cosine_sim = cosine_similarity_score(
        benchmark_embedding, candidate_embedding)
    dtw_sim = dtw_similarity_score(benchmark_sequence, candidate_sequence)
    # wasserstein_sim = wasserstein_similarity_score(
    #     benchmark_sequence, candidate_sequence
    # )

    # weighted_score = (
    #     weights[0] * cosine_sim + weights[1] *
    #     dtw_sim + weights[2] * wasserstein_sim
    # )
    weighted_score = (
        0.8 * cosine_sim + 0.2 *
        dtw_sim
    )
    max(weighted_score, 1)
    return weighted_score


def build_client_map(
    df,
    client_id,
    feature_cols,
    sequence_cols,
    symbols=None,
    max_seq_length=200,
    scaler=None,
    model=None,
):
    """
    Build client map for a single client (with optional multiple symbols).

    Parameters:
        df (pd.DataFrame): Trade data containing 'client_id', 'symbol_id',
        'entry_datetime', etc.
        client_id (str): ID of the client to process.
        feature_cols (list): Columns to be used for feature vector.
        sequence_cols (list): Columns to be used in sequence modeling.
        symbols (list or None): List of symbols to include; if None, use all.
        max_seq_length (int): Max length of time-series sequence.
        scaler (StandardScaler or None): Optional pre-fitted scaler.
        model (tf.keras.Model or similar): Trained embedding model.

    Returns:
        dict: A map of symbol-wise processed data including embeddings.
    """
    if client_id is None:
        raise ValueError("client_id must not be None")

    client_map = {}
    features = []
    sequences = []
    volume_dists = []
    keys = []

    client_df = df[df["client_id"] == client_id]
    if client_df.empty:
        logger.warning(f"No data found for client {client_id}")
        return {}

    for symbol, group in client_df.groupby("symbol_id"):
        if symbols is not None and symbol not in symbols:
            continue

        if group.empty:
            logger.warning(
                f"No data for client {client_id} and symbol {symbol}")
            continue

        group = group.sort_values("entry_datetime")

        feature_vector = group[feature_cols].mean().values
        sequence = pad_sequence(group[sequence_cols].values, max_seq_length)
        volume_dist = group["volume"].values

        key = f"{client_id}_{symbol}"
        keys.append(key)
        features.append(feature_vector)
        sequences.append(sequence)
        volume_dists.append(volume_dist)

    if not features:
        logger.warning(f"No valid data segments for client {client_id}")
        return {}

    features = np.array(features)
    sequences = np.array(sequences)

    if scaler is None:
        scaler = StandardScaler()
        (scaler.fitfeatures)

    features_scaled = scaler.transform(features)

    if model is None:
        raise RuntimeError(
            "A trained model must be provided to generate embeddings.")

    embeddings = model.predict([features_scaled, sequences], verbose=0)

    for i, key in enumerate(keys):
        _, symbol = key.split("_")
        client_map[key] = {
            "client_id": client_id,
            "symbol_id": symbol,
            "feature_vector": features[i],
            "sequence": sequences[i],
            "volume_dist": volume_dists[i],
            "embedding": embeddings[i],
        }

    return client_map


def match_clients(
    df,
    benchmark=None,
    clients_to_match=None,
    feature_cols=None,
    sequence_cols=None,
    top_n=None,
    max_seq_length=200,
    scaler=None,
    model=None,
    save_path=None,
    date_filter=None,
):
    """
    Match benchmark client-symbol pair with others or specific
    client-symbol pairs.

    Cases:
    - Case 1: Match all client-symbol pairs with each other
    (excluding themselves).
    - Case 2: Match benchmark against all others.
    - Case 3: Match benchmark against specified client-symbols.
    """
    similarity_scores = []

    if date_filter:
        logger.info(f"Filtering data with date >= {date_filter}")
        df = df[df["entry_datetime"] >= date_filter]
        logger.info(f"Remaining rows after date filter: {len(df)}")

    if not benchmark and not clients_to_match:
        logger.info("Running CASE I: All-vs-All client-symbol similarity.")
        similarity_scores = []

        real_pairs = df[["client_id", "symbol_id"]].drop_duplicates()
        logger.info(f"Total unique client-symbol pairs: {len(real_pairs)}")

        # Step 1: Build all client-symbol sequences & embeddings
        all_maps = {}
        keys = []
        for i, row in tqdm(
            real_pairs.iterrows(), total=len(real_pairs), desc="Building maps"
        ):
            cid, sid = row["client_id"], row["symbol_id"]
            c_key = f"{cid}_{sid}"

            client_map = build_client_map(
                df,
                client_id=cid,
                feature_cols=feature_cols,
                sequence_cols=sequence_cols,
                symbols=[sid],
                max_seq_length=max_seq_length,
                scaler=scaler,
                model=model,
            )
            if not client_map or c_key not in client_map:
                logger.warning(f"Skipping {c_key}: no data.")
                continue

            all_maps[c_key] = client_map[c_key]
            keys.append(c_key)

        logger.info(f"Built {len(all_maps)} valid client-symbol embeddings.")

        embedding_matrix = np.array([all_maps[k]["embedding"] for k in keys]).astype(
            "float32"
        )
        index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        index.add(embedding_matrix)
        top_k = 200  # Tune this for quality vs performance

        logger.info("Using FAISS to find top-k similar candidates...")
        D, I = index.search(embedding_matrix, top_k)

        # Step 3: Compute real similarity only for top candidates, in parallel
        def compute_score(i, j):
            key1 = keys[i]
            key2 = keys[j]
            vec1 = all_maps[key1]
            vec2 = all_maps[key2]

            score = calculate_similarity(
                vec1["embedding"], vec1["sequence"], vec2["embedding"],
                vec2["sequence"]
            )
            if score > 70:
                logger.info(
                    f"High similarity ({score:.2f}) between {key1} and {key2}")
            return ((key1, key2), score)

        logger.info(
            "Calculating true similarity scores using multithreading...")

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = []
            for i, neighbors in enumerate(I):
                for j in neighbors:
                    if i >= j:
                        continue
                    futures.append(executor.submit(compute_score, i, j))

            for future in tqdm(
                as_completed(futures), total=len(futures),
                desc="Computing similarities"
            ):
                try:
                    pair, score = future.result()
                    similarity_scores.append((pair, score))
                except Exception as e:
                    logger.error(f"Error in similarity computation: {e}")

        final_scores = sorted(
            similarity_scores, key=lambda x: x[1], reverse=True)

    # ---- CASE II: Benchmark vs All Others ----
    elif benchmark and not clients_to_match:
        logger.info("Running CASE II: Benchmark-vs-All.")
        bench_id, bench_symbols = benchmark
        logger.info(
            f"""Building benchmark map for client {bench_id},
            symbols={bench_symbols}"""
        )
        benchmark_map = build_client_map(
            df,
            client_id=bench_id,
            feature_cols=feature_cols,
            sequence_cols=sequence_cols,
            symbols=bench_symbols,
            max_seq_length=max_seq_length,
            scaler=scaler,
            model=model,
        )
        if not benchmark_map:
            logger.warning(f"No benchmark data found for client {bench_id}")
            return []

        real_pairs = df[["client_id", "symbol_id"]].drop_duplicates()
        logger.info(f"Total candidate pairs: {len(real_pairs)}")

        for i, row in tqdm(
            real_pairs.iterrows(), total=len(real_pairs),
            desc="Benchmark-vs-All loop"
        ):
            cid, sid = row["client_id"], row["symbol_id"]
            if cid == bench_id and sid in bench_symbols:
                continue

            c_key = f"{cid}_{sid}"
            logger.info(f"Processing candidate {c_key}")
            candidate_map = build_client_map(
                df,
                client_id=cid,
                feature_cols=feature_cols,
                sequence_cols=sequence_cols,
                symbols=[sid],
                max_seq_length=max_seq_length,
                scaler=scaler,
                model=model,
            )
            if not candidate_map:
                logger.warning(f"No data for candidate {c_key}, skipping.")
                continue
            c_vec = candidate_map[c_key]

            for b_key, b_vec in benchmark_map.items():
                score = calculate_similarity(
                    b_vec["embedding"],
                    b_vec["sequence"],
                    c_vec["embedding"],
                    c_vec["sequence"],
                )
                similarity_scores.append(((b_key, c_key), score))

    # ---- CASE III: Benchmark vs Specific Client-Symbols ----
    elif benchmark and clients_to_match:
        logger.info("Running CASE III: Benchmark-vs-Specific Matches.")
        bench_id, bench_symbols = benchmark
        logger.info(
            f"""Building benchmark map for client {bench_id},
            symbols={bench_symbols}"""
        )
        benchmark_map = build_client_map(
            df,
            client_id=bench_id,
            feature_cols=feature_cols,
            sequence_cols=sequence_cols,
            symbols=bench_symbols,
            max_seq_length=max_seq_length,
            scaler=scaler,
            model=model,
        )
        if not benchmark_map:
            logger.warning(f"No benchmark data found for client {bench_id}")
            return []

        for client_id, symbol_list in tqdm(
            clients_to_match, desc="Benchmark-vs-Specific loop"
        ):
            logger.info(
                f"""Processing candidate client {client_id}
                symbols={symbol_list}"""
            )
            candidate_map = build_client_map(
                df,
                client_id=client_id,
                feature_cols=feature_cols,
                sequence_cols=sequence_cols,
                symbols=symbol_list,
                max_seq_length=max_seq_length,
                scaler=scaler,
                model=model,
            )
            if not candidate_map:
                logger.warning(f"No candidate data for client {client_id}")
                continue

            for c_key, c_vec in candidate_map.items():
                for b_key, b_vec in benchmark_map.items():
                    score = calculate_similarity(
                        b_vec["embedding"],
                        b_vec["sequence"],
                        c_vec["embedding"],
                        c_vec["sequence"],
                    )
                    similarity_scores.append(((b_key, c_key), score))

    else:
        logger.warning("No valid matching case triggered.")
        return []

    # Sort and keep top N
    similarity_scores = sorted(
        similarity_scores, key=lambda x: x[1], reverse=True)
    if top_n:
        similarity_scores = similarity_scores[:top_n]

    # Save output
    if save_path and similarity_scores:
        os.makedirs(save_path, exist_ok=True)
        df_out = pd.DataFrame(
            [(a, b, score) for ((a, b), score) in similarity_scores],
            columns=["benchmark", "candidate", "similarity"],
        )
        output_file = os.path.join(save_path, "similarity_results.csv")
        df_out.to_csv(output_file, index=False)
        logger.info(f"Results saved to {output_file}")

    logger.info(
        f"Matching complete. Total comparisons: {len(similarity_scores)}")
    return similarity_scores
