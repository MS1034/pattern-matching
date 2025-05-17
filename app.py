import streamlit as st
from utils import load_data
from model import load_model
from disp_types import one_one_visualize, one_n_visualization
from disp_types import all_all_visualization
# from disp_types import one_vs_all_visualization
from sequence_encoder import encode_roles, generate_sequences
from distance_engine import match_clients
from sklearn.preprocessing import StandardScaler

# Placeholder Machine Learning Model function

max_seq_length = 200


def f(df, benchmark, clients_to_match):
    features, sequences, feature_cols, sequence_cols, _ = generate_sequences(
        df, max_seq_length=max_seq_length)
    model = load_model(features=features,
                       sequences=sequences,
                       model_path="model/trading_model_final.keras",
                       max_seq_length=max_seq_length,
                       embedding_dim=features.shape[1]
                       )
    df = encode_roles(df)
    df = df.fillna(0)
    scaler = StandardScaler()
    scaler.fit_transform(features)

    result = match_clients(
        df,
        benchmark=benchmark,
        clients_to_match=clients_to_match,
        feature_cols=feature_cols,
        sequence_cols=sequence_cols,
        top_n=None,
        max_seq_length=max_seq_length,
        scaler=scaler,
        model=model,
    )

    return result


def main():
    st.set_page_config(layout="wide", initial_sidebar_state='collapsed')
    st.sidebar.title("Visualization Selector")
    vis_type = st.sidebar.selectbox(
        "Select Visualization Type",
        # ["1-1", "1-N (select N)", "1-ALL", "All-All"],
        ["Single Client", "Multiple Clients", "Top Matches"],
        index=0
    )
    df = load_data("data/final_combined.parquet")

    if vis_type == "Single Client":
        one_one_visualize(df, f)
    elif vis_type == "Multiple Clients":
        one_n_visualization(df, f)
    # elif vis_type == "1-ALL":
    #     one_vs_all_visualization(df, f)
    elif vis_type == "Top Matches":
        all_all_visualization(df)


if __name__ == "__main__":
    main()
