import streamlit as st
import pandas as pd
from utils import get_logger, get_clients, get_symbols, display_client_trades

from utils import get_sim_data
from sklearn.preprocessing import LabelEncoder

logger = get_logger()
le = LabelEncoder()
le.fit(['SELLER', 'BUYER'])


def get_selected_symbols(df, client_id, key_prefix):
    symbols = get_symbols(df, client_id)
    symbol_map = {
        f"{row['symbol_name']} (ID:{row['symbol_id']})": row['symbol_id']
        for _, row in df[df['client_id'] == client_id][['symbol_id', 'symbol_name']].drop_duplicates().iterrows()
    }
    selected_display = st.multiselect(
        f"Select Symbols for Client {client_id}", list(symbol_map.keys()), key=key_prefix)
    return [symbol_map[s] for s in selected_display]


def render_similarity_table(df, score):
    formatted_rows = []
    pair_labels = []

    for ((bench_key, client_key), sim_score) in score:
        bench_client, bench_symbol = bench_key.split('_')
        client_id, client_symbol = client_key.split('_')

        bench_name = df[(df['client_id'] == int(bench_client)) & (
            df['symbol_id'] == int(bench_symbol))]['symbol_name'].iloc[0]
        client_name = df[(df['client_id'] == int(client_id)) & (
            df['symbol_id'] == int(client_symbol))]['symbol_name'].iloc[0]

        percent_score = round(float(sim_score) * 100, 2)
        formatted_rows.append({
            "Benchmark Client": bench_client,
            "Benchmark Symbol": f"{bench_name} (ID:{bench_symbol})",
            "Compared Client": client_id,
            "Compared Symbol": f"{client_name} (ID:{client_symbol})",
            "Similarity Index": f"{percent_score}%",
        })

        pair_labels.append(
            f"{bench_client}_{bench_name}_{bench_symbol} vs {client_id}_{client_name}_{client_symbol}"
        )

    st.table(pd.DataFrame(formatted_rows))
    return pair_labels


def display_selected_pair(df, selected_pair_label):
    parts = selected_pair_label.split(" vs ")[0].split(
        "_") + selected_pair_label.split(" vs ")[1].split("_")
    bench_client, _, bench_symbol, client_id, _, client_symbol = parts

    bench_df = df[(df['client_id'] == int(bench_client)) &
                  (df['symbol_id'] == int(bench_symbol))]
    comp_df = df[(df['client_id'] == int(client_id)) &
                 (df['symbol_id'] == int(client_symbol))]

    col1, col2 = st.columns(2)
    display_client_trades(col1, bench_df, le, bench_client,
                          bench_symbol, label='Benchmark')
    display_client_trades(col2, comp_df, le, client_id,
                          client_symbol, label='Compared')


def one_one_visualize(df, ml_func):
    st.title("Single Client Comparison")
    clients = get_clients(df)

    client1 = st.selectbox("Select Benchmark Client", clients, key="client1")
    selected_symbols1 = get_selected_symbols(
        df, client1, "symbols1") if client1 else []

    client2 = st.selectbox("Select Compared Client", [
                           c for c in clients if c != client1], key="client2")
    selected_symbols2 = get_selected_symbols(
        df, client2, "symbols2") if client2 else []

    input_state = (client1, tuple(selected_symbols1),
                   client2, tuple(selected_symbols2))

    if st.session_state.get("last_input_state") != input_state:
        st.session_state.pop("score_data", None)
    st.session_state["last_input_state"] = input_state

    if st.button("Check Similarity"):
        benchmark = (client1, selected_symbols1)
        score = ml_func(df, benchmark, [(client2, selected_symbols2)])
        st.session_state["score_data"] = score

    score = st.session_state.get("score_data")
    if score:
        st.subheader("Similarity Index Table")
        pair_labels = render_similarity_table(df, score)

        st.markdown("### Select Pair to View Trades")
        selected_pair_label = st.selectbox("Choose a Pair", pair_labels)
        if selected_pair_label:
            display_selected_pair(df, selected_pair_label)


def one_n_visualization(df, ml_func):
    st.title("Multiple Client Symbol Comparison")
    clients = get_clients(df)

    benchmark_client = st.selectbox(
        "Select Benchmark Client", clients, key="benchmark_client")
    selected_benchmark_symbols = get_selected_symbols(
        df, benchmark_client, "benchmark_symbols") if benchmark_client else []

    n = st.number_input("Enter number of comparison clients (N)",
                        min_value=1, max_value=len(clients) - 1, step=1)
    comparison_clients = st.multiselect("Select Compared Clients", [
                                        c for c in clients if c != benchmark_client], max_selections=n, key="comparison_clients")

    comparison_selections = {
        comp_client: get_selected_symbols(
            df, comp_client, f"symbols_{comp_client}")
        for comp_client in comparison_clients
    }

    input_state = (benchmark_client, tuple(selected_benchmark_symbols), tuple(
        (k, tuple(v)) for k, v in comparison_selections.items()))

    if st.session_state.get("last_input_state_n") != input_state:
        st.session_state.pop("score_data_n", None)
    st.session_state["last_input_state_n"] = input_state

    if st.button("Check Similarity", key="check_one_n"):
        benchmark = (benchmark_client, selected_benchmark_symbols)
        clients_data = [(k, v) for k, v in comparison_selections.items()]
        st.session_state["score_data_n"] = ml_func(df, benchmark, clients_data)

    score = st.session_state.get("score_data_n")
    if score:
        st.subheader("Similarity Index Table")
        pair_labels = render_similarity_table(df, score)

        st.markdown("### Select Pair to View Trades")
        selected_pair_label = st.selectbox("Choose a Pair", pair_labels)
        if selected_pair_label:
            display_selected_pair(df, selected_pair_label)


def all_all_visualization(df: pd.DataFrame):
    st.title("Top Matches")

    # Inputs
    top_n = st.number_input("Number of Top Matches",
                            min_value=1, max_value=1500, value=10, step=1)
    sort_order = st.radio("Sort by Similarity", ["Descending", "Ascending"])
    ascending = sort_order == "Ascending"

    # Fetch once
    sim_df = get_sim_data(split=True).sort_values(
        by="similarity", ascending=ascending).reset_index(drop=True)

    if sim_df.empty:
        st.warning("No similarity data available.")
        return

    sim_df = sim_df[:top_n]

    formatted_rows = []
    pair_map = {}

    for _, row in sim_df.iterrows():
        bench_client, bench_symbol = row['benchmark'].split('_')
        client_id, client_symbol = row['candidate'].split('_')
        sim_score = row['similarity']

        bench_row = df[(df['client_id'] == int(bench_client)) &
                       (df['symbol_id'] == int(bench_symbol))]
        client_row = df[(df['client_id'] == int(client_id)) &
                        (df['symbol_id'] == int(client_symbol))]

        if bench_row.empty or client_row.empty:
            continue

        bench_name = bench_row['symbol_name'].iloc[0]
        client_name = client_row['symbol_name'].iloc[0]

        # Count trades
        bench_count = len(bench_row)
        client_count = len(client_row)
        bench_buys = (bench_row['role'] == 'BUYER').sum()
        bench_sells = (bench_row['role'] == 'SELLER').sum()
        client_buys = (client_row['role'] == 'BUYER').sum()
        client_sells = (client_row['role'] == 'SELLER').sum()

        percent_score = round(float(sim_score) * 100, 2)

        formatted_rows.append({
            "Benchmark Client": bench_client,
            "Benchmark Symbol": f"{bench_name} (ID:{bench_symbol})",
            "Benchmark Trades": f"{bench_buys} + {bench_sells} = {bench_count}",
            "Compared Client": client_id,
            "Compared Symbol": f"{client_name} (ID:{client_symbol})",
            "Compared Trades": f"{client_buys} + {client_sells} = {client_count}",
            "Similarity Index": f"{percent_score}%",
            "Difference in Trade Count": f"{abs(bench_count - client_count)}",
        })

        label = f"{bench_client}_{bench_name}_{bench_symbol} vs {client_id}_{client_name}_{client_symbol}"
        pair_map[label] = {
            "bench_client": int(bench_client),
            "bench_symbol": int(bench_symbol),
            "client_id": int(client_id),
            "client_symbol": int(client_symbol)
        }

    if not formatted_rows:
        st.warning("No valid pairs to display.")
        return

    # Display table
    st.subheader("Similarity Index Table")
    st.table(pd.DataFrame(formatted_rows))

    # Select pair
    st.markdown("### Select Pair to View Trades")
    selected_label = st.selectbox("Choose a Pair", list(pair_map.keys()))

    if selected_label:
        pair = pair_map[selected_label]
        bench_df = df[(df['client_id'] == pair['bench_client'])
                      & (df['symbol_id'] == pair['bench_symbol'])]
        comp_df = df[(df['client_id'] == pair['client_id']) &
                     (df['symbol_id'] == pair['client_symbol'])]

        col1, col2 = st.columns(2)
        display_client_trades(
            col1, bench_df, le, pair['bench_client'], pair['bench_symbol'], label='Benchmark')
        display_client_trades(
            col2, comp_df, le, pair['client_id'], pair['client_symbol'], label='Compared')
