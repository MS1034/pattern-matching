from utils import get_clients, get_symbols, display_trade_tables
import streamlit as st
import pandas as pd
from utils import get_logger, get_sim_data

logger = get_logger()


def one_one_visualize(df, ml_func):
    st.title("Single Client Comparison")

    # --- Selection Inputs ---
    clients = get_clients(df)
    client1 = st.selectbox("Select Benchmark Client", clients, key="client1")

    selected_symbols1 = []
    if client1:
        symbols1 = get_symbols(df, client1)
        selected_symbols1 = st.multiselect(
            "Select Symbols for Benchmark Client", symbols1, key="symbols1")

    client2 = st.selectbox("Select Compared Client", [
        c for c in clients if c != client1], key="client2")

    selected_symbols2 = []
    if client2:
        symbols2 = get_symbols(df, client2)
        selected_symbols2 = st.multiselect(
            "Select Symbols for Compared Client", symbols2, key="symbols2")

    # --- Unique Key to Track Input State ---
    input_state = (client1, tuple(selected_symbols1),
                   client2, tuple(selected_symbols2))

    # Invalidate score data if input has changed
    if "last_input_state" in st.session_state:
        if st.session_state["last_input_state"] != input_state:
            st.session_state.pop("score_data", None)
    st.session_state["last_input_state"] = input_state

    # --- Button and Score Computation ---
    if st.button("Check Similarity"):
        benchmark = (client1, selected_symbols1)
        clients = [(client2, selected_symbols2)]
        score = ml_func(df, benchmark, clients)
        st.session_state["score_data"] = score
    else:
        score = st.session_state.get("score_data")

    # --- Show Results ---
    if score:
        st.subheader("Similarity Index Table")

        formatted_rows = []
        pair_labels = []

        for ((bench_key, client_key), sim_score) in score:
            bench_client, bench_symbol,  = bench_key.split('_')
            client_id, client_symbol = client_key.split('_')

            percent_score = round(float(sim_score) * 100, 2)
            formatted_rows.append({
                "Benchmark Client": bench_client,
                "Benchmark Symbol": bench_symbol,
                "Compared Client": client_id,
                "Compared Symbol": client_symbol,
                "Similarity Index": f"{percent_score}%",
            })

            pair_labels.append(
                f"{bench_client}_{bench_symbol} vs {client_id}_{client_symbol}"
            )

        st.table(pd.DataFrame(formatted_rows))

        # --- Trade View Interface ---
        st.markdown("###  Select Pair to View Trades")
        selected_pair_label = st.selectbox("Choose a Pair", pair_labels)

        if selected_pair_label:
            parts = selected_pair_label.replace(
                " vs ", " ").replace("_", " ").split()
            bench_client, bench_symbol, client_id, client_symbol = parts

            bench_df = df[(df['client_id'] == int(bench_client)) &
                          (df['symbol_id'] == int(bench_symbol))]
            comp_df = df[(df['client_id'] == int(client_id)) &
                         (df['symbol_id'] == int(client_symbol))]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"**Trades for Benchmark Client `{bench_client}` - Symbol `{bench_symbol}`**")
                st.dataframe(bench_df, use_container_width=True, height=300)
            with col2:
                st.markdown(
                    f"**Trades for Compared Client `{client_id}` - Symbol `{client_symbol}`**")
                st.dataframe(comp_df, use_container_width=True, height=300)


def one_n_visualization(df, ml_func):
    st.title("Multiple Client Symbol Comparison")

    clients = get_clients(df)
    benchmark_client = st.selectbox(
        "Select Benchmark Client", clients, key="benchmark_client")

    selected_benchmark_symbols = []
    if benchmark_client:
        benchmark_symbols = get_symbols(df, benchmark_client)
        selected_benchmark_symbols = st.multiselect(
            "Select Symbols for Benchmark Client", benchmark_symbols, key="benchmark_symbols"
        )

    n = st.number_input("Enter number of comparison clients (N)",
                        min_value=1, max_value=len(clients) - 1, step=1)

    comparison_clients = st.multiselect(
        "Select Compared Clients",
        [c for c in clients if c != benchmark_client],
        max_selections=n,
        key="comparison_clients"
    )

    comparison_selections = {}
    for comp_client in comparison_clients:
        symbols = get_symbols(df, comp_client)
        selected_symbols = st.multiselect(
            f"Select Symbols for Compared Client {comp_client}",
            symbols,
            key=f"symbols_{comp_client}"
        )
        comparison_selections[comp_client] = selected_symbols

    # Track input state
    input_state = (benchmark_client, tuple(selected_benchmark_symbols),
                   tuple((k, tuple(v)) for k, v in comparison_selections.items()))

    if "last_input_state_n" in st.session_state:
        if st.session_state["last_input_state_n"] != input_state:
            st.session_state.pop("score_data_n", None)
    st.session_state["last_input_state_n"] = input_state

    if st.button("Check Similarity", key="check_one_n"):
        benchmark = (benchmark_client, selected_benchmark_symbols)
        clients_data = [(comp_client, comp_symbols)
                        for comp_client, comp_symbols in comparison_selections.items()]
        score = ml_func(df, benchmark, clients_data)
        st.session_state["score_data_n"] = score
    else:
        score = st.session_state.get("score_data_n")

    if score:
        st.subheader("Similarity Index Table")

        formatted_rows = []
        pair_labels = []

        for ((bench_key, client_key), sim_score) in score:
            bench_client, bench_symbol = bench_key.split('_')
            client_id, client_symbol = client_key.split('_')

            percent_score = round(float(sim_score) * 100, 2)
            formatted_rows.append({
                "Benchmark Client": bench_client,
                "Benchmark Symbol": bench_symbol,
                "Compared Client": client_id,
                "Compared Symbol": client_symbol,
                "Similarity Index": f"{percent_score}%",
            })

            pair_labels.append(
                f"{bench_client}_{bench_symbol} vs {client_id}_{client_symbol}"
            )

        st.table(pd.DataFrame(formatted_rows))

        # Trade view
        st.markdown("###  Select Pair to View Trades")
        selected_pair_label = st.selectbox("Choose a Pair", pair_labels)

        if selected_pair_label:
            # Safe parsing
            parts = selected_pair_label.replace(
                " vs ", " ").replace("_", " ").split()
            bench_client, bench_symbol, client_id, client_symbol = parts

            # Cast to int for filtering
            bench_client = int(bench_client)
            bench_symbol = int(bench_symbol)
            client_id = int(client_id)
            client_symbol = int(client_symbol)

            bench_df = df[(df['client_id'] == bench_client) &
                          (df['symbol_id'] == bench_symbol)]
            comp_df = df[(df['client_id'] == client_id) &
                         (df['symbol_id'] == client_symbol)]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"**Trades for Benchmark Client `{bench_client}` - Symbol `{bench_symbol}`**")
                st.dataframe(bench_df, use_container_width=True, height=300)
            with col2:
                st.markdown(
                    f"**Trades for Compared Client `{client_id}` - Symbol `{client_symbol}`**")
                st.dataframe(comp_df, use_container_width=True, height=300)


# def one_vs_all_visualization(df, ml_func):
#     st.title("1 vs All Client Symbol Comparison")

#     clients = get_clients(df)
#     benchmark_client = st.selectbox(
#         "Select Benchmark Client", clients, key="one_vs_all_benchmark_client")

#     selected_symbol = None
#     if benchmark_client:
#         symbols = get_symbols(df, benchmark_client)
#         selected_symbol = st.selectbox(
#             "Select Benchmark Symbol", symbols, key="one_vs_all_symbol")

#     if st.button("Compare with All", key="compare_with_all"):
#         if not benchmark_client or selected_symbol is None:
#             st.warning("Please select both a benchmark client and symbol.")
#             return

#         logger.info(
#             f"Starting comparison: Benchmark Client = {benchmark_client}, Symbol = {selected_symbol}")

#         other_clients = [c for c in clients if c != benchmark_client]
#         total = sum(len(get_symbols(df, c)) for c in other_clients)
#         logger.info(f"Total client-symbol comparisons to perform: {total}")
#         progress = st.progress(0)

#         benchmark = (int(benchmark_client), [int(selected_symbol)])
#         results = []
#         count = 0

#         for client in other_clients:
#             symbols = get_symbols(df, client)
#             for symbol in symbols:
#                 logger.debug(f"Comparing to Client {client}, Symbol {symbol}")
#                 score = ml_func(df, benchmark, [(int(client), [int(symbol)])])

#                 if not score:
#                     logger.warning(
#                         f"No similarity score returned for {client}_{symbol}")
#                     continue

#                 ((bench_key, comp_key), sim_score) = score[0]
#                 bench_client, bench_symbol = bench_key.split('_')
#                 comp_client, comp_symbol = comp_key.split('_')

#                 percent_score = round(float(sim_score) * 100, 2)
#                 logger.info(
#                     f"Score: {bench_key} vs {comp_key} = {percent_score}%")

#                 results.append({
#                     "Benchmark Client": bench_client,
#                     "Benchmark Symbol": bench_symbol,
#                     "Compared Client": comp_client,
#                     "Compared Symbol": comp_symbol,
#                     "Similarity Index": f"{percent_score}%",
#                 })

#                 count += 1
#                 progress.progress(min(1.0, count / total))

#         if results:
#             results_df = pd.DataFrame(results)
#             results_df["Similarity Index %"] = results_df["Similarity Index"].str.replace(
#                 "%", "").astype(float)
#             results_df = results_df.sort_values(
#                 "Similarity Index %", ascending=False)

#             st.subheader("Similarity Index Table (Sorted)")
#             st.table(results_df.drop(columns=["Similarity Index %"]))
#             logger.info("Displaying sorted similarity results")

#             st.markdown("###  Select Pair to View Trades")
#             pair_labels = [
#                 f"{row['Benchmark Client']}_{row['Benchmark Symbol']} vs {row['Compared Client']}_{row['Compared Symbol']}"
#                 for row in results_df.to_dict("records")
#             ]
#             selected_pair_label = st.selectbox("Choose a Pair", pair_labels)

#             if selected_pair_label:
#                 parts = selected_pair_label.replace(
#                     " vs ", " ").replace("_", " ").split()
#                 bench_client, bench_symbol, client_id, client_symbol = map(
#                     int, parts)

#                 logger.info(
#                     f"Loading trades for: {bench_client}_{bench_symbol} vs {client_id}_{client_symbol}")

#                 bench_df = df[(df['client_id'] == bench_client)
#                               & (df['symbol_id'] == bench_symbol)]
#                 comp_df = df[(df['client_id'] == client_id) &
#                              (df['symbol_id'] == client_symbol)]

#                 col1, col2 = st.columns(2)
#                 with col1:
#                     st.markdown(
#                         f"**Trades for Benchmark Client `{bench_client}` - Symbol `{bench_symbol}`**")
#                     st.dataframe(
#                         bench_df, use_container_width=True, height=300)
#                 with col2:
#                     st.markdown(
#                         f"**Trades for Compared Client `{client_id}` - Symbol `{client_symbol}`**")
#                     st.dataframe(comp_df, use_container_width=True, height=300)
#         else:
#             st.info("No results found.")
#             logger.info("No matches found during comparison.")


def all_all_visualization(df: pd.DataFrame):
    st.title("Top Matches")
    top_n = st.number_input(
        "Number of Top Matches", min_value=1, max_value=100, value=10, step=1)

    display_df = get_sim_data()[:top_n]
    st.subheader("Similarity Index Table")
    st.table(display_df)

    score = get_sim_data(split=True)[:top_n]

    if score is not None and not score.empty:
        formatted_rows = []
        pair_labels = []

        for _, row in score.iterrows():
            bench_client, bench_symbol = row['benchmark'].split('_')
            client_id, client_symbol = row['candidate'].split('_')
            sim_score = row['similarity']

            percent_score = round(float(sim_score) * 100, 2)
            formatted_rows.append({
                "Benchmark Client": bench_client,
                "Benchmark Symbol": bench_symbol,
                "Compared Client": client_id,
                "Compared Symbol": client_symbol,
                "Similarity Index": f"{percent_score}%",
            })

            pair_labels.append(
                f"{bench_client}_{bench_symbol} vs {client_id}_{client_symbol}")

        st.markdown("###  Select Pair to View Trades")
        selected_pair_label = st.selectbox("Choose a Pair", pair_labels)

        if selected_pair_label:
            parts = selected_pair_label.replace(
                " vs ", " ").replace("_", " ").split()
            bench_client, bench_symbol, client_id, client_symbol = parts

            bench_df = df[
                (df['client_id'] == int(bench_client)) &
                (df['symbol_id'] == int(bench_symbol))
            ]
            comp_df = df[
                (df['client_id'] == int(client_id)) &
                (df['symbol_id'] == int(client_symbol))
            ]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"*Trades for Benchmark Client {bench_client} - Symbol {bench_symbol}*")
                st.dataframe(bench_df, use_container_width=True, height=300)
            with col2:
                st.markdown(
                    f"*Trades for Compared Client {client_id} - Symbol {client_symbol}*")
                st.dataframe(comp_df, use_container_width=True, height=300)
