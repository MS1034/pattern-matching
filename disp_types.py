from utils import get_clients, get_symbols, display_trade_tables
import streamlit as st
import pandas as pd
from utils import get_logger, get_sim_data

logger = get_logger()


def one_one_visualize(df, ml_func):
    st.title("Single Client Comparison")

    clients = get_clients(df)
    client1 = st.selectbox("Select Benchmark Client", clients, key="client1")

    selected_symbols1 = []
    if client1:
        symbols1 = get_symbols(df, client1)
        symbol_map1 = {f"{row['symbol_name']} (ID:{row['symbol_id']})": row['symbol_id']
                       for _, row in df[df['client_id'] == client1][['symbol_id', 'symbol_name']].drop_duplicates().iterrows()}
        selected_display1 = st.multiselect("Select Symbols for Benchmark Client",
                                           list(symbol_map1.keys()), key="symbols1")
        selected_symbols1 = [symbol_map1[s] for s in selected_display1]

    client2 = st.selectbox("Select Compared Client", [
        c for c in clients if c != client1], key="client2")

    selected_symbols2 = []
    if client2:
        symbols2 = get_symbols(df, client2)
        symbol_map2 = {f"{row['symbol_name']} (ID:{row['symbol_id']})": row['symbol_id']
                       for _, row in df[df['client_id'] == client2][['symbol_id', 'symbol_name']].drop_duplicates().iterrows()}
        selected_display2 = st.multiselect("Select Symbols for Compared Client",
                                           list(symbol_map2.keys()), key="symbols2")
        selected_symbols2 = [symbol_map2[s] for s in selected_display2]

    input_state = (client1, tuple(selected_symbols1),
                   client2, tuple(selected_symbols2))

    if "last_input_state" in st.session_state:
        if st.session_state["last_input_state"] != input_state:
            st.session_state.pop("score_data", None)
    st.session_state["last_input_state"] = input_state

    if st.button("Check Similarity"):
        benchmark = (client1, selected_symbols1)
        clients = [(client2, selected_symbols2)]
        score = ml_func(df, benchmark, clients)
        st.session_state["score_data"] = score
    else:
        score = st.session_state.get("score_data")

    if score:
        st.subheader("Similarity Index Table")

        formatted_rows = []
        pair_labels = []

        for ((bench_key, client_key), sim_score) in score:
            bench_client, bench_symbol = bench_key.split('_')
            client_id, client_symbol = client_key.split('_')

            bench_name = df[(df['client_id'] == int(bench_client)) &
                            (df['symbol_id'] == int(bench_symbol))]['symbol_name'].iloc[0]
            client_name = df[(df['client_id'] == int(client_id)) &
                             (df['symbol_id'] == int(client_symbol))]['symbol_name'].iloc[0]

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

        st.markdown("###  Select Pair to View Trades")
        selected_pair_label = st.selectbox("Choose a Pair", pair_labels)

        if selected_pair_label:
            parts = selected_pair_label.split(" vs ")[0].split(
                "_") + selected_pair_label.split(" vs ")[1].split("_")
            bench_client, _, bench_symbol, client_id, _, client_symbol = parts

            bench_df = df[(df['client_id'] == int(bench_client)) &
                          (df['symbol_id'] == int(bench_symbol))]
            comp_df = df[(df['client_id'] == int(client_id)) &
                         (df['symbol_id'] == int(client_symbol))]
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"**Trades for Benchmark Client `{bench_client}` - Symbol `{bench_symbol} - COUNT: Trades {len(bench_df)}`**")
                st.dataframe(bench_df, use_container_width=True, height=300)
            with col2:
                st.markdown(
                    f"**Trades for Compared Client `{client_id}` - Symbol `{client_symbol} - COUNT: Trades {len(comp_df)}`**")
                st.dataframe(comp_df, use_container_width=True, height=300)


def one_n_visualization(df, ml_func):
    st.title("Multiple Client Symbol Comparison")

    clients = get_clients(df)
    benchmark_client = st.selectbox(
        "Select Benchmark Client", clients, key="benchmark_client")

    selected_benchmark_symbols = []
    if benchmark_client:
        symbol_map = {f"{row['symbol_name']} (ID:{row['symbol_id']})": row['symbol_id']
                      for _, row in df[df['client_id'] == benchmark_client][['symbol_id', 'symbol_name']].drop_duplicates().iterrows()}
        selected_display = st.multiselect(
            "Select Symbols for Benchmark Client", list(symbol_map.keys()), key="benchmark_symbols")
        selected_benchmark_symbols = [symbol_map[s] for s in selected_display]

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
        symbol_map = {f"{row['symbol_name']} (ID:{row['symbol_id']})": row['symbol_id']
                      for _, row in df[df['client_id'] == comp_client][['symbol_id', 'symbol_name']].drop_duplicates().iterrows()}
        selected_display = st.multiselect(
            f"Select Symbols for Compared Client {comp_client}",
            list(symbol_map.keys()),
            key=f"symbols_{comp_client}"
        )
        comparison_selections[comp_client] = [symbol_map[s]
                                              for s in selected_display]

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

            bench_name = df[(df['client_id'] == int(bench_client)) &
                            (df['symbol_id'] == int(bench_symbol))]['symbol_name'].iloc[0]
            client_name = df[(df['client_id'] == int(client_id)) &
                             (df['symbol_id'] == int(client_symbol))]['symbol_name'].iloc[0]

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

        st.markdown("###  Select Pair to View Trades")
        selected_pair_label = st.selectbox("Choose a Pair", pair_labels)

        if selected_pair_label:
            parts = selected_pair_label.split(" vs ")[0].split(
                "_") + selected_pair_label.split(" vs ")[1].split("_")
            bench_client, _, bench_symbol, client_id, _, client_symbol = parts

            bench_df = df[(df['client_id'] == int(bench_client)) &
                          (df['symbol_id'] == int(bench_symbol))]
            comp_df = df[(df['client_id'] == int(client_id)) &
                         (df['symbol_id'] == int(client_symbol))]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(
                    f"**Trades for Benchmark Client `{bench_client}` - Symbol `{bench_symbol} - COUNT: Trades {len(bench_df)}`**")
                st.dataframe(bench_df, use_container_width=True, height=300)
            with col2:
                st.markdown(
                    f"**Trades for Compared Client `{client_id}` - Symbol `{client_symbol} - COUNT: Trades {len(comp_df)}`**")
                st.dataframe(comp_df, use_container_width=True, height=300)


# def all_all_visualization(df: pd.DataFrame):
#     st.title("Top Matches")
#     top_n = st.number_input(
#         "Number of Top Matches", min_value=1, max_value=1000, value=10, step=1)

#     display_df = get_sim_data()[:top_n]
#     st.subheader("Similarity Index Table")
#     st.table(display_df)

#     score = get_sim_data(split=True)[:top_n]

#     if score is not None and not score.empty:
#         formatted_rows = []
#         pair_labels = []

#         for _, row in score.iterrows():
#             bench_client, bench_symbol = row['benchmark'].split('_')
#             client_id, client_symbol = row['candidate'].split('_')
#             sim_score = row['similarity']

#             bench_name = df[(df['client_id'] == int(bench_client)) &
#                             (df['symbol_id'] == int(bench_symbol))]['symbol_name'].iloc[0]
#             client_name = df[(df['client_id'] == int(client_id)) &
#                              (df['symbol_id'] == int(client_symbol))]['symbol_name'].iloc[0]

#             percent_score = round(float(sim_score) * 100, 2)
#             formatted_rows.append({
#                 "Benchmark Client": bench_client,
#                 "Benchmark Symbol": f"{bench_name} (ID:{bench_symbol})",
#                 "Compared Client": client_id,
#                 "Compared Symbol": f"{client_name} (ID:{client_symbol})",
#                 "Similarity Index": f"{percent_score}%",
#             })

#             pair_labels.append(
#                 f"{bench_client}_{bench_name}_{bench_symbol} vs {client_id}_{client_name}_{client_symbol}"
#             )

#         st.markdown("###  Select Pair to View Trades")
#         selected_pair_label = st.selectbox("Choose a Pair", pair_labels)

#         if selected_pair_label:
#             parts = selected_pair_label.split(" vs ")[0].split(
#                 "_") + selected_pair_label.split(" vs ")[1].split("_")
#             bench_client, _, bench_symbol, client_id, _, client_symbol = parts

#             bench_df = df[
#                 (df['client_id'] == int(bench_client)) &
#                 (df['symbol_id'] == int(bench_symbol))
#             ]
#             comp_df = df[
#                 (df['client_id'] == int(client_id)) &
#                 (df['symbol_id'] == int(client_symbol))
#             ]

#             col1, col2 = st.columns(2)
#             with col1:
#                 st.markdown(
#                     f"*Trades for Benchmark Client {bench_client} - Symbol {bench_symbol}- COUNT: Trades {len(bench_df)}*")
#                 st.dataframe(bench_df, use_container_width=True, height=300)
#             with col2:
#                 st.markdown(
#                     f"*Trades for Compared Client {client_id} - Symbol {client_symbol} - COUNT: Trades {len(comp_df)}*")
#                 st.dataframe(comp_df, use_container_width=True, height=300)

# def all_all_visualization(df: pd.DataFrame):
#     st.title("Top Matches")

#     # Input for number of top matches
#     top_n = st.number_input(
#         "Number of Top Matches", min_value=1, max_value=1000, value=10, step=1)

#     # Ascending or Descending Sort Order
#     sort_order = st.radio("Sort by Similarity", ["Descending", "Ascending"])

#     # Sort direction flag
#     ascending = sort_order == "Ascending"

#     # Fetch and sort data
#     display_df = get_sim_data().sort_values(
#         by="similarity", ascending=ascending).reset_index(drop=True)[:top_n]
#     st.subheader("Similarity Index Table")
#     st.table(display_df)

#     # Get detailed data
#     score = get_sim_data(split=True).sort_values(
#         by="similarity", ascending=ascending).reset_index(drop=True)[:top_n]

#     if score is not None and not score.empty:
#         formatted_rows = []
#         pair_labels = []

#         for _, row in score.iterrows():
#             bench_client, bench_symbol = row['benchmark'].split('_')
#             client_id, client_symbol = row['candidate'].split('_')
#             sim_score = row['similarity']

#             bench_name = df[(df['client_id'] == int(bench_client)) &
#                             (df['symbol_id'] == int(bench_symbol))]['symbol_name'].iloc[0]
#             client_name = df[(df['client_id'] == int(client_id)) &
#                              (df['symbol_id'] == int(client_symbol))]['symbol_name'].iloc[0]

#             percent_score = round(float(sim_score) * 100, 2)
#             formatted_rows.append({
#                 "Benchmark Client": bench_client,
#                 "Benchmark Symbol": f"{bench_name} (ID:{bench_symbol})",
#                 "Compared Client": client_id,
#                 "Compared Symbol": f"{client_name} (ID:{client_symbol})",
#                 "Similarity Index": f"{percent_score}%",
#             })

#             pair_labels.append(
#                 f"{bench_client}_{bench_name}_{bench_symbol} vs {client_id}_{client_name}_{client_symbol}"
#             )

#         st.markdown("### Select Pair to View Trades")
#         selected_pair_label = st.selectbox("Choose a Pair", pair_labels)

#         if selected_pair_label:
#             parts = selected_pair_label.split(" vs ")[0].split(
#                 "_") + selected_pair_label.split(" vs ")[1].split("_")
#             bench_client, _, bench_symbol, client_id, _, client_symbol = parts

#             bench_df = df[
#                 (df['client_id'] == int(bench_client)) &
#                 (df['symbol_id'] == int(bench_symbol))
#             ]
#             comp_df = df[
#                 (df['client_id'] == int(client_id)) &
#                 (df['symbol_id'] == int(client_symbol))
#             ]

#             col1, col2 = st.columns(2)
#             with col1:
#                 st.markdown(
#                     f"*Trades for Benchmark Client {bench_client} - Symbol {bench_symbol} - COUNT: {len(bench_df)}*")
#                 st.dataframe(bench_df, use_container_width=True, height=300)
#             with col2:
#                 st.markdown(
#                     f"*Trades for Compared Client {client_id} - Symbol {client_symbol} - COUNT: {len(comp_df)}*")
#                 st.dataframe(comp_df, use_container_width=True, height=300)

def all_all_visualization(df: pd.DataFrame):
    st.title("Top Matches")

    # Input for number of top matches
    top_n = st.number_input(
        "Number of Top Matches", min_value=1, max_value=1000, value=10, step=1)

    # Ascending or Descending Sort Order
    sort_order = st.radio("Sort by Similarity", ["Descending", "Ascending"])

    # Sort direction flag
    ascending = sort_order == "Ascending"

    # Fetch and sort data
    display_df = get_sim_data().sort_values(
        by="similarity", ascending=ascending).reset_index(drop=True)[:top_n]

    # Get detailed data
    score = get_sim_data(split=True).sort_values(
        by="similarity", ascending=ascending).reset_index(drop=True)[:top_n]

    if score is not None and not score.empty:
        formatted_rows = []
        pair_labels = []

        for _, row in score.iterrows():
            bench_client, bench_symbol = row['benchmark'].split('_')
            client_id, client_symbol = row['candidate'].split('_')
            sim_score = row['similarity']

            # Get symbol names
            bench_row = df[(df['client_id'] == int(bench_client))
                           & (df['symbol_id'] == int(bench_symbol))]
            client_row = df[(df['client_id'] == int(client_id))
                            & (df['symbol_id'] == int(client_symbol))]

            if bench_row.empty or client_row.empty:
                continue

            bench_name = bench_row['symbol_name'].iloc[0]
            client_name = client_row['symbol_name'].iloc[0]

            # Count trades
            bench_count = len(bench_row)
            client_count = len(client_row)

            percent_score = round(float(sim_score) * 100, 2)
            formatted_rows.append({
                "Benchmark Client": bench_client,
                "Benchmark Symbol": f"{bench_name} (ID:{bench_symbol})",
                "Benchmark Trades": bench_count,
                "Compared Client": client_id,
                "Compared Symbol": f"{client_name} (ID:{client_symbol})",
                "Compared Trades": client_count,
                "Similarity Index": f"{percent_score}%",
                "Difference in Trade Count": f"{abs(bench_count - client_count)}",
            })

            pair_labels.append(
                f"{bench_client}_{bench_name}_{bench_symbol} vs {client_id}_{client_name}_{client_symbol}"
            )

        # Display enriched table
        st.subheader("Similarity Index Table")
        st.table(pd.DataFrame(formatted_rows))

        # Pair selection dropdown
        st.markdown("### Select Pair to View Trades")
        selected_pair_label = st.selectbox("Choose a Pair", pair_labels)

        if selected_pair_label:
            parts = selected_pair_label.split(" vs ")[0].split(
                "_") + selected_pair_label.split(" vs ")[1].split("_")
            bench_client, _, bench_symbol, client_id, _, client_symbol = parts

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
                    f"*Trades for Benchmark Client {bench_client} - Symbol {bench_symbol} - COUNT: {len(bench_df)}*")
                st.dataframe(bench_df, use_container_width=True, height=300)
            with col2:
                st.markdown(
                    f"*Trades for Compared Client {client_id} - Symbol {client_symbol} - COUNT: {len(comp_df)}*")
                st.dataframe(comp_df, use_container_width=True, height=300)
