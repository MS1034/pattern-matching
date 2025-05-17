import pandas as pd
import streamlit as st
import logging


@st.cache_data
def load_data(PARQUET_PATH='data/new.parquet'):
    df = pd.read_parquet(PARQUET_PATH)
    cols = list(df.columns)
    a, b = cols.index('symbol_id'), cols.index('broker_id')
    cols[b], cols[a] = cols[a], cols[b]
    df = df[cols]
    return df


def display_trade_tables(df, client, symbols, benchmark=False):

    df_enriched = df[(df['client_id'] == client) &
                     (df['symbol_id'].isin(symbols))]

    if benchmark:
        st.subheader(f"Benchmark Trades: {client}")
    else:
        st.subheader(f"Trades for: {client}")

    st.dataframe(df_enriched, height=200, width=None, use_container_width=True)


@st.cache_data
def get_clients(df):
    return df['client_id'].unique().tolist()


@st.cache_data
def get_symbols(df, client_id):
    return df[df['client_id'] == client_id]['symbol_id'].unique().tolist()


def get_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    return logger


@st.cache_data
def get_sim_data(split=False):
    def load_csv(CSV_PATH='data/similarity_results.csv'):
        df = pd.read_csv(CSV_PATH)
        return df

    if split:
        return load_csv()

    sim_df = load_csv()

    sim_df[['benchmark', 'benchmarkSymbol']] = sim_df['benchmark'].str.split(
        '_', expand=True).astype(int)
    sim_df[['client', 'clientSymbol']] = sim_df['candidate'].str.split(
        '_', expand=True).astype(int)

    display_df = sim_df[['benchmark', 'benchmarkSymbol',
                         'client', 'clientSymbol', 'similarity']]
    return display_df
