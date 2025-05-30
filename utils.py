import plotly.express as px
import pandas as pd
import streamlit as st
import logging


@st.cache_data
def load_data(PARQUET_PATH='data/final_combined.parquet', symbol_join=False):
    df = pd.read_parquet(PARQUET_PATH)

    if symbol_join:
        symbol_map = pd.read_parquet(
            'data/symbols-info.parquet')[['symbol', 'symbol_id']]
        df = df.merge(symbol_map, on='symbol_id', how='left')
        df.rename(columns={'symbol': 'symbol_name'}, inplace=True)

    cols = list(df.columns)
    a, b = cols.index('symbol_name'), cols.index('broker_id')
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
    def load_csv(CSV_PATH='data/similarity_results_1.csv'):
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


def display_client_trades(col, df, le, client_name, symbol, label):
    df['role_encoded'] = le.transform(df['role'])
    df['role'] = df['role'].astype(str)

    with col:
        st.markdown(
            f"### Trades for {label} Client `{client_name}` - Symbol `{symbol}` - COUNT: `{len(df)}`"
        )
        st.dataframe(df, use_container_width=True, height=300)

        # Chart 1: Buy/Sell Price Activity (Shown at top)
        fig = px.line(
            df,
            x='entry_datetime',
            y='price',
            color='role',
            markers=True,
            title=f'Buy/Sell Price Activity Over Time ({label})',
            labels={'price': 'Trade Price', 'role': 'Trade Role'},
            hover_data=['amount', 'volume', 'price', 'role']
        )
        fig.update_traces(mode='lines+markers',
                          marker=dict(size=8, opacity=0.7))
        st.plotly_chart(fig)

        # Chart 2: Price Change %
        st.plotly_chart(px.line(
            df, x='entry_datetime', y='price_change_pct',
            title=f'Price Change % Over Time ({label})'
        ))

        # Chart 3: Hold Time Between Trades
        st.plotly_chart(px.bar(
            df, x='entry_datetime', y='secs_since_prev_trade',
            title=f'Hold Time Between Trades ({label})',
            labels={'secs_since_prev_trade': 'Hold Time (sec)'}
        ))

        # Chart 4: Cumulative Net Volume
        st.plotly_chart(px.line(
            df, x='entry_datetime', y='cum_net_vol',
            title=f'Cumulative Net Volume ({label})'
        ))

        # Chart 5: Amount Per Trade
        st.plotly_chart(px.bar(
            df, x='entry_datetime', y='amount',
            title=f'Amount Per Trade ({label})'
        ))

        # Chart 6: Role Switches
        fig = px.line(
            df,
            x='entry_datetime',
            y='is_role_switch',
            title=f'Role Switches Over Time ({label})',
            labels={'is_role_switch': 'Role Switch'}
        )
        fig.update_yaxes(range=[0, 1], tickvals=[0, 1])
        st.plotly_chart(fig)
