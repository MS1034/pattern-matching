# The Model

The model we are using for our analysis is the LSTM model for time series embeddings, and then building our own custom distance engine to calculate the distance between two trading sequences that are essentially temporal in nature. What this means, is that the trading data is essentially segregated over a period of 16 years (onwards from 2009) in the training dataset.

## The Problem in the Dataset and How we Solve it

The principal problem in the dataset is not the data, but its nature. The very domain of the dataset is financial trades, which makes it a day-to-day dataset. The drawback is the unlabelled nature of the dataset. This causes a problem that
"how do we define the *anomalous behavior*?"

Recall the anomalous behavior is defined as a group of malicious clients in the market that manipulate a security (or group thereof) to have an unjust effect.

To solve this problem, we rewrite the problem statement itself. Instead of assuming outright the definition of anomalous behavior, we pick up two (or more) clients from the dataset, and check how much their behavior is same. This defines anomaly as a measure of how much two (or more) behaviors are related. This leaves the definition of '*anomaly*' to the surveillance manager, thereby filling a critical hole in our ML model's baseline assumption.

# Our Thought Process and How we Arrived at an Implementation

First things first, we need to define some constraints of the problem. Considering `n` clients having at most `m` symbols, the worst case benchmark combinations will be of the complexity `O(n*m!)`. (*you're welcome to do the math!*)

This is the worst nightmare for us. We needed to reduce the benchmark's size on the runtime (you can use the unbounded benchmark size on the pre-compiled training results, but if the surveillance manager demands runtime evaluation, you can't possibly run the job on every possible benchmark-compare combination). RECALL that the benchmark and clients are both tuples of {ClientId, Relevant Symbols S_1, S_2,....S_m}.

Based on discussions with the surveillance manager, we limit the m! factorial part of the problem to just 2 on compile time (the comparison is 1CId x 1Symbol pair vs 1CId x 1SymbolId). This makes the analysis part straightforward.

After the problem defintion, we move on to what approaches can work on this type of problem. The first thing that popped into our mind was a clustering based approach (either K-Means or DBScan Clustering). Why it doesn't work is because K-Means segregates the data into pre-told `k` number of classes. This would be anomalous and not anomalous in practice. But our problem asks us to tell how similar two clients are in their nuances. The definition of anomaly comes at a very later stage, and even then it is simply how much two behaviors match.

Considering a problem of this stage, parallelization pops into our mind. Literature suggests an approach based on Transformers, where we leverage an AnomalyDetector Transformer. Why we discard this is because our problem doesn't deal with detecting anomalies in the first place. It deals with checking how two behaviors are related.

# Our Solution - Distance Engine

Based on the above information, we devised a new approach to checking the behavior. If the trading behavior could be represented as a function of sequences, we can use some king of scoring system to see how close are two sequences. An important point to consider in these abstractions is that the data is temporal. If we are converting it into a sequence based, we need to preserve this trait. (For example, benchmark can be of 3 years and we are asked to see comparison of 3 month activity to see if it is according to the mark).

## Some Insights From the Data

As the `sequence_encoder.py` suggests, we extract some more features from the data itself. Some of these features are lag based and some are not. The main thing is to transform the time-series dataset to a compatible numpy array format.

```python-repl
sequence_cols = ['role_encoded', 'price',
                 'volume', 'price_change_pct', 
		 'cum_net_vol'
]
feature_cols = [
        'price', 'volume', 'secs_since_prev_trade', 'price_change_pct',
        'cum_net_vol', 'is_role_switch', 'role_encoded', 'amount'
]
```

## The Embedding Generator

From these sequences, the next thing is to generate embeddings. We use LSTM Embeddings for this, because LSTM, by its inherent nature, preserves temporal aspect of the data. The LSTM embedding generator is composed of two parts, static and sequential input layers.

### Static Input Layer

These are aggregated features defined in `feature_cols`.

### Sequence Input Layer

These are time-series based features defined in `sequence_cols`. We use a Multi-Head Attention Mechanism Layer after the input layer, followed by a 64-dimensional LSTM output layer.

### Output

The output of these layers is a concatenated static+sequence embeddings in a 128-dim Dense layers, followed by a 64-dim output layer.

## Distance Engine

This approach builds on our EmbeddingGenerator Approach.

Once we have embeddings of the trading behavior (essentially Dense representation of trades), we calculate the distance between these sequences. We use two types of scores to rank (their weightage is a hyper-parameter to the distance engine):

1. Cosine Similarity
2. DTW Score

The rationale to include DTW is that two sequences may have different speeds, and in fact they do, the factor of overall similarity is still present. For example, a trader sells, buys, holds, sells. This pattern is over 2 days. The second one does the same in a matter of 4 hours. Ignoring the factor of time, the overall behavior is same (*its exact definition then goes to other features as defined above*).

# Demonstration

You can run the app and see the results of our work.
