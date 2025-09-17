# DiRe Examples

This directory contains example applications demonstrating the power and versatility of DiRe for various dimensionality reduction tasks.

## Available Examples

### 1. Financial Market Microstructure Analysis

A comprehensive demonstration of using DiRe's PyTorch/PyKeOps backend for analyzing financial tick data and revealing market microstructure patterns through dimensionality reduction.

**Files:**
- `finance_tick_embedding.py` - Simple minute-bar data analysis
- `finance_tick_hierarchical.py` - Advanced tick-level data with hierarchical embedding  
- `finance_analysis_notebook.py` - Interactive notebook-ready analysis tools

**Features:**
- Real-time market data fetching from Polygon.io
- Microstructure feature extraction (volatility, liquidity, trade intensity)
- Market regime identification and anomaly detection
- Interactive visualizations with Plotly

**Quick Start:**
```python
from finance_tick_hierarchical import HierarchicalTickEmbedder

analyzer = HierarchicalTickEmbedder()
embedding = analyzer.analyze_trading_session("SPY")
```

## Installation

All examples require the base DiRe installation. Some examples have additional dependencies:

### Financial Examples
```bash
# Install DiRe with PyTorch backend (recommended for these examples)
pip install dire-rapids[pytorch]

# Additional dependencies for financial examples
pip install polygon-api-client plotly
```

## Running the Examples

### Command Line
```bash
python finance_tick_embedding.py
```

### Jupyter Notebook
```python
# Run the analysis
%run finance_tick_embedding.py

# Access the results
embedding.head()
```

### Interactive Analysis
```python
from finance_analysis_notebook import create_interactive_dashboard
dashboard = create_interactive_dashboard()
dashboard.show()
```

## Understanding the Outputs

### Financial Market Examples

The embeddings reveal several market patterns:

1. **Time-based Structure**: Market open (9:30-10:00) and close (15:30-16:00) form distinct clusters
2. **Volatility Regimes**: High volatility periods cluster together in embedding space
3. **Liquidity Patterns**: Lunch-time trading shows different microstructure characteristics
4. **News Events**: Visible as sudden jumps or outliers in the 2D space
5. **Trading Patterns**: Different types of trading activity (HFT vs institutional) separate naturally

### Interpreting the Visualizations

- **Color by Time**: Shows intraday patterns and market rhythm
- **Color by Volatility**: Identifies risk regimes and market stress
- **Color by Volume**: Reveals liquidity conditions and trading intensity
- **Animation**: Track market evolution throughout the trading day

## Extending the Examples

### Adding Custom Features

```python
def add_custom_features(data):
    # Add your domain-specific features
    data['custom_metric'] = calculate_custom_metric(data)
    return data

# Use with the analyzers
analyzer.feature_extractor = add_custom_features
```

### Custom Distance Metrics

DiRe now supports custom distance metrics for k-NN computation, useful for domain-specific similarity measures:

```python
# Financial example: Using correlation-based distance
def correlation_distance(x, y):
    # Compute 1 - correlation coefficient as distance
    x_centered = x - x.mean(dim=-1, keepdim=True)
    y_centered = y - y.mean(dim=-1, keepdim=True)
    correlation = (x_centered * y_centered).sum(-1) / (x_centered.norm(dim=-1) * y_centered.norm(dim=-1) + 1e-8)
    return 1 - correlation.abs()  # Distance based on absolute correlation

# Use with financial data analyzer
analyzer = HierarchicalTickEmbedder(metric=correlation_distance)
embedding = analyzer.analyze_trading_session("SPY")

# Or use string expressions for simple metrics
analyzer = HierarchicalTickEmbedder(
    metric='(x - y).abs().sum(-1)'  # L1 distance for robust outlier handling
)
```

### Multi-Asset Analysis

```python
# Analyze multiple assets in the same embedding space
assets = ['SPY', 'TLT', 'GLD', 'QQQ']
embeddings = {}

for asset in assets:
    embeddings[asset] = analyzer.analyze_trading_session(asset)

# Combine for cross-asset analysis
combined = pd.concat(embeddings.values(), keys=embeddings.keys())
```

### Real-Time Applications

The PyTorch backend's speed makes these examples suitable for real-time applications:

```python
# Streaming analysis
def process_live_data(ticker):
    while market_is_open():
        new_data = fetch_latest_trades(ticker)
        embedding = analyzer.embed_incremental(new_data)
        update_dashboard(embedding)
        time.sleep(1)  # Update every second
```

## Performance Considerations

- **Memory**: For large datasets (>100K points), use the hierarchical approach
- **Speed**: PyTorch backend with PyKeOps is recommended for GPU acceleration
- **Batching**: Process data in chunks for memory efficiency

## Common Use Cases

1. **Market Surveillance**: Detect unusual trading patterns
2. **Risk Management**: Identify regime changes and volatility clusters  
3. **Trading Strategy**: Find similar market conditions for backtesting
4. **Research**: Explore market microstructure relationships

## Troubleshooting

### No Data Returned
- Markets are closed on weekends/holidays
- Check your Polygon.io API key and rate limits

### Memory Issues
- Reduce `max_trades` parameter
- Use larger time windows for aggregation
- Enable GPU if available

### Visualization Issues
- For headless servers, save plots as HTML: `fig.write_html('output.html')`
- Use notebook environments for interactive plots

## Contributing

Have an interesting use case for DiRe? We welcome contributions! Please:

1. Create a new Python file with your example
2. Include clear documentation and comments
3. Add any specific dependencies to this README
4. Submit a pull request

## License

All examples are provided under the same Apache 2.0 license as the main DiRe package.

## Support

For questions about these examples:
- Open an issue on [GitHub](https://github.com/sashakolpakov/dire-jax)
- Check the main [DiRe documentation](https://sashakolpakov.github.io/dire-jax/)

## API Keys

The financial examples include a demo Polygon.io API key for testing. For production use, please obtain your own free key at [polygon.io](https://polygon.io/).
