#!/usr/bin/env python3
"""Quick test of WebGL visualization with large datasets."""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

# Force browser renderer
pio.renderers.default = 'browser'

# Generate test data
n_points = 50000
print(f"Generating {n_points:,} random points...")

x = np.random.randn(n_points)
y = np.random.randn(n_points)
colors = np.random.randint(0, 5, n_points)  # 5 clusters

# Create WebGL scatter
print("Creating Scattergl plot...")
fig = go.Figure(data=go.Scattergl(
    x=x,
    y=y,
    mode='markers',
    marker=dict(
        size=3,
        color=colors,
        colorscale='Viridis',
        opacity=0.7
    ),
    text=[f"Cluster {c}" for c in colors],
    hovertemplate='%{text}<extra></extra>'
))

fig.update_layout(
    title=f'WebGL Test - {n_points:,} Points',
    xaxis_title='X',
    yaxis_title='Y',
    width=900,
    height=700,
    hovermode='closest'
)

print("Opening in browser...")
fig.show()
