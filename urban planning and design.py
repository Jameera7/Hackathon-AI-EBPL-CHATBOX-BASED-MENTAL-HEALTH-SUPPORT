import pandas as pd
import plotly.graph_objects as go
import numpy as np

# Sample urban neighborhood data
data = {
    'Neighborhood': ['A', 'B', 'C', 'D', 'E'],
    'Population': [10000, 15000, 8000, 20000, 12000],
    'Area_km2': [2.0, 3.5, 1.5, 4.0, 2.5],
    'Green_Space_km2': [0.3, 0.5, 0.2, 0.7, 0.4],
    'Traffic_Index': [80, 90, 60, 100, 70],
    'Pollution_Index': [65, 80, 55, 90, 70]
}

df = pd.DataFrame(data)

# Compute metrics
df['Population_Density'] = df['Population'] / df['Area_km2']
df['Green_Space_Ratio'] = df['Green_Space_km2'] / df['Area_km2']
df['Green_Space_per_Person'] = df['Green_Space_km2'] / df['Population']
df['Traffic_Green_Ratio'] = df['Traffic_Index'] / (df['Green_Space_km2'] + 0.01)

x = list(range(len(df)))
neighborhoods = df['Neighborhood']

# Plot 1: Population Density using 3D scatter
fig1 = go.Figure(data=[
    go.Scatter3d(
        x=neighborhoods,
        y=[0]*len(df),
        z=df['Population_Density'],
        mode='markers+text',
        marker=dict(
            size=15,
            color=df['Population_Density'],
            colorscale='Blues',
            opacity=0.9
        ),
        text=df['Population_Density'].round(1),
        textposition="top center"
    )
])
fig1.update_layout(
    title='Population Density per Neighborhood (3D Hologram Style)',
    scene=dict(
        xaxis_title='Neighborhood',
        yaxis_title='',
        zaxis_title='People per km²',
        bgcolor='black',
    ),
    paper_bgcolor='black',
    font=dict(color='white')
)
fig1.show()

# Plot 2: Green Space Ratio using 3D scatter
fig2 = go.Figure(data=[
    go.Scatter3d(
        x=neighborhoods,
        y=[0]*len(df),
        z=df['Green_Space_Ratio'],
        mode='markers+text',
        marker=dict(
            size=15,
            color=df['Green_Space_Ratio'],
            colorscale='Greens',
            opacity=0.9
        ),
        text=df['Green_Space_Ratio'].round(2),
        textposition="top center"
    )
])
fig2.update_layout(
    title='Green Space Ratio per Neighborhood (3D Hologram Style)',
    scene=dict(
        xaxis_title='Neighborhood',
        yaxis_title='',
        zaxis_title='Green Space / Area',
        bgcolor='black',
    ),
    paper_bgcolor='black',
    font=dict(color='white')
)
fig2.show()

# Plot 3: Traffic vs Pollution using 3D scatter lines
fig3 = go.Figure()
fig3.add_trace(go.Scatter3d(
    x=neighborhoods,
    y=[0]*len(df),
    z=df['Traffic_Index'],
    mode='lines+markers+text',
    marker=dict(size=6, color='red'),
    line=dict(color='red'),
    text=df['Traffic_Index'],
    name='Traffic Index'
))
fig3.add_trace(go.Scatter3d(
    x=neighborhoods,
    y=[0.5]*len(df),
    z=df['Pollution_Index'],
    mode='lines+markers+text',
    marker=dict(size=6, color='gray'),
    line=dict(color='gray'),
    text=df['Pollution_Index'],
    name='Pollution Index'
))
fig3.update_layout(
    title='Traffic vs Pollution Index (3D Hologram Style)',
    scene=dict(
        xaxis_title='Neighborhood',
        yaxis_title='Layer',
        zaxis_title='Index Value',
        bgcolor='black',
    ),
    paper_bgcolor='black',
    font=dict(color='white')
)
fig3.show()
