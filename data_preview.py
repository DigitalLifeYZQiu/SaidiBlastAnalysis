#* 数据读取
import pandas as pd
import pyarrow.parquet as pq

#* 绘图
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from Analysis.DataframeAnalysis import DataframeAnalysis

root_path = 'data'
# data_path = '化验数据.parquet'
data_path = '异常炉况数据.csv'
DA = DataframeAnalysis(root_path,data_path)
import pdb
pdb.set_trace()


#* 铁水硅预测
parquet_file_test = pq.ParquetFile("data/化验数据.parquet")
data_test = parquet_file_test.read().to_pandas()
data_test_nan_time = data_test[data_test.loc[:,'F1':'OT'].isna().values==True]

parquet_file_monitor = pq.ParquetFile("data/监测数据.parquet")
data_monitor = parquet_file_monitor.read().to_pandas()
data_monitor_nan_time = data_monitor[data_monitor.loc[:,'F6':'F40'].isnull().values==True].index.unique()

#* 异常炉况预警
data_anomaly = pd.read_csv('data/异常炉况数据.csv')
anomaly_label_list = [data_anomaly[data_anomaly['flag']==1].index.tolist()]
data_nan_time = data_anomaly[data_anomaly.loc[:,'feature1':'feature23'].isnull().values==True]['time'].unique()
# for i in range(1,24):
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(
#         x = data_anomaly['time'][:1000],
#         y = data_anomaly[f'feature{i}'][:1000]
#     ))
#     fig.update_layout(
#         title=f'Anomaly Feature {i}',
#         xaxis=dict(title="time"),
#         yaxis=dict(title=f"feature {i}"),
#     )


import pdb
pdb.set_trace()

from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

df = pd.read_csv('data/异常炉况数据.csv')
anomaly_date_list = df[df['flag']==1]['time'].tolist()

app = Dash()

app.layout = [
    html.H1(children='Saidi Blast Data Analysis'),
    dcc.Dropdown(df.columns, 'feature1', id='dropdown-selection1'),
    dcc.Graph(id='graph-content1'),
]

@callback(
    Output('graph-content1', 'figure'),
    Input('dropdown-selection1', 'value')
)
def update_graph(value):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x = df['time'],
        y = df[f"{value}"],
    ))
    for dt in anomaly_date_list:
        fig.add_vline(x=dt,line_color='red',line_width=1)
    fig.update_layout(
        title=f'Anomaly Feature {value}',
        xaxis=dict(title="time"),
        yaxis=dict(title=f"feature{value}"),
    )
    return fig


if __name__ == '__main__':
    app.run(debug=True)