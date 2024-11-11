from detector_webcam import df
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool, ColumnDataSource
import pandas as pd


if 'Start' not in df.columns or 'End' not in df.columns:
    print("El DataFrame no contiene las columnas 'Start' y 'End'.")
else:
    df['Start'] = pd.to_datetime(df['Start'])
    df['End'] = pd.to_datetime(df['End'])

    if df.empty:
        print("El DataFrame está vacío. No se puede graficar.")
    else:
        cds = ColumnDataSource(df)
        p = figure(x_axis_type='datetime', height=300, width=500, sizing_mode='stretch_both', title="Motion Graph")

        p.ygrid.visible = True

        hover = HoverTool(tooltips=[("Start", "@Start{%F %T}"), ("End", "@End{%F %T}")], formatters={'@Start': 'datetime', '@End': 'datetime'})
        p.add_tools(hover)

        q = p.quad(left="Start", right="End", bottom=0, top=1, color="green", source=cds)

        output_file("Graph.html")
        show(p)
