from detector_webcam import df
from bokeh.plotting import figure, show, output_file
import pandas as pd

df['Start'] = pd.to_datetime(df['Start'])
df['End'] = pd.to_datetime(df['End'])

if df.empty:
    print("El DataFrame está vacío. No se puede graficar.")
else:
    p = figure(x_axis_type='datetime', height=100, width=500, sizing_mode='stretch_both', title="Motion Graph")
    

    p.ygrid.visible = True

    q = p.quad(left=df["Start"], right=df["End"], bottom=0, top=1, color="green")

    output_file("Graph.html")
    show(p)
