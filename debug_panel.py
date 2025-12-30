
import panel as pn
import pandas as pd
import hvplot.pandas

print("Checking Panel version:", pn.__version__)

try:
    pn.extension('tabulator', 'echarts', 'plotly', design='material')
    print("Extension OK")
except Exception as e:
    print(f"Extension Error: {e}")

try:
    df = pd.DataFrame({'a': [1,2], 'b': [3,4]})
    plot = df.hvplot.line(x='a', y='b')
    print("Hvplot OK")
except Exception as e:
    print(f"Hvplot Error: {e}")

print("Done")
