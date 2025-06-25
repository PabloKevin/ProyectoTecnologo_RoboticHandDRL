import re
import pandas as pd
import plotly.express as px
import plotly.io as pio      

pio.renderers.default = 'browser'

eps, scs = [], []

pattern = re.compile(
    r"Episode:\s*(\d+);"
    r"\s*Tool:\s*\[([^\]]+)\];"
    r"\s*Score:\s*\[([^\]]+)\];"
    r"\s*Action:\s*\(\[([^\]]+)\],\s*\[([^\]]+)\]\)"
)

with open('Desarrollo/simulation/Env04/logs_txt/experiment_log_30_v2.txt') as f: # log_27: fullset v2 ; log_30: trainset v2
    buffer = ''
    for i, line in enumerate(f):
        if i%2 == 0:
            buffer += line.strip()
        else:
            buffer += line.strip()
            line = buffer
            buffer = ''
            m = pattern.match(line)
            if m:
                eps.append(int(m.group(1)))
                scs.append(float(m.group(3)))

df = pd.DataFrame({'episode': eps, 'score': scs})

fig = px.line(df, x='episode', y='score', markers=True,
              title='Score por Episodio')
fig.update_layout(xaxis_title='Episodio', yaxis_title='Score')
fig.show()