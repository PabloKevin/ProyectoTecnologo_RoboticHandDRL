import re
import pandas as pd
import plotly.express as px
import plotly.io as pio      
import plotly.graph_objects as go
import plotly.graph_objects as go

pio.renderers.default = 'browser'

"""eps, scs = [], []

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

df = pd.DataFrame({'episode': eps, 'score': scs})"""

df = pd.read_csv('Desarrollo/Documentacion/actor/train_set/logs_score30.csv')

# supongamos que ya tienes tu DataFrame df con columnas 'episode' y 'score'
# smoothing_factor en [0,1], cuanto más cerca de 1 más peso al pasado (más suavizado)
smoothing_factor = 0.92

#  alpha = 1 - smoothing_factor 
#df['score_ewm'] = df['score'].ewm(alpha=1-smoothing_factor, adjust=False).mean()
df['score_ewm'] = df['Value'].ewm(alpha=1-smoothing_factor, adjust=False).mean()


def plot_scores_vs_episodes(df, save_dir=None, save=False, show=True):
    # ahora dibujamos raw vs. smoothed
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['Step'], y=df['Value'],
        mode='lines',
        name='raw',
        line=dict(color="#EE4E0F", width=1),
        opacity=0.3
    ))
    fig.add_trace(go.Scatter(
        x=df['Step'], y=df['score_ewm'],
        mode='lines',
        name=f'smoothed (α={1-smoothing_factor:.2f})',
        line=dict(color="#E9612B", width=2)
    ))



    # Lista de episodios a marcar
    highlight_eps = [1500, 4000, 6500, 25000]

    # Supongamos que tu DataFrame se llama df y tiene columnas 'episode' y 'score'
    # Si quisieras el suavizado en su lugar, cambia 'score' por 'score_ewm'
    val_map = df.set_index('Step')['score_ewm'].to_dict()
    #print(val_map[4000])

    # 1) Recolecta todos los puntos de una sola vez:
    xs, ys = [], []
    for ep in highlight_eps:
        if ep in val_map:
            xs.append(ep)
            ys.append(val_map[ep])

    # 2) Añade un único scatter con todos los highlights
    fig.add_trace(go.Scatter(
        x=xs,
        y=ys,
        mode='markers',
        marker=dict(color="#F5B02F", size=15),
        name='Puntos de verificación'
    ))

    # 3) (Opcional) si quieres las etiquetas como texto junto al marker:
    annotations = []
    for i, ep in enumerate(highlight_eps):
        val = val_map.get(ep)
        if val is None:
            continue
        annotations.append(dict(
            x=ep,
            y=val,
            text=f"Episodio: {ep}; Score (smoothed): {val:.2f}",
            showarrow=True,
            arrowhead=0,        # estilo de punta (0–8)
            arrowsize=1,        # escala de la punta
            arrowwidth=3,       # grosor de la línea
            arrowcolor="#F5B02F",  # color de la flecha
            ax=0,              # desplaza la cola 40px a la derecha
            ay=300,             # desplaza la cola 30px hacia arriba
            xanchor='left' if i<3 else 'right',
            yanchor='bottom',
            font=dict(color='white', size=23),
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1,
            borderpad=4
        ))


    # Ajustes de estilo para fondo y ejes
    fig.update_layout(
        title='Actor TD3 - Train Set V2',
        paper_bgcolor='#303030',   # fondo general
        plot_bgcolor='#303030',    # fondo del área de dibujo
        font=dict(color='white', size=25),  # texto en blanco
        xaxis=dict(
            title='Episodio',
            gridcolor='#666666',    # rejilla más clara
            zerolinecolor='#444444' # línea cero opcional
        ),
        yaxis=dict(
            title='Score',
            gridcolor='#666666',
            zerolinecolor='#444444'
        ),
        annotations=annotations
    )
    fig.update_yaxes(range=[-30, -5])
    fig.update_xaxes(range=[0, 25200])

    if save:
        fig.write_image(save_dir, width=1920, height=1080)
    
    if show:
        fig.show()


if __name__ == "__main__":
    path = 'Desarrollo/Documentacion/actor/train_set/plot/'
    #for i in range(df.shape[0]+1):
    #    plot_scores_vs_episodes(df.iloc[:i], save_dir=path+f"sc_vs_ep_{df['Step'].iloc[i]}.png",save=True, show=False)

    plot_scores_vs_episodes(df)