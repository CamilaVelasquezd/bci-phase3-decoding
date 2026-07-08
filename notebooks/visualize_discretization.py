import zarr
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Cargar data ---
ds = zarr.open('/home/camilavelasquez/Documents/Datasets/Combined_Motor_Datasets_V9.zarr', mode='r')
session = ds['20161021']
velocity = session['processed data']['velocity'][:]
trial_id = session['processed data']['trial_id'][:]
labels = np.load('/home/camilavelasquez/labels_17class.npy')

vx = velocity[:, 0]
vy = velocity[:, 1]
speed = np.sqrt(vx**2 + vy**2)

# --- Nombres de clases ---
dir_names = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
class_names = ['Stationary'] + \
              [f'Slow {d}' for d in dir_names] + \
              [f'Fast {d}' for d in dir_names]

# --- Colores por clase ---
colors_map = {
    0: '#c0005a',  # stationary - rojo oscuro
}
slow_colors = ['#ff6b9d', '#ff7aaa', '#ff89b7', '#ff98c4', '#ffa7d1', '#ffb6de', '#ffc5eb', '#ffd4f8']
fast_colors = ['#b30051', '#990045', '#800039', '#66002e', '#4d0022', '#330017', '#1a000b', '#0d0006']
for i, c in enumerate(slow_colors):
    colors_map[i + 1] = c
for i, c in enumerate(fast_colors):
    colors_map[i + 9] = c

# --- Usar solo primeros 50000 bins para que no sea muy pesado (~50 segundos) ---
N = 50000
t = np.arange(N)
lab = labels[:N]
tid = trial_id[:N]
spd = speed[:N]

# --- Distribucion global (todos los datos) ---
class_counts = [np.sum(labels == c) for c in range(17)]
class_pcts = [100 * c / len(labels) for c in class_counts]

# --- Figura ---
fig = make_subplots(
    rows=4, cols=1,
    row_heights=[0.15, 0.35, 0.25, 0.25],
    subplot_titles=[
        'Distribucion global de clases (% de bins)',
        'Etiquetas discretizadas en el tiempo (primeros 50s)',
        'Speed del cursor',
        'Trial ID'
    ],
    vertical_spacing=0.08
)

# Panel 1: Bar chart de distribucion
fig.add_trace(go.Bar(
    x=class_names,
    y=class_pcts,
    marker_color=[colors_map[c] for c in range(17)],
    text=[f'{p:.1f}%' for p in class_pcts],
    textposition='outside',
    hovertemplate='<b>%{x}</b><br>%{y:.2f}% de bins<extra></extra>',
    name='Distribucion'
), row=1, col=1)

# Panel 2: Timeline de etiquetas
fig.add_trace(go.Scatter(
    x=t,
    y=lab,
    mode='lines',
    line=dict(color='#ff6b9d', width=0.5),
    hovertemplate='t=%{x}ms<br>clase=%{y}<br>' +
                  '<extra></extra>',
    name='Clase'
), row=2, col=1)

# Sombrear trials activos
trial_changes = np.where(np.diff(tid.astype(int)) != 0)[0]
in_trial = False
start = 0
for idx in trial_changes:
    if tid[idx] > 0 and not in_trial:
        start = idx
        in_trial = True
    elif tid[idx] <= 0 and in_trial:
        fig.add_vrect(x0=start, x1=idx,
                     fillcolor='rgba(200,200,200,0.15)',
                     layer='below', line_width=0,
                     row=2, col=1)
        in_trial = False

# Panel 3: Speed
fig.add_trace(go.Scatter(
    x=t, y=spd,
    mode='lines',
    line=dict(color='#c0005a', width=0.5),
    name='Speed',
    hovertemplate='t=%{x}ms<br>speed=%{y:.3f}<extra></extra>'
), row=3, col=1)
fig.add_hline(y=0.03, line_dash='dash', line_color='gray',
              annotation_text='0.03', row=3, col=1)
fig.add_hline(y=0.15, line_dash='dash', line_color='#ff6b9d',
              annotation_text='0.15', row=3, col=1)

# Panel 4: Trial ID
fig.add_trace(go.Scatter(
    x=t, y=tid,
    mode='lines',
    line=dict(color='#333333', width=0.5),
    name='Trial ID',
    hovertemplate='t=%{x}ms<br>trial=%{y}<extra></extra>'
), row=4, col=1)

# --- Layout ---
fig.update_layout(
    height=900,
    title=dict(
        text='Discretizacion de velocidad — Sesion 20161021 (DANDI 000688)<br>'
             '<sup>17 clases: 1 estacionario + 8 slow + 8 fast | '
             'Umbrales: 0.03 (stat) / 0.15 (slow-fast)</sup>',
        font=dict(size=14)
    ),
    showlegend=False,
    plot_bgcolor='white',
    paper_bgcolor='white',
    font=dict(family='Arial', size=11)
)

fig.update_yaxes(title_text='% bins', row=1, col=1)
fig.update_yaxes(title_text='Clase (0-16)', range=[-0.5, 16.5], row=2, col=1)
fig.update_yaxes(title_text='Speed', row=3, col=1)
fig.update_yaxes(title_text='Trial ID', row=4, col=1)
fig.update_xaxes(title_text='Tiempo (ms)', row=4, col=1)

output_path = '/home/camilavelasquez/bci-phase3-decoding/notebooks/discretization_explorer.html'
fig.write_html(output_path)
print(f"Guardado en {output_path}")
print("Abre el archivo HTML en tu navegador para verlo interactivo.")
