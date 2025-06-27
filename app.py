import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import numpy as np
import time
import json
import os
from helpers.blob_data_helper import BlobDataHelper

blob_data_helper = BlobDataHelper()

# ---- Initial State ----
image_index = 0
blob_index = 0
SAVE_FILE = "label_store.json"
SAVE_INTERVAL = 10
save_counter = 0

# ---- Load Label Store ----
if os.path.exists(SAVE_FILE):
    with open(SAVE_FILE, 'r') as f:
        label_store = json.load(f)
    print("[INFO] Loaded label store from disk")
else:
    label_store = {}
    print("[INFO] Initialized new label store")

# ---- Load first blob ----
volume, blob_index, image_index, edge_blob = blob_data_helper.get_blob(image_index, blob_index)
z_slices = volume.shape[0]

# ---- Initialize Dash App ----
app = dash.Dash(__name__)
app.title = "3D Blob Labeling App"

# ---- Layout ----
app.layout = html.Div(style={'maxWidth': '1200px', 'margin': '0 auto', 'fontSize': '20px'}, children=[
    html.H2(id='title', style={'textAlign': 'center'}),

    html.Div(style={'display': 'flex'}, children=[
        html.Div(style={'flex': '0 0 200px', 'paddingRight': '20px'}, children=[

            html.Div([
                html.Button("Prev Image", id='prev-image', n_clicks=0, style={'width': '90px'}),
                html.Button("Next Image", id='next-image', n_clicks=0, style={'width': '90px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '10px', 'gap': '10px'}),

            html.Div([
                html.Button("Prev Blob", id='prev-blob', n_clicks=0, style={'width': '90px'}),
                html.Button("Next Blob", id='next-blob', n_clicks=0, style={'width': '90px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '30px', 'gap': '10px'}),

            html.Div([
                html.Button("Class 1", id='class-1', n_clicks=0, style={'width': '90px'}),
                html.Button("Class 2", id='class-2', n_clicks=0, style={'width': '90px'}),
                html.Button("Class 3", id='class-3', n_clicks=0, style={'width': '90px'}),
                html.Button("Class 4", id='class-4', n_clicks=0, style={'width': '90px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px'}),

            html.Div([
                html.Button("Save Now", id='save-now', n_clicks=0, style={'width': '190px', 'marginTop': '30px'})
            ])
        ]),

        html.Div(style={'flex': '1'}, children=[
            dcc.Graph(id='image-display', style={'height': '600px'}),
            dcc.Slider(
                id='z-slider',
                min=0,
                max=z_slices - 1,
                step=1,
                value=0,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ])
    ]),

    dcc.Store(id='image-index', data=image_index),
    dcc.Store(id='blob-index', data=blob_index),
    dcc.Store(id='selected-class', data=None),
])

# ---- Unified CALLBACK ----
@app.callback(
    Output('image-display', 'figure'),
    Output('z-slider', 'max'),
    Output('title', 'children'),
    Output('image-index', 'data'),
    Output('blob-index', 'data'),
    Output('selected-class', 'data'),
    Output('class-1', 'style'),
    Output('class-2', 'style'),
    Output('class-3', 'style'),
    Output('class-4', 'style'),
    Input('z-slider', 'value'),
    Input('next-image', 'n_clicks'),
    Input('prev-image', 'n_clicks'),
    Input('next-blob', 'n_clicks'),
    Input('prev-blob', 'n_clicks'),
    Input('class-1', 'n_clicks'),
    Input('class-2', 'n_clicks'),
    Input('class-3', 'n_clicks'),
    Input('class-4', 'n_clicks'),
    Input('save-now', 'n_clicks'),
    State('image-index', 'data'),
    State('blob-index', 'data'),
    State('selected-class', 'data')
)
def update_view(z, next_img, prev_img, next_blob, prev_blob, n1, n2, n3, n4, save_now, img_idx, blob_idx, current_class):
    global save_counter
    start = time.time()
    ctx = dash.callback_context.triggered_id

    # Determine navigation
    if ctx == 'next-image':
        img_idx += 1
        blob_idx = 0
    elif ctx == 'prev-image':
        img_idx = max(0, img_idx - 1)
        blob_idx = 0
    elif ctx == 'next-blob':
        blob_idx += 1
    elif ctx == 'prev-blob':
        blob_idx -= 1

    # Save manually
    if ctx == 'save-now':
        with open(SAVE_FILE, 'w') as f:
            json.dump(label_store, f, indent=2)
        print("[INFO] Manual save completed")

    # Update class selection
    selected = current_class
    class_click_map = {
        'class-1': '1',
        'class-2': '2',
        'class-3': '3',
        'class-4': '4'
    }
    if ctx in class_click_map:
        selected = class_click_map[ctx]
        label_store.setdefault(f"img{img_idx}", {})[str(blob_idx)] = int(selected)
        save_counter += 1
        print(f"[INFO] Set label for img{img_idx}, blob {blob_idx} to class {selected}")
        if save_counter >= SAVE_INTERVAL:
            with open(SAVE_FILE, 'w') as f:
                json.dump(label_store, f, indent=2)
            print("[INFO] Autosave triggered")
            save_counter = 0

    # Load blob
    start_blob = time.time()
    volume, blob_index, image_index, edge_blob = blob_data_helper.get_blob(img_idx, blob_idx)
    print(f"[INFO] get_blob took {time.time() - start_blob:.2f}s")

    # Preselect class from store if exists
    selected = label_store.get(f"img{image_index}", {}).get(str(blob_index), None)
    print(f"[INFO] Preselected class for img{image_index}, blob {blob_index}: {selected}")

    z_slices = volume.shape[0]
    z = min(z, z_slices - 1)
    slice_img = volume[z]

    fig = px.imshow(slice_img, color_continuous_scale='gray')
    fig.update_layout(coloraxis_showscale=False)

    title = f"Zeige Zelle Nummer {blob_index} im Bild {image_index}"
    if edge_blob:
        title += " (die Zelle geht aus dem Bildrand)"

    def style(active):
        base = {'width': '90px'}
        if active:
            base['backgroundColor'] = '#007BFF'
            base['color'] = 'white'
        return base

    return (fig, z_slices - 1, title, image_index, blob_index, selected,
            style(selected == 1), style(selected == 2), style(selected == 3), style(selected == 4))

# ---- Run App ----
if __name__ == "__main__":
    app.run(debug=True)
