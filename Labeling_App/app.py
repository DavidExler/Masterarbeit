import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import numpy as np
import time
import json
import os
from helpers.blob_data_helper import BlobDataHelper, get_next_undef
from helpers.visualization_helper import normalize_with_cutoffs

blob_data_helper = BlobDataHelper()

# ---- Initial State ----
image_index = 0
blob_index = 1
offset = 0
upper_pct = 100
lower_pct = 0
SAVE_FILE = os.path.join("output", "label_store.json")
SAVE_INTERVAL = 3
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
volume, blob_index, image_index, edge_blob, rel_coords = blob_data_helper.get_blob(image_index, blob_index, offset=offset)
fullscreen, abs_coords = blob_data_helper.get_fullscreen_for_current_blob(0)

z_slices = volume.shape[0]
print("loaded first blob")
# ---- Initialize Dash App ----
app = dash.Dash(__name__)
app.title = "3D Blob Labeling App"

# ---- Layout ----
app.layout = html.Div(style={'maxWidth': '1200px', 'margin': '0 auto', 'fontSize': '20px'}, children=[
    html.H2(id='title', style={'textAlign': 'center'}),

    html.Div(style={'display': 'flex'}, children=[
        html.Div(style={'flex': '0 0 200px', 'paddingRight': '20px'}, children=[

            html.Div([
                html.Button("Vorheriges Bild", id='prev-image', n_clicks=0, style={'width': '90px'}),
                html.Button("Nächstes Bild", id='next-image', n_clicks=0, style={'width': '90px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '10px', 'gap': '10px'}),

            html.Div([
                html.Button("Vorherige Zelle", id='prev-blob', n_clicks=0, style={'width': '90px'}),
                html.Button("Nächste Zelle", id='next-blob', n_clicks=0, style={'width': '90px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'marginBottom': '30px', 'gap': '10px'}),

            html.Div([
                html.Button("Class 1", id='class-1', n_clicks=0, style={'width': '90px'}),
                html.Button("Class 2", id='class-2', n_clicks=0, style={'width': '90px'}),
                html.Button("Class 3", id='class-3', n_clicks=0, style={'width': '90px'}),
                html.Button("Class 4", id='class-4', n_clicks=0, style={'width': '90px'}),
            ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px'}),

            html.Div([
                html.Button("Nächster undefinierter", id='next-undef', n_clicks=0, style={'width': '190px', 'marginTop': '30px'})
            ]),

            html.Div([
                html.Button("Speichern", id='save-now', n_clicks=0, style={'width': '190px', 'marginTop': '30px'})
            ]),

            html.Div([
                html.Label("Untere % Abschneiden:", style={'marginRight': '5px', 'font-size': 12, 'font-family': 'Arial'}),
                dcc.Input(id='lower-pct', type='number', value=1, min=0, max=100, step=0.1, style={'width': '60px'}),
                html.Label("Obere % Abschneiden:", style={'marginRight': '5px', 'font-size': 12, 'font-family': 'Arial'}),
                dcc.Input(id='upper-pct', type='number', value=97, min=0, max=100, step=0.1, style={'width': '60px'}),
                html.Button("Normalisieren", id='normalize-btn', n_clicks=0, style={'marginTop': '10px', 'width': '190px'})
            ], style={'marginTop': '20px'}),

            html.Div([
            html.Button("Zoom In", id='zoom-in', n_clicks=0, style={'width': '90px'}),
            html.Button("Zoom Out", id='zoom-out', n_clicks=0, style={'width': '90px'})
        ], style={'display': 'flex', 'gap': '10px', 'marginTop': '20px'})


        ]),

        html.Div(style={'flex': '1', 'display': 'flex', 'flexDirection': 'row', 'gap': '20px'}, children=[
            html.Div(children=[
                dcc.Graph(id='image-display', style={'height': '600px'}),
                dcc.Slider(
                    id='z-slider',
                    min=0,
                    max=z_slices - 1,
                    step=1,
                    value=0,
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
            ], style={'flex': '1'}),

            html.Div(children=[
                dcc.Graph(id='fullscreen-display', style={'height': '600px'})
            ], style={'flex': '1'})
        ])

    ]),

    dcc.Store(id='image-index', data=image_index),
    dcc.Store(id='blob-index', data=blob_index),
    dcc.Store(id='offset', data=0),
    dcc.Store(id='selected-class', data=None),
])

print("loaded html")
# ---- Unified CALLBACK ----
@app.callback(
    Output('image-display', 'figure'),
    Output('fullscreen-display', 'figure'),
    Output('z-slider', 'max'),
    Output('title', 'children'),
    Output('image-index', 'data'),
    Output('blob-index', 'data'),
    Output('selected-class', 'data'),
    Output('class-1', 'style'),
    Output('class-2', 'style'),
    Output('class-3', 'style'),
    Output('class-4', 'style'),
    Output('offset', 'data'), 
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
    Input('next-undef', 'n_clicks'),
    Input('normalize-btn', 'n_clicks'), 
    Input('zoom-in', 'n_clicks'), 
    Input('zoom-out', 'n_clicks'), 
    State('offset', 'data'),
    State('image-index', 'data'),
    State('blob-index', 'data'),
    State('selected-class', 'data'),
    State('lower-pct', 'value'),
    State('upper-pct', 'value')
)

def update_view(
    z,                 # Input('z-slider', 'value')
    next_img,          # Input('next-image', 'n_clicks')
    prev_img,          # Input('prev-image', 'n_clicks')
    next_blob,         # Input('next-blob', 'n_clicks')
    prev_blob,         # Input('prev-blob', 'n_clicks')
    n1, n2, n3, n4,    # Inputs for class buttons
    save_now,          # Input('save-now', 'n_clicks')
    next_undef,        # Input('next-undef', 'n_clicks'),
    normalize_clicks,  # Input('normalize-btn', 'n_clicks')
    zoom_in,           # Input('zoom-in', 'n_clicks')
    zoom_out,          # Input('zoom-out', 'n_clicks')
    offset,            # State('offset', 'data') ← must come here!
    img_idx,           # State('image-index', 'data')
    blob_idx,          # State('blob-index', 'data')
    current_class,     # State('selected-class', 'data')
    lower_pct,         # State('lower-pct', 'value')
    upper_pct          # State('upper-pct', 'value')
):
    print(f"ins: {prev_img}, {next_blob}, {prev_blob},{img_idx}, {blob_idx}")
    global save_counter
    start = time.time()
    ctx = dash.callback_context.triggered_id

    # Determine navigation
    if ctx == 'next-image':
        img_idx += 1
        blob_idx = 1
        offset = 0
    elif ctx == 'prev-image':
        img_idx = max(0, img_idx - 1)
        blob_idx = 1
        offset = 0
    elif ctx == 'next-blob':
        blob_idx += 1
        offset = 0
        #print(f"[INFO] pushed next-blob Set blob Index {blob_idx}")
    elif ctx == 'prev-blob':
        blob_idx -= 1
        offset = 0
        #print(f"[INFO] pushed prev-blob Set blob Index {blob_idx}")
    elif ctx == 'zoom-in':
        offset = max(0, offset - 4)
    elif ctx == 'zoom-out':
        offset = offset + 4

    # Save manually
    if ctx == 'save-now':
        with open(SAVE_FILE, 'w') as f:
            json.dump(label_store, f, indent=2)
        print("[INFO] Manual save completed")

    if ctx == 'next-undef':
        img_idx_temp = img_idx
        blob_idx_temp = blob_idx
        img_idx, blob_idx = get_next_undef(label_store)
        if img_idx == None:
            title = "ALL DONE! :)"
            img_idx, blob_idx = img_idx_temp, blob_idx_temp

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
    #start_blob = time.time()
    print(f"[INFO] before get_blob: blob index: {blob_idx}, image index: {img_idx} ")
    volume, blob_index, image_index, edge_blob, rel_coords = blob_data_helper.get_blob(image_index=img_idx, blob_index=blob_idx, offset=offset)
    print(f"[INFO] after get_blob: blob index: {blob_index}, image index: {image_index} ")

    #print(f"[INFO] get_blob took {time.time() - start_blob:.2f}s")

    # Preselect class from store if exists
    selected = label_store.get(f"img{image_index}", {}).get(str(blob_index), None)
    print(f"[INFO] Preselected class for img{image_index}, blob {blob_index}: {selected}")

    z_slices = volume.shape[0]
    z = min(z, z_slices - 1)
    slice_img = volume[z]

    if ctx == 'normalize-btn':
        print(f"[DEBUG] Applying normalization with {lower_pct=}, {upper_pct=}")    
        slice_img = normalize_with_cutoffs(slice_img, lower_pct, upper_pct)


    fig = px.imshow(slice_img, color_continuous_scale='gray')
    fig.update_layout(coloraxis_showscale=False)
    (x_min, x_max, y_min, y_max) = rel_coords
    fig.add_shape(
        type="rect",
        x0=y_min, x1=y_max,  # note: Plotly X-axis is image Y
        y0=x_min, y1=x_max,  # and Y-axis is image X
        line=dict(color="red", width=2)
    )

    print(f"[INFO] before get_fullscreen:")
    fullscreen_img, abs_coords = blob_data_helper.get_fullscreen_for_current_blob(z)
    print(f"[INFO] after get_fullscreen")
    (x_min_abs, x_max_abs, y_min_abs, y_max_abs, z_min, z_max) = abs_coords
    
    fig_full = px.imshow(fullscreen_img, color_continuous_scale='gray')
    fig_full.update_layout(coloraxis_showscale=False)
    fig_full.add_shape(
        type="rect",
        x0=y_min_abs, x1=y_max_abs,
        y0=x_min_abs, y1=x_max_abs,
        line=dict(color="red", width=2)
    )


    title = f"Zelle Nummer {blob_index} im Bild {image_index}"
    if edge_blob:
        title += " (die Zelle geht aus dem Bildrand)"

    def style(active):
        base = {'width': '90px'}
        if active:
            base['backgroundColor'] = '#007BFF'
            base['color'] = 'white'
        return base

    print(f"retruns: {image_index}, {blob_index}")
    return (fig, fig_full, z_slices - 1, title, image_index, blob_index, selected,
            style(selected == 1), style(selected == 2), style(selected == 3), style(selected == 4), offset)

# ---- Run App ----
if __name__ == "__main__":
    print("starting App")
    app.run(debug=False, host="0.0.0.0")
    print("App run finished somehow")

