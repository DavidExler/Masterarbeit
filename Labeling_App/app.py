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
offset_left = 0
offset_right = 10
upper_pct = 100
lower_pct = 0
overlay_channels = [1]
#upper_pct_right = 100
#lower_pct_right = 0
#upper_pct_left = 100
#lower_pct_left = 0
SAVE_FILE = os.path.join("output", "label_store.json")
SAVE_INTERVAL = 3
save_counter = 0
image_size = '550px'

# ---- Load Label Store ----
if os.path.exists(SAVE_FILE):
    with open(SAVE_FILE, 'r') as f:
        label_store = json.load(f)
    print("[INFO] Loaded label store from disk")
else:
    label_store = {}
    print("[INFO] Initialized new label store")

# ---- Load first blob ----
volume, blob_index, image_index, edge_blob, rel_coords = blob_data_helper.get_blob(image_index, blob_index, overlay_channels, offset=offset_left)
fullscreen, abs_coords = blob_data_helper.get_fullscreen_for_current_blob(10, offset=offset_right)

z_slices = volume.shape[0]
print("loaded first blob")
# ---- Initialize Dash App ----
app = dash.Dash(__name__)
app.title = "3D Blob Labeling App"

# ---- Layout ----
app.layout = html.Div(style={'maxWidth': '1200px', 'margin': '0 auto', 'fontSize': '20px'}, children=[
    html.H2(id='title', style={'textAlign': 'center'}),

    html.Div(style={'display': 'flex'}, children=[

  html.Div(style={'flex': '0 0 200px', 'paddingRight': '10px', 'display': 'flex', 'flexDirection': 'column', 'gap': '20px'}, children=[
    # Image navigation
        html.Div([
            html.Button("Vorheriges Bild", id='prev-image', n_clicks=0, style={'width': '90px'}),
            html.Button("Nächstes Bild", id='next-image', n_clicks=0, style={'width': '90px'}),
        ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),

        # Blob navigation
        html.Div([
            html.Button("Vorherige Zelle", id='prev-blob', n_clicks=0, style={'width': '90px'}),
            html.Button("Nächste Zelle", id='next-blob', n_clicks=0, style={'width': '90px'}),
        ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),

        # Jump to next undefined
        html.Button("Nächster undefinierter", id='next-undef', n_clicks=0, style={'width': '190px'}),

        # Go to specific image/blob
        html.Div([
            html.Label("Bild", style={'fontSize': 12, 'alignSelf': 'center'}),
            dcc.Input(id='image-input', type='number', value=0, min=0, max=22, step=1, style={'width': '40px'}),
            html.Label("Zelle", style={'fontSize': 12, 'alignSelf': 'center'}),
            dcc.Input(id='blob-input', type='number', value=1, min=1, max=9999, step=1, style={'width': '40px'}),
            html.Button("Gehe zu...", id='goto-btn', n_clicks=0, style={'width': '190px'}),
        ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),

        # Class selection buttons
        html.Div([
            html.Button("Myo", id='class-1', n_clicks=0, style={'width': '90px'}),
            html.Button("Deb", id='class-2', n_clicks=0, style={'width': '90px'}),
            html.Button("Others", id='class-3', n_clicks=0, style={'width': '90px'}),
            html.Button("Schwann", id='class-4', n_clicks=0, style={'width': '90px'}),
        ], style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '10px'}),

        # Save
        html.Button("Speichern", id='save-now', n_clicks=0, style={'width': '190px'}),
        
        html.Div([
            html.Label("Overlay Myotubes (Green)", style={'fontSize': 12}),
            dcc.Checklist(
                id='overlay-ch1',
                options=[{'label': 'Enable', 'value': 'ch1'}],
                value=[],  # empty = off
                inline=True,
                style={'fontSize': 12}
            ),
            html.Label("Overlay Marker (Blue)", style={'fontSize': 12}),
            html.Div([
                dcc.RadioItems(
                    id='overlay-ch2to4',
                    options=[
                        {'label': 'Channel 2', 'value': 'ch2'},
                        {'label': 'Channel 3', 'value': 'ch3'},
                        {'label': 'Channel 4', 'value': 'ch4'},
                        {'label': 'None', 'value': 'none'},
                    ],
                    value='none',
                    labelStyle={'display': 'inline-block', 'width': '90px', 'fontSize': 12},
                    style={'columnCount': 2}
                )
            ])
        ]),



        # Normalization controls
        html.Div([
            html.Div([
                html.Div([
                    html.Label("Unteres Perzentil", style={'fontSize': 12}),
                    dcc.Input(id='lower-pct', type='number', value=1, min=0, max=100, step=0.1, style={'width': '30px'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '5px'}),

                html.Div([
                    html.Label("Oberes Perzentil", style={'fontSize': 12}),
                    dcc.Input(id='upper-pct', type='number', value=97, min=0, max=100, step=0.1, style={'width': '30px'}),
                ], style={'display': 'flex', 'alignItems': 'center', 'gap': '5px'}),
            ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px'}),

            html.Button("Normalisieren", id='normalize-btn', n_clicks=0, style={'height': '100%'}),
        ], style={'display': 'flex', 'gap': '20px'})

    ]),

        # RIGHT: Graphs and Slider + Buttons below
        html.Div(style={'flex': '1'}, children=[
            # Row: Side-by-side graphs with zoom buttons
            html.Div(style={'display': 'flex', 'flexDirection': 'row', 'gap': '5px'}, children=[
                # Left Graph and Zoom Controls
                html.Div(style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'flex': '1'}, children=[
                    dcc.Graph(id='image-display', style={'height': image_size, 'width': image_size, 'flex': '1'}),
                    html.Div([
                        html.Button("Zoom In", id='zoom-in-left', n_clicks=0, style={'width': '100px', 'marginLeft': '10px'}),
                        html.Button("Zoom Out", id='zoom-out-left', n_clicks=0, style={'width': '100px'}),
                    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap', 'marginTop': '5px'})
                ]),

                # Right Graph and Zoom Controls
                html.Div(style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'flex': '1'}, children=[
                    dcc.Graph(id='fullscreen-display', style={'height': image_size, 'width': image_size, 'flex': '1'}),
                    html.Div([
                        html.Button("Zoom In", id='zoom-in-right', n_clicks=0, style={'width': '100px', 'marginLeft': '10px'}),
                        html.Button("Zoom Out", id='zoom-out-right', n_clicks=0, style={'width': '100px'}),
                    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap', 'marginTop': '5px'})
                ]),
            ]),

            # Z-slider
            dcc.Slider(
                id='z-slider',
                min=0,
                max=z_slices - 1,
                step=1,
                value=0,
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ])

    ]),

    # Store components
    dcc.Store(id='image-index', data=image_index),
    dcc.Store(id='blob-index', data=blob_index),
    dcc.Store(id='offset_right', data=10),
    dcc.Store(id='offset_left', data=0),
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
    Output('offset_right', 'data'), 
    Output('offset_left', 'data'), 

    Input('z-slider', 'value'),
    Input('next-image', 'n_clicks'),
    Input('prev-image', 'n_clicks'),
    Input('next-blob', 'n_clicks'),
    Input('prev-blob', 'n_clicks'),
    Input('goto-btn', 'n_clicks'),
    Input('class-1', 'n_clicks'),
    Input('class-2', 'n_clicks'),
    Input('class-3', 'n_clicks'),
    Input('class-4', 'n_clicks'),
    Input('save-now', 'n_clicks'),
    Input('next-undef', 'n_clicks'),
    Input('normalize-btn', 'n_clicks'),
    #Input('normalize-btn-left', 'n_clicks'),
    #Input('normalize-btn-right', 'n_clicks'),
    Input('zoom-in-left', 'n_clicks'),
    Input('zoom-in-right', 'n_clicks'),
    Input('zoom-out-left', 'n_clicks'),
    Input('zoom-out-right', 'n_clicks'),
    Input('overlay-ch1', 'value'),
    Input('overlay-ch2to4', 'value'),

    State('image-input', 'value'),
    State('blob-input', 'value'),
    State('offset_right', 'data'),
    State('offset_left', 'data'),
    State('image-index', 'data'),
    State('blob-index', 'data'),
    State('selected-class', 'data'),
    State('lower-pct', 'value'),
    State('upper-pct', 'value'),
    #State('lower-pct-left', 'value'),
    #State('upper-pct-left', 'value'),
    #State('lower-pct-right', 'value'),
    #State('upper-pct-right', 'value'),
)


def update_view(
    z,
    next_img,
    prev_img,
    next_blob,
    prev_blob,
    goto,
    n1, n2, n3, n4,
    save_now,
    next_undef,
    normalize,
    zoom_in_left,
    zoom_in_right,
    zoom_out_left,
    zoom_out_right,
    overlay_ch1,
    overlay_ch2to4,
    image_input,
    blob_input,
    offset_right,
    offset_left,
    img_idx,
    blob_idx,
    current_class,
    lower_pct,
    upper_pct
    #lower_pct_left,
    #upper_pct_left,
    #lower_pct_right,
    #upper_pct_right
):

    global save_counter
    start = time.time()
    ctx = dash.callback_context.triggered_id

    # Determine navigation
    if ctx == 'next-image':
        img_idx += 1
        blob_idx = 1
        offset_right = 10
        offset_left = 0
    elif ctx == 'prev-image':
        img_idx = max(0, img_idx - 1)
        blob_idx = 1
        offset_right = 10
        offset_left = 0
    elif ctx == 'next-blob':
        blob_idx += 1
        offset_right = 10
        offset_left = 0
        #print(f"[INFO] pushed next-blob Set blob Index {blob_idx}")
    elif ctx == 'prev-blob':
        blob_idx -= 1
        offset_right = 10
        offset_left = 0
        #print(f"[INFO] pushed prev-blob Set blob Index {blob_idx}")
    elif ctx == 'zoom-in-right':
        offset_right = max(0, offset_right - 12)
    elif ctx == 'zoom-out-right':
        offset_right = offset_right + 12
    elif ctx == 'zoom-in-left':
        offset_left = max(0, offset_left - 4)
    elif ctx == 'zoom-out-left':
        offset_left = offset_left + 4
    elif ctx == 'goto-btn':
        img_idx = image_input
        blob_idx = blob_input

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

    overlay_channels = []
    if 'ch1' in overlay_ch1:
        overlay_channels.append(1)
    if overlay_ch2to4 == 'ch2':
        overlay_channels.append(2)
    elif overlay_ch2to4 == 'ch3':
        overlay_channels.append(3)
    elif overlay_ch2to4 == 'ch4':
        overlay_channels.append(4)

    # Load blob
    #start_blob = time.time()
    #print(f"[INFO] before get_blob: blob index: {blob_idx}, image index: {img_idx} ")
    volume, blob_index, image_index, edge_blob, rel_coords = blob_data_helper.get_blob(image_index=img_idx, blob_index=blob_idx, selected_channels=overlay_channels, offset=offset_left)
    #print(f"[INFO] after get_blob: blob index: {blob_index}, image index: {image_index} ")

    #print(f"[INFO] get_blob took {time.time() - start_blob:.2f}s")

    # Preselect class from store if exists
    selected = label_store.get(f"img{image_index}", {}).get(str(blob_index), None)
    print(f"[INFO] Preselected class for img {image_index}, blob {blob_index}: {selected}")

    z_slices = volume.shape[0]
    z = min(z, z_slices - 1)
    slice_img = volume[z]

    #print(f"[INFO] before get_fullscreen:")
    fullscreen_img, abs_coords = blob_data_helper.get_fullscreen_for_current_blob(z, offset=offset_right)
    #print(f"[INFO] after get_fullscreen")
    (x_min_abs, x_max_abs, y_min_abs, y_max_abs, z_min, z_max) = abs_coords

    if ctx == 'normalize-btn':
        print(f"[DEBUG] Applying normalization with {lower_pct}, {upper_pct}")    
        slice_img = normalize_with_cutoffs(slice_img, lower_pct, upper_pct)
        fullscreen_img = normalize_with_cutoffs(fullscreen_img, lower_pct, upper_pct)


    fig = px.imshow(slice_img)
    fig.update_layout(coloraxis_showscale=False)
    (x_min, x_max, y_min, y_max) = rel_coords
    fig.add_shape(
        type="rect",
        x0=y_min, x1=y_max,  # note: Plotly X-axis is image Y
        y0=x_min, y1=x_max,  # and Y-axis is image X
        line=dict(color="white", width=2)
    )

    fig_full = px.imshow(fullscreen_img)
    fig_full.update_layout(coloraxis_showscale=False)
    fig_full.add_shape(
        type="rect",
        x0=y_min_abs, x1=y_max_abs,
        y0=x_min_abs, y1=x_max_abs,
        line=dict(color="white", width=2)
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
    return (fig, fig_full, z_slices - 1, title, image_index, blob_index, selected,
            style(selected == 1), style(selected == 2), style(selected == 3), style(selected == 4), offset_right, offset_left)

# ---- Run App ----
if __name__ == "__main__":
    print("starting App")
    app.run(debug=False, host="0.0.0.0")
    print("App run finished somehow")

