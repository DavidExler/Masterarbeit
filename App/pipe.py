"""
Combined Dash app: 4 tabs
- Segmentierung
- Labeling (your provided labeling UI, integrated as Tab 2)
- Methoden Vergleich
- Inference

Constraints respected:
- I integrated your labeling UI almost verbatim and kept its callbacks / logic.
  (To integrate into a single Dash app instance I replaced creation of a second Dash app
  and the `app.run` call from your snippet — that's necessary to register its callbacks
  with the single Dash server. I did NOT otherwise alter the internal logic of your labeling
  callbacks or the data-handling lines; I left your helper imports and variables intact.)

- Input/Output folder selection: each of the three pipeline tabs (Seg / Methods / Inference)
  has a "Select folder" button that opens a small dialog (modal) with either:
    * a text input for the path, OR
    * an optional "Use demo" preset.
  (Browsers cannot open native OS folder dialogs from Dash safely; this modal is the
  requested fallback and allows typing a path.)

- I added dcc.Store items to keep chosen input/output paths per-tab.

Run:
  pip install dash plotly numpy
  python combined_zell_pipeline_with_labeling_tab.py
  open http://127.0.0.1:8050

"""

from dash import html, dcc, Input, Output, State, ctx
import dash
import plotly.express as px
import numpy as np
import time
import json
import os

from dash import no_update
import threading, uuid


from helpers.blob_data_helper import BlobDataHelper, get_next_undef
from helpers.visualization_helper import normalize_with_cutoffs
from helpers.optimize import run_optimize_job
from helpers.segmentation import segment_folder
from helpers.visualization_helper import visualizer
from helpers.inference import inference_combo

# ---- Labeling code ----

blob_data_helper = BlobDataHelper()
RESULTS = {}   # job_id -> best_result

# ---- Initial State ----
image_index = 0
blob_index = 1
offset_left = 0
offset_right = 10
upper_pct = 100
lower_pct = 0
overlay_channels = [1]
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_FILE = os.path.join(BASE_DIR, "output", "label_store.json")
SAVE_INTERVAL = 3
save_counter = 0
image_size = '550px'

# ---- Load Label Store ----
label_store = {}
print("[INFO] Initialized new label store")

# ---- Load first blob (we assume helpers/mock data present) ----
volume, blob_index, image_index, edge_blob, rel_coords = blob_data_helper.get_blob(image_index, blob_index, overlay_channels, offset=offset_left)
fullscreen, abs_coords = blob_data_helper.get_fullscreen_for_current_blob(10, offset=offset_right)
z_slices = volume.shape[0]
print("loaded first blob")



# ---- Main app initialization ----
app = dash.Dash(__name__)
server = app.server
app.title = "Zell Pipeline (Seg / Label / Methods / Inference)"

# --- Layout ---------------------------------------------------------------
app.layout = html.Div([
    html.H2("3D-Zelldaten-Pipeline"),
    dcc.Tabs(id='main-tabs', value='tab-seg', children=[
        dcc.Tab(label='Segmentierung', value='tab-seg'),
        dcc.Tab(label='Labeling-App', value='tab-label'),
        dcc.Tab(label='Methodenvergleich', value='tab-methods'),
        dcc.Tab(label='Inference', value='tab-inference'),
        dcc.Tab(label='Visualisierung', value='tab-vis'),
    ]),
    html.Div(id='tab-content'),

    # Stores for chosen folders
    dcc.Store(id='seg-input-folder'),
    dcc.Store(id='seg-output-folder'),
    dcc.Store(id='label-input-folder'),
    dcc.Store(id='label-output-folder'),
    dcc.Store(id='methods-input-folder'),
    dcc.Store(id='methods-output-folder'),
    dcc.Store(id='inference-input-folder'),
    dcc.Store(id='inference-output-folder'),
    dcc.Store(id="reload-trigger", data=0)

])



# --- Tab render callback --------------------------------------------------


@app.callback(Output('tab-content', 'children'),
              Input('main-tabs', 'value'))
#def render_tab(tab):
#    print("test")
#    # minimal sanity check
#    return html.Div([html.H3(f"DEBUG tab: {tab}"), html.P("If you see this, callback runs.")])


def render_tab(tab):
    if tab == 'tab-seg':
        return html.Div([
            html.H3("1) Segmentierung"),

            html.Div([
                html.Label("Eingabe Ordner"),
                dcc.Input(id="seg-input-folder-path", type="text", value="Demo/Input", style={"width": "70%"}),
                html.Button("Auswahl", id="select-input-folder", n_clicks=0)
            ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),

            html.Div([
                html.Label("Ausgabe Ordner"),
                dcc.Input(id="seg-output-folder-path", type="text", value="Demo/Data", style={"width": "70%"}),
                html.Button("Auswahl", id="select-output-folder", n_clicks=0)
            ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),

            html.Div(id='seg-folders-display', style={'marginTop': '10px', 'fontStyle': 'italic'}),
            html.Br(),

            html.Button("Starte Segmentierung", id='seg-start-btn'),
            html.Button("Abbrechen", id='seg-abort-btn', style={'marginLeft': '10px'}),
            html.Br(), html.Br(),

            # Running state + job id store + polling interval
            dcc.Store(id="seg-running", data=False),
            dcc.Store(id="seg-job-id", data=""),
            dcc.Interval(id="seg-complete-interval", interval=1000, n_intervals=0),

            # Loading spinner + status text + result display
            dcc.Loading(id="seg-loading", children=[
                html.Div(id="seg-running-indicator", style={"marginTop":"6px","fontSize":"16px","color":"#333"})
            ], type="circle"),
            html.Div(id="seg-results-display", style={"marginTop":"6px","fontSize":"14px","color":"#006600"}),


        ], style={'padding': '20px'})

    if tab == 'tab-label':
        blob_data_helper.reload_images()
        labeling_layout = html.Div(style={'maxWidth': '1200px', 'margin': '0 auto', 'fontSize': '20px', 'fontFamily': 'inherit'}, children=[
            html.H2(id='title', style={'textAlign': 'center'}),
            html.Div([
                html.Label("Eingabe Ordner", style={'fontSize': 16}),
                dcc.Input(id="labeling-input-folder-path", type="text", value="Demo/Data", style={"width": "70%"}),
                html.Button("Auswahl", id="labeling-select-input-folder", n_clicks=0)
            ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),
            html.Div([
                html.Label("Ausgabe Ordner", style={'fontSize': 16}),
                dcc.Input(id="labeling-output-folder-path", type="text", value="Demo/Data", style={"width": "70%"}),
                html.Button("Auswahl", id="labeling-select-output-folder", n_clicks=0)
            ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),
            html.Div(id="labeling-folders-display", style={"marginTop": "10px", "fontStyle": "italic"}),

            html.Div(style={'display': 'flex'}, children=[

                html.Div(style={'flex': '0 0 300px', 'paddingRight': '10px', 'display': 'flex', 'flexDirection': 'column', 'gap': '20px'}, children=[
                    # Image navigation
                    html.Div([
                        html.Button("Previous picture", id='prev-image', n_clicks=0, style={'width': '140px'}),
                        html.Button("Next picture", id='next-image', n_clicks=0, style={'width': '140px'}),
                    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),

                    # Blob navigation
                    html.Div([
                        html.Button("Previous nucleus", id='prev-blob', n_clicks=0, style={'width': '140px'}),
                        html.Button("Next nucleus", id='next-blob', n_clicks=0, style={'width': '140px'}),
                    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),

                    # Jump to next undefined
                    # Save
                    html.Div([
                        html.Button("Next undefined", id='next-undef', n_clicks=0, style={'width': '140px'}),
                        html.Button("Save", id='save-now', n_clicks=0, style={'width': '140px'}),
                    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),


                    # Go to specific image/blob
                    html.Div([
                        html.Label("Image", style={'fontSize': 16, 'alignSelf': 'center'}),
                        dcc.Input(id='image-input', type='number', value=0, min=0, max=22, step=1, style={'width': '60px', 'fontSize': 16}),
                        html.Label("Nucleus", style={'fontSize': 16, 'alignSelf': 'center'}),
                        dcc.Input(id='blob-input', type='number', value=1, min=1, max=9999, step=1, style={'width': '60px', 'fontSize': 16}),
                        html.Button("go to...", id='goto-btn', n_clicks=0, style={'width': '280px'}),
                    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),

                    # Class selection buttons
                    html.Div([
                        html.Button("Myo", id='class-1', n_clicks=0, style={'width': '140px', 'flexShrink': '0'}),
                        html.Button("Debris", id='class-2', n_clicks=0, style={'width': '140px', 'flexShrink': '0'}),
                    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap'}),
                    html.Div([
                        html.Button("Others", id='class-3', n_clicks=0, style={'width': '140px', 'flexShrink': '0'}),
                        html.Button("Schwann", id='class-4', n_clicks=0, style={'width': '140px', 'flexShrink': '0'}),
                    ], style={'display': 'flex', 'gap': '10px', 'flexWrap': 'wrap', 'minWidth': '300px'}),


                    html.Div([
                        html.Label("Overlay Myotubes (Green)", style={'fontSize': 16}),
                        dcc.Checklist(
                            id='overlay-ch1',
                            options=[{'label': 'Enable', 'value': 'ch1'}],
                            value=[],  # empty = off
                            inline=True,
                            style={'fontSize': 16}
                        ),
                        html.Label("Overlay Marker (Blue)", style={'fontSize': 16}),
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
                                labelStyle={'display': 'inline-block', 'width': '90px', 'fontSize': 16},
                                style={'columnCount': 2}
                            )
                        ])
                    ]),
                    # Normalization controls
                    html.Div([
                        html.Div([
                            html.Div([
                                html.Label("Lower percentile", style={'fontSize': 16}),
                                dcc.Input(id='lower-pct', type='number', value=1, min=0, max=100, step=0.1, style={'width': '50px'}),
                            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '5px'}),

                            html.Div([
                                html.Label("Upper percentile", style={'fontSize': 16}),
                                dcc.Input(id='upper-pct', type='number', value=97, min=0, max=100, step=0.1, style={'width': '50px'}),
                            ], style={'display': 'flex', 'alignItems': 'center', 'gap': '5px'}),
                        ], style={'display': 'flex', 'flexDirection': 'column', 'gap': '10px'}),

                        html.Button("Normalize", id='normalize-btn', n_clicks=0, style={'height': '100%'}),
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
        return labeling_layout

    if tab == 'tab-methods':
        return html.Div([
            html.H3("3) Methodenvergleich"),
            html.Div([
                html.Label("Eingabe Ordner"),
                dcc.Input(id="methods-input-folder-path", type="text", value="Demo/Data", style={"width": "70%"}),
                html.Button("Auswahl", id="methods-select-input-folder", n_clicks=0)
            ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),
            html.Div([
                html.Label("Ausgabe Ordner"),
                dcc.Input(id="methods-output-folder-path", type="text", value="Demo/Models", style={"width": "70%"}),
                html.Button("Auswahl", id="methods-select-output-folder", n_clicks=0)
            ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),
            html.Div(id="methods-folders-display", style={"marginTop": "10px", "fontStyle": "italic"}),

            html.Br(),
            html.Label("Methoden Optimierung (Mindestens eine Methode je Gruppe)"),
            html.Div([
                html.Div([html.B('Encoder'), dcc.Checklist(['CellposeSAM', 'ResNet18', 'ResNet101', 'SwinV2', 'ConvNeXt', 'EfficientNetV2'], id='methods-group-1')]),
                html.Div([html.B('Klassifikations-Kopf'), dcc.Checklist(['Schichten-Klassifikator', 'Volumen-Klassifikator'], id='methods-group-2')]),
                html.Div([html.B('Vorverarbeitung (Nucleus Kanal)'), dcc.Checklist(['Distanztransformation', 'Segmentierungsmaske'], id='methods-group-3')]),
                html.Div([html.B('Vortraining'), dcc.Checklist(['Kein Vortraining', 'Semi-supervised', 'Fully-supervised'], id='methods-group-4')]),
            ], style={'columnCount': 2, 'gap': '10px', 'marginTop': '10px'}),
            html.Br(),
            html.Button("Starte Methodenvergleich", id='methods-start-btn'),
            html.Button("Abbrechen", id='methods-stop-btn', style={'marginLeft': '10px'}),
            html.Br(), html.Br(),

            dcc.Store(id="methods-running", data=False),
            dcc.Store(id="methods-job-id", data=""),
            dcc.Interval(id="methods-complete-interval", interval=1000, n_intervals=0),
            html.Div(id="methods-running-indicator", style={"marginTop":"6px","fontSize":"16px","color":"#333"}),
            html.Div(id="methods-best-result", style={"marginTop":"6px","fontSize":"14px","color":"#006600"}),

        ], style={'padding': '20px'})

    if tab == 'tab-inference':
        return html.Div([
            html.H3("4) Inferenz"),
            html.Div([
                html.Label("Eingabe Ordner (Bilder und Masken)", style={"width": "12%"}),
                dcc.Input(id="inference-input-folder-path-images", type="text", value="Demo/Data", style={"width": "40%"}),
                html.Button("Auswahl", id="inference-select-input-folder-images", n_clicks=0),
            ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),
            html.Div([
                html.Label("Eingabe Ordner (Modelle)", style={"width": "12%"}),
                dcc.Input(id="inference-input-folder-path", type="text", value="Demo/Models", style={"width": "40%"}),
                html.Button("Auswahl", id="inference-select-input-folder", n_clicks=0),
            ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),

            html.Div([
                html.Label("Ausgabe Ordner", style={"width": "12%"}),
                dcc.Input(id="inference-output-folder-path", type="text", value="Demo/Output", style={"width": "40%"}),
                html.Button("Auswahl", id="inference-select-output-folder", n_clicks=0),
            ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),
            html.Div(id="inference-selected-folder-display", style={"marginTop": "10px", "fontStyle": "italic"}),
            html.Br(),
            html.Button("Starte Inferenz", id='inference-start-btn'),
            html.Br(), html.Br(),

            # Stores / status / polling
            dcc.Store(id="inference-job-id", data=""),
            dcc.Interval(id="inference-complete-interval", interval=1000, n_intervals=0, disabled=True),
            html.Div(id="inference-running-indicator", style={"marginTop":"6px","fontSize":"16px","color":"#333"}),
            html.Div(id="inference-results-display", style={"marginTop":"6px","fontSize":"14px","color":"#006600"}),

        ], style={'padding': '20px'})
    if tab == 'tab-vis':
        return html.Div([
            html.H3("5) Visualisierung"),

            html.Div([
                html.Label("Eingabe Ordner"),
                dcc.Input(id="vis-input-folder-path", type="text", value="Demo/Output", style={"width": "70%"})
            ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),
            html.Div([
                html.Label("Ausgabe Ordner"),
                dcc.Input(id="vis-output-folder-path", type="text", value="Demo/Output", style={"width": "70%"})
            ], style={"display": "flex", "gap": "10px", "alignItems": "center"}),

            html.Br(),
            html.Button("Starte Visualisierung", id='vis-start-btn'),

            html.Br(), html.Br(),

            html.Div([
                dcc.Graph(id="vis-graph-volumes", style={"width": "32%"}),
                dcc.Graph(id="vis-graph-classes", style={"width": "32%"}),
                dcc.Graph(id="vis-graph-mask", style={"width": "32%"})
            ], style={"display": "flex", "gap": "10px"}),

        ], style={'padding': '20px'})
    return html.Div()

# --- Folder modal open/close & confirming handling --------------------------
# For each of seg/methods/inference we need to open modal when respective select button clicked,
# and write chosen path into the corresponding dcc.Store.

# Generic pattern: button opens modal; modal OK writes to store + closes; Cancel closes; Use demo populates demo path.
# ===== Replace the entire old folder-modal section with this =====

# --- Segmentation folders ---
@app.callback(
    Output("seg-folders-display", "children"),
    Input("seg-input-folder-path", "value"),
    Input("seg-output-folder-path", "value"),
)
def seg_update_folder_display(input_path, output_path):
    return html.Div([
        html.Div(f"📁 Eingabe Ordner: {input_path or '(not set)'}"),
        html.Div(f"📁 Output folder: {output_path or '(not set)'}")
    ])


# --- control callback for segmentation start/abort/polling ---
@app.callback(
    Output("seg-running", "data"),
    Output("seg-job-id", "data"),
    Output("seg-running-indicator", "children"),
    Output("seg-results-display", "children"),
    Output("seg-complete-interval", "disabled"),   # <--- new output
    Input("seg-start-btn", "n_clicks"),
    Input("seg-complete-interval", "n_intervals"),
    State("seg-input-folder-path", "value"),
    State("seg-output-folder-path", "value"),
    State("seg-job-id", "data"),
    prevent_initial_call=True,
)
def seg_control(start_clicks, n_intervals, in_folder, out_folder, job_id):
    trigger = ctx.triggered_id

    # --- Start segmentation ---
    if trigger == "seg-start-btn":
        if not in_folder:
            return dash.no_update, dash.no_update, "Please set Eingabe Ordner", "", True  # keep interval disabled

        job_id = str(uuid.uuid4())

        def _seg_worker(job_id, in_folder, out_folder):
            try:
                number_of_segments = segment_folder(in_folder, out_folder)
                RESULTS[job_id] = {"count": int(number_of_segments)}
            except Exception as e:
                RESULTS[job_id] = {"error": str(e)}

        threading.Thread(target=_seg_worker, args=(job_id, in_folder, out_folder), daemon=True).start()
        # start polling: enabled = False -> disabled False means interval runs
        return True, job_id, "Segmentation running…", "", False

    # --- Polling interval checks results ---
    elif trigger == "seg-complete-interval":
        if not job_id:
            return dash.no_update, dash.no_update, None, "", True
        if job_id in RESULTS:
            val = RESULTS.pop(job_id)
            if isinstance(val, dict) and "error" in val:
                # stop polling (disabled True), clear indicator
                return False, dash.no_update, None, f"Error: {val['error']}", True
            count = val.get("count", None)
            # stop polling and clear spinner (indicator None)
            return False, dash.no_update, None, f"Segments processed: {count}", True
        # still running: keep polling enabled, show running text
        return dash.no_update, dash.no_update, "Segmentation running…", "", False

    return dash.no_update, dash.no_update, None, "", True


# --- Methods/Optimization folders ---
@app.callback(
    Output("methods-folders-display", "children"),
    Input("methods-input-folder-path", "value"),
    Input("methods-output-folder-path", "value"),
)
def methods_update_folder_display(input_path, output_path):
    return html.Div([
        html.Div(f"📁 Eingabe Ordner: {input_path or '(not set)'}"),
        html.Div(f"📁 Ausgabe Ordner: {output_path or '(not set)'}")
    ])

@app.callback(
    Output("methods-best-result", "children"),
    Input("methods-start-btn", "n_clicks"),
    State("methods-group-1", "value"),
    State("methods-group-2", "value"),
    State("methods-group-3", "value"),
    State("methods-group-4", "value"),
    State("methods-input-folder-path", "value"),
    State("methods-output-folder-path", "value"),
    prevent_initial_call=True
)
def methods_control(start_clicks, g1, g2, g3, g4, in_folder, out_folder):
    if not (g1 and g2 and g3 and g4):
        return "Please select at least one method per group."

    try:
        best = run_optimize_job(g1, g2, g3, g4, in_folder, out_folder)
        return f"Best result: {best}"
    except Exception as e:
        return f"Error during training: {e}"

@app.callback(
    Output("vis-graph-volumes", "figure"),
    Output("vis-graph-classes", "figure"),
    Output("vis-graph-mask", "figure"),
    Input("vis-start-btn", "n_clicks"),
    State("vis-input-folder-path", "value"),
    State("vis-output-folder-path", "value"),
    prevent_initial_call=True
)
def vis_update(n_clicks, in_folder, out_folder):
    if not in_folder or not out_folder:
        raise dash.exceptions.PreventUpdate

    try:
        fig1, fig2, fig3 = visualizer(in_folder, out_folder)
    except Exception as e:
        # fallback: empty figures with error message
        import plotly.graph_objects as go
        fig1 = go.Figure().add_annotation(text=f"Error: {e}", x=0.5, y=0.5, showarrow=False)
        fig2 = go.Figure().add_annotation(text=f"Error: {e}", x=0.5, y=0.5, showarrow=False)
        fig3 = go.Figure().add_annotation(text=f"Error: {e}", x=0.5, y=0.5, showarrow=False)

    return fig1, fig2, fig3

# --- Inference folders ---
@app.callback(
    Output("inference-folders-display", "children"),
    Input("inference-input-folder-path", "value"),
    Input("inference-input-folder-path-images", "value"),
    Input("inference-output-folder-path", "value"),
)
def inference_update_folder_display(input_path, input_path_images, output_path):
    return html.Div([
        html.Div(f"Eingabe Ordner (Bilder und Masken): {input_path or '(not set)'}"),
        html.Div(f"Eingabe Ordner (Modelle): {input_path_images or '(not set)'}"),
        html.Div(f"Output folder: {output_path or '(not set)'}")
    ])
@app.callback(
    Output("inference-job-id", "data"),                 # 1
    Output("inference-running-indicator", "children"),  # 2
    Output("inference-results-display", "children"),    # 3
    Output("inference-complete-interval", "disabled"),  # 4
    Input("inference-start-btn", "n_clicks"),
    Input("inference-complete-interval", "n_intervals"),
    State("inference-input-folder-path", "value"),
    State("inference-input-folder-path-images", "value"),
    State("inference-output-folder-path", "value"),
    State("inference-job-id", "data"),
    prevent_initial_call=True,
)
def inference_control(start_clicks, n_intervals, in_folder, in_folder_images, out_folder, job_id):
    trigger = ctx.triggered_id

    # --- Start inference ---
    if trigger == "inference-start-btn":
        if not in_folder:
            # keep job_id unchanged, show message, keep interval disabled
            return dash.no_update, dash.no_update, "Bitte setze Eingabe Ordner", True

        job_id = str(uuid.uuid4())

        def _inference_worker(job_id, in_folder, in_folder_images, out_folder):
            try:
                best = inference_combo(in_folder, in_folder_images, out_folder)
                RESULTS[job_id] = {"best": best}
            except Exception as e:
                RESULTS[job_id] = {"error": str(e)}

        threading.Thread(target=_inference_worker, args=(job_id, in_folder, in_folder_images, out_folder), daemon=True).start()
        # set job id, show running text, clear results area, enable polling (disabled=False)
        return job_id, "Inference running…", "", False

    # --- Polling interval checks results ---
    elif trigger == "inference-complete-interval":
        if not job_id:
            # nothing running: disable interval
            return dash.no_update, None, "", True

        if job_id in RESULTS:
            val = RESULTS.pop(job_id)
            if isinstance(val, dict) and "error" in val:
                # finished with error: clear job_id, stop polling
                return "", None, f"Error: {val['error']}", True
            best = val.get("best", None)
            # success: clear job_id, stop polling, show result
            return "", None, f"Best Inference result: {best}", True

        # still running: no change to job_id, keep indicator and polling enabled
        return dash.no_update, "Inference running…", dash.no_update, False

    return dash.no_update, dash.no_update, dash.no_update, True

# --- Labeling callbacks ---------------------------------------------------
@app.callback(
    Output("labeling-folders-display", "children"),
    Output("reload-trigger", "data"),
    Input("labeling-select-input-folder", "n_clicks"),
    Input("labeling-select-output-folder", "n_clicks"),
    State("labeling-input-folder-path", "value"),
    State("labeling-output-folder-path", "value"),
)
def labeling_update_folder_display(in_click, out_click, input_path, output_path):
    global SAVE_FILE, label_store
    trigger = ctx.triggered_id
    new_store_value = 0
    if trigger == "labeling-select-input-folder":
        print(f"[INFO] Setting labeling input folder to: {input_path}")
        blob_data_helper.set_input_folder(input_path)
        new_store_value = int(time.time())
        #update_view(z=12,next_img = "initialize",prev_img = 0,next_blob = 0,prev_blob = 0,goto = 1,n1 = 0, n2 = 0, n3 = 0, n4 = 0,save_now = 0,next_undef = 0,normalize = 0,zoom_in_left = 0,zoom_in_right = 0,zoom_out_left = 0,zoom_out_right = 0,overlay_ch1 = 0,overlay_ch2to4 = 0,image_input = 0,blob_input = 1,offset_right = 0,offset_left = 0,img_idx = 1,blob_idx = 1,current_class = 1,lower_pct=1,upper_pct=99)
    if trigger == "labeling-select-output-folder":
        SAVE_FILE = os.path.join(BASE_DIR, 'helpers/data', output_path, "label_store.json")
        if os.path.exists(SAVE_FILE):
            with open(SAVE_FILE, 'r') as f:
                label_store = json.load(f)
            print(f"[INFO] Loaded label store from {SAVE_FILE}")
        else:
            label_store = {}
            os.makedirs(os.path.dirname(SAVE_FILE), exist_ok=True)
            print(f"[INFO] Initialized new label store to: {SAVE_FILE}")
            with open(SAVE_FILE, 'w') as f:
                json.dump(label_store, f, indent=2)

    return html.Div([
        html.Div(f"Eingabe Ordner: {input_path or '(not set)'}", style={"fontSize":"14px"}),
        html.Div(f"Ausgabe Ordner: {output_path or '(not set)'}", style={"fontSize":"14px"})
    ]), new_store_value

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
    Input('zoom-in-left', 'n_clicks'),
    Input('zoom-in-right', 'n_clicks'),
    Input('zoom-out-left', 'n_clicks'),
    Input('zoom-out-right', 'n_clicks'),
    Input('overlay-ch1', 'value'),
    Input('overlay-ch2to4', 'value'),
    Input("reload-trigger", "data"),

    State('image-input', 'value'),
    State('blob-input', 'value'),
    State('offset_right', 'data'),
    State('offset_left', 'data'),
    State('image-index', 'data'),
    State('blob-index', 'data'),
    State('selected-class', 'data'),
    State('lower-pct', 'value'),
    State('upper-pct', 'value'),
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
    reload_trigger,
    image_input,
    blob_input,
    offset_right,
    offset_left,
    img_idx,
    blob_idx,
    current_class,
    lower_pct,
    upper_pct
):
    global save_counter
    start = time.time()
    trigger = dash.callback_context.triggered_id

    # Determine navigation
    if trigger == 'next-image':
        img_idx += 1
        blob_idx = 1
        offset_right = 10
        offset_left = 0
    elif trigger == 'prev-image':
        img_idx = max(0, img_idx - 1)
        blob_idx = 1
        offset_right = 10
        offset_left = 0
    elif trigger == 'next-blob':
        blob_idx += 1
        offset_right = 10
        offset_left = 0
    elif trigger == 'prev-blob':
        blob_idx -= 1
        offset_right = 10
        offset_left = 0
    elif trigger == 'zoom-in-right':
        offset_right = max(0, offset_right - 12)
    elif trigger == 'zoom-out-right':
        offset_right = offset_right + 12
    elif trigger == 'zoom-in-left':
        offset_left = max(0, offset_left - 4)
    elif trigger == 'zoom-out-left':
        offset_left = offset_left + 4
    elif trigger == 'goto-btn':
        img_idx = image_input
        blob_idx = blob_input

    # Save manually
    if trigger == 'save-now':
        with open(SAVE_FILE, 'w') as f:
            json.dump(label_store, f, indent=2)
        print("[INFO] Manual save completed")

    if trigger == 'next-undef':
        img_idx_temp = img_idx
        blob_idx_temp = blob_idx
        img_idx_new, blob_idx_new = get_next_undef(label_store)
        if img_idx_new is None:
            title = "ALL DONE! :)"
            img_idx, blob_idx = img_idx_temp, blob_idx_temp
        else:
            img_idx, blob_idx = img_idx_new, blob_idx_new

    if next_img == 'initialize':
        blob_data_helper.first_load = True
    # Update class selection
    selected = current_class
    class_click_map = {
        'class-1': '1',
        'class-2': '2',
        'class-3': '3',
        'class-4': '4'
    }
    if trigger in class_click_map:
        selected = class_click_map[trigger]
        label_store.setdefault(f"img{img_idx}", {})[str(blob_idx)] = int(selected)
        save_counter += 1
        print(f"[INFO] Set label for img{img_idx}, blob {blob_idx} to class {selected}")
        if save_counter >= SAVE_INTERVAL:
            with open(SAVE_FILE, 'w') as f:
                json.dump(label_store, f, indent=2)
            print(f"[INFO] Autosave triggered to {SAVE_FILE}")
            save_counter = 0

    overlay_channels = []
    if overlay_ch1 and 'ch1' in overlay_ch1:
        overlay_channels.append(1)
    if overlay_ch2to4 == 'ch2':
        overlay_channels.append(2)
    elif overlay_ch2to4 == 'ch3':
        overlay_channels.append(3)
    elif overlay_ch2to4 == 'ch4':
        overlay_channels.append(4)

    # Load blob (calls into your helper - per your instruction we assume it's available)
    volume, blob_index, image_index, edge_blob, rel_coords = blob_data_helper.get_blob(image_index=img_idx, blob_index=blob_idx, selected_channels=overlay_channels, offset=offset_left)
    fullscreen_img, abs_coords = blob_data_helper.get_fullscreen_for_current_blob(z, offset=offset_right)

    # Preselect class from store if exists
    selected = label_store.get(f"img{image_index}", {}).get(str(blob_index), None)
    print(f"[INFO] Preselected class for img {image_index}, blob {blob_index}: {selected}")

    z_slices = volume.shape[0]
    z = min(z, z_slices - 1)
    slice_img = volume[z]

    if trigger == 'normalize-btn':
        print(f"[DEBUG] Applying normalization with {lower_pct}, {upper_pct}")
        slice_img = normalize_with_cutoffs(slice_img, lower_pct, upper_pct)
        fullscreen_img = normalize_with_cutoffs(fullscreen_img, lower_pct, upper_pct)

    print(f"[DEBUG] Update view with shapes {slice_img.shape}, {fullscreen_img.shape}")
    fig = px.imshow(slice_img)
    fig.update_layout(coloraxis_showscale=False)
    (x_min, x_max, y_min, y_max) = rel_coords
    fig.add_shape(
        type="rect",
        x0=y_min, x1=y_max,
        y0=x_min, y1=x_max,
        line=dict(color="white", width=2)
    )

    fig_full = px.imshow(fullscreen_img)
    fig_full.update_layout(coloraxis_showscale=False)
    (x_min_abs, x_max_abs, y_min_abs, y_max_abs, z_min, z_max) = abs_coords
    fig_full.add_shape(
        type="rect",
        x0=y_min_abs, x1=y_max_abs,
        y0=x_min_abs, y1=x_max_abs,
        line=dict(color="white", width=2)
    )

    title = f"Nucleus {blob_index} in picture {image_index}"
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

# ---------------------------------------------------------------------
# Run server
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("starting App")
    app.run(debug=True, host="0.0.0.0")

