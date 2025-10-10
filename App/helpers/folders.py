from dash import html, dcc
def folder_modal(modal_id_prefix, title="Select folder"):
    """
    Returns a modal layout for selecting a folder path.
    modal_id_prefix: prefix string for ids (e.g. 'seg' -> 'seg-folder-modal')
    """
    mid = f"{modal_id_prefix}-folder-modal"
    inp = f"{modal_id_prefix}-folder-input"
    ok = f"{modal_id_prefix}-folder-ok"
    cancel = f"{modal_id_prefix}-folder-cancel"
    use_demo = f"{modal_id_prefix}-folder-demo"
    return html.Div([
        # hidden modal container; visibility toggled via style
        html.Div(id=mid, style={'display': 'none'}, children=[
            html.Div(style={
                'position': 'fixed', 'left': '50%', 'top': '50%',
                'transform': 'translate(-50%, -50%)',
                'backgroundColor': 'white', 'padding': '20px', 'boxShadow': '0 2px 8px rgba(0,0,0,0.3)',
                'zIndex': 1000, 'width': '600px', 'borderRadius': '8px'
            }, children=[
                html.H4(title),
                html.Div([
                    html.Label("Folder path (type here):"),
                    dcc.Input(id=inp, type='text', placeholder='/path/to/folder', style={'width': '100%'}),
                ], style={'marginTop': '10px'}),
                html.Div(style={'marginTop': '10px', 'display': 'flex', 'gap': '10px'}, children=[
                    html.Button("OK", id=ok, n_clicks=0),
                    html.Button("Cancel", id=cancel, n_clicks=0),
                    html.Button("Use demo path", id=use_demo, n_clicks=0, style={'marginLeft': 'auto'})
                ]),
                html.Div(id=f"{modal_id_prefix}-folder-modal-msg", style={'marginTop': '8px', 'color': 'darkred'})
            ]),
            # overlay to darken background
            html.Div(style={'position': 'fixed', 'left': 0, 'top': 0, 'width': '100%', 'height': '100%', 'backgroundColor': 'rgba(0,0,0,0.3)'}),
        ])
    ])
