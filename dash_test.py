import sys
import os
import argparse
import torch
import json
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output

sys.path.append(os.path.abspath(sys.path[0] + '/../'))
#__package__ = "deep_view_syn"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=int, default=0,
                        help='Which CUDA device to use.')
    opt = parser.parse_args()

    # Select device
    torch.cuda.set_device(opt.device)
    print("Set CUDA:%d as current device." % torch.cuda.current_device())
torch.autograd.set_grad_enabled(False)

from deep_view_syn.data.spherical_view_syn import *
from deep_view_syn.configs.spherical_view_syn import SphericalViewSynConfig
from deep_view_syn.my import netio
from deep_view_syn.my import util
from deep_view_syn.my import device
from deep_view_syn.my import view
from deep_view_syn.my.gen_final import GenFinal
from deep_view_syn.nets.modules import Sampler


datadir = None


def load_net(path):
    print(path)
    config = SphericalViewSynConfig()
    config.from_id(os.path.splitext(os.path.basename(path))[0])
    config.SAMPLE_PARAMS['perturb_sample'] = False
    net = config.create_net().to(device.GetDevice())
    netio.LoadNet(path, net)
    return net


def load_net_by_name(name):
    for path in os.listdir(datadir):
        if path.startswith(name + '@'):
            return load_net(datadir + path)
    return None


def load_views(data_desc_file) -> view.Trans:
    with open(datadir + data_desc_file, 'r', encoding='utf-8') as file:
        data_desc = json.loads(file.read())
        view_centers = torch.tensor(
            data_desc['view_centers'], device=device.GetDevice()).view(-1, 3)
        view_rots = torch.tensor(
            data_desc['view_rots'], device=device.GetDevice()).view(-1, 3, 3)
        return view.Trans(view_centers, view_rots)


scenes = {
    'gas': '__0_user_study/us_gas_all_in_one',
    'mc': '__0_user_study/us_mc_all_in_one',
    'bedroom': 'bedroom_all_in_one',
    'gallery': 'gallery_all_in_one',
    'lobby': 'lobby_all_in_one'
}

fov_list = [20, 45, 110]
res_list = [(128, 128), (256, 256), (256, 230)]
res_full = (1600, 1440)


scene = 'gas'
view_file = 'views.json'

app = dash.Dash(__name__, external_stylesheets=[
                'https://codepen.io/chriddyp/pen/bWLwgP.css'])

styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'min-height': '100px',
        'overflowX': 'scroll'
    }
}

datadir = 'data/' + scenes[scene] + '/'

fovea_net = load_net_by_name('fovea')
periph_net = load_net_by_name('periph')
gen = GenFinal(fov_list, res_list, res_full, fovea_net, periph_net,
               device=device.GetDevice())

sampler = Sampler(depth_range=(1, 50), n_samples=32, perturb_sample=False,
                  spherical=True, lindisp=True, inverse_r=True)
x = y = None

views = load_views(view_file)
print('%d Views loaded.', views.size())

view_idx = 27
center = (0, 0)

test_view = views.get(view_idx)
images = gen(center, test_view)

fig = px.imshow(util.Tensor2MatImg(images['fovea']))
fig1 = px.scatter(x=[0, 1, 2], y=[2, 0, 1])

app.layout = html.Div([
    html.H3("Drag and draw annotations"),
    html.Div(className='row', children=[
        dcc.Graph(id='image', figure=fig),  # , config=config),
        dcc.Graph(id='scatter', figure=fig1),  # , config=config),
        dcc.Slider(id='samples-slider', min=4, max=128, step=None,
                   marks={
                       4: '4',
                       8: '8',
                       16: '16',
                       32: '32',
                       64: '64',
                       128: '128',
                   },
                   value=32,
                   updatemode='drag'
                   )
    ])
])


def draw_scatter():
    global fig1
    p = torch.tensor([x, y], device=gen.layer_cams[0].c.device)
    ray_d = test_view.trans_vector(gen.layer_cams[0].unproj(p))
    ray_o = test_view.t
    raw, depths = fovea_net.sample_and_infer(ray_o, ray_d, sampler=sampler)
    colors, alphas = fovea_net.rendering.raw2color(raw, depths)

    scatter_x = (1 / depths[0]).cpu().detach().numpy()
    scatter_y = alphas[0].cpu().detach().numpy()
    scatter_color = colors[0].cpu().detach().numpy() * 255
    marker_colors = [
        i#'rgb(%d,%d,%d)' % (scatter_color[i][0], scatter_color[i][1], scatter_color[i][2])
        for i in range(scatter_color.shape[0])
    ]
    marker_colors_str = [
        'rgb(%d,%d,%d)' % (scatter_color[i][0], scatter_color[i][1], scatter_color[i][2])
        for i in range(scatter_color.shape[0])
    ]

    fig1 = px.scatter(x=scatter_x, y=scatter_y, color=marker_colors, color_continuous_scale=marker_colors_str)#, color_discrete_map='identity')
    fig1.update_traces(mode='lines+markers')
    fig1.update_xaxes(showgrid=False)
    fig1.update_yaxes(type='linear')
    fig1.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})


@app.callback(
    [Output('image', 'figure'),
     Output('scatter', 'figure')],
    [Input('image', 'clickData'),
     dash.dependencies.Input('samples-slider', 'value')]
)
def display_hover_data(clickData, samples):
    global x, y, sampler
    if clickData:
        x = clickData['points'][0]['x']
        y = clickData['points'][0]['y']

    sampler = Sampler(depth_range=(1, 50), n_samples=samples,
                      perturb_sample=False, spherical=True,
                      lindisp=True, inverse_r=True)
    if x != None and y != None:
        draw_scatter()
        fig.update_shapes(dict(visible=False))
        fig.add_shape(type="line", xref="x", yref="y",
                      x0=x, y0=y - 5,
                      x1=x, y1=y + 5,
                      line=dict(
                          color="LightSeaGreen",
                          width=3,
                      ))
        fig.add_shape(type="line", xref="x", yref="y",
                      x0=x - 5, y0=y,
                      x1=x + 5, y1=y,
                      line=dict(
                          color="LightSeaGreen",
                          width=3,
                      ))
    return fig, fig1


if __name__ == '__main__':
    app.run_server(debug=True)
