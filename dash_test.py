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

#sys.path.append(os.path.abspath(sys.path[0] + '/../'))
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

from data.spherical_view_syn import *
from configs.spherical_view_syn import SphericalViewSynConfig
from my import netio
from my import util
from my import device
from my import view
from my.gen_final import GenFinal
from nets.modules import Sampler


datadir = 'data/__0_user_study/us_gas_periph_r135x135_t0.3_2021.01.16/'
data_desc_file = 'train.json'
net_config = 'periph_rgb@msl-rgb_e10_fc96x4_d1.00-50.00_s16'
net_path = datadir + net_config + '/model-epoch_200.pth'
fov = 45
res = (256, 256)
view_idx = 4
center = (0, 0)


def load_net(path):
    print(path)
    config = SphericalViewSynConfig()
    config.from_id(net_config)
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


cam = view.CameraParam({
    'fov': fov,
    'cx': 0.5,
    'cy': 0.5,
    'normalized': True
}, res, device=device.GetDevice())
net = load_net(net_path)
sampler = Sampler(depth_range=(1, 50), n_samples=32, perturb_sample=False,
                  spherical=True, lindisp=True, inverse_r=True)
x = y = None

views = load_views(data_desc_file)
print('%d Views loaded.' % views.size()[0])

test_view = views.get(view_idx)
rays_o, rays_d = cam.get_global_rays(test_view, True)
image = net(rays_o.view(-1, 3), rays_d.view(-1, 3)).view(1,
                                                         res[0], res[1], -1).permute(0, 3, 1, 2)


styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'min-height': '100px',
        'overflowX': 'scroll'
    }
}
fig = px.imshow(util.Tensor2MatImg(image))
fig1 = px.scatter(x=[0, 1, 2], y=[2, 0, 1])
fig2 = px.scatter(x=[0, 1, 2], y=[2, 0, 1])
app = dash.Dash(__name__, external_stylesheets=[
                'https://codepen.io/chriddyp/pen/bWLwgP.css'])
app.layout = html.Div([
    html.H3("Drag and draw annotations"),
    html.Div(className='row', children=[
        dcc.Graph(id='image', figure=fig),  # , config=config),
        dcc.Graph(id='scatter', figure=fig1),  # , config=config),
        dcc.Graph(id='scatter1', figure=fig2),  # , config=config),
        dcc.Slider(id='samples-slider', min=4, max=128, step=None,
                   marks={
                       4: '4',
                       8: '8',
                       16: '16',
                       32: '32',
                       64: '64',
                       128: '128',
                   },
                   value=33,
                   updatemode='drag'
                   )
    ])
])


def raw2alpha(raw, dists, act_fn=torch.relu):
    """
    Function for computing density from model prediction.
    This value is strictly between [0, 1].
    """
    print('act_fn: ', act_fn(raw))
    print('act_fn * dists: ', act_fn(raw) * dists)
    return -torch.exp(-act_fn(raw) * dists) + 1.0


def raw2color(raw: torch.Tensor, z_vals: torch.Tensor):
    """
    Raw value inferred from model to color and alpha

    :param raw ```Tensor(N.rays, N.samples, 2|4)```: model's output
    :param z_vals ```Tensor(N.rays, N.samples)```: integration time
    :return ```Tensor(N.rays, N.samples, 1|3)```: color
    :return ```Tensor(N.rays, N.samples)```: alpha
    """

    # Compute 'distance' (in time) between each integration time along a ray.
    # The 'distance' from the last integration time is infinity.
    # dists: (N_rays, N_samples)
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    last_dist = z_vals[..., 0:1] * 0 + 1e10

    dists = torch.cat([dists, last_dist], -1)
    print('dists: ', dists)

    # Extract RGB of each sample position along each ray.
    color = torch.sigmoid(raw[..., :-1])  # (N_rays, N_samples, 1|3)
    alpha = raw2alpha(raw[..., -1], dists)

    return color, alpha


def draw_scatter():
    global fig1, fig2
    p = torch.tensor([x, y], device=device.GetDevice())
    ray_d = test_view.trans_vector(cam.unproj(p))
    ray_o = test_view.t
    raw, depths = net.sample_and_infer(ray_o, ray_d, sampler=sampler)
    colors, alphas = raw2color(raw, depths)

    scatter_x = (1 / depths[0]).cpu().detach().numpy()
    scatter_y = alphas[0].cpu().detach().numpy()
    scatter_y1 = raw[0, :, 3].cpu().detach().numpy()
    scatter_color = colors[0].cpu().detach().numpy() * 255
    marker_colors = [
        # 'rgb(%d,%d,%d)' % (scatter_color[i][0], scatter_color[i][1], scatter_color[i][2])
        i
        for i in range(scatter_color.shape[0])
    ]
    marker_colors_str = [
        'rgb(%d,%d,%d)' % (scatter_color[i][0],
                           scatter_color[i][1], scatter_color[i][2])
        for i in range(scatter_color.shape[0])
    ]

    fig1 = px.scatter(x=scatter_x, y=scatter_y, color=marker_colors,
                      color_continuous_scale=marker_colors_str)  # , color_discrete_map='identity')
    fig1.update_traces(mode='lines+markers')
    fig1.update_xaxes(showgrid=False)
    fig1.update_yaxes(type='linear')
    fig1.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})

    fig2 = px.scatter(x=scatter_x, y=scatter_y1, color=marker_colors,
                      color_continuous_scale=marker_colors_str)  # , color_discrete_map='identity')
    fig2.update_traces(mode='lines+markers')
    fig2.update_xaxes(showgrid=False)
    fig2.update_yaxes(type='linear')
    fig2.update_layout(height=225, margin={'l': 20, 'b': 30, 'r': 10, 't': 10})


@app.callback(
    [Output('image', 'figure'),
     Output('scatter', 'figure'),
     Output('scatter1', 'figure')],
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
    return fig, fig1, fig2


if __name__ == '__main__':
    app.run_server(debug=True)
