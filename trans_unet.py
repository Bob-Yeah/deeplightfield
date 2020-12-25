from typing import List
import torch
import torch.nn as nn
from .pytorch_prototyping.pytorch_prototyping import *
from .my import util
from .my import device


class Encoder(nn.Module):
    def __init__(self, nf0, out_channels, input_resolution, output_sidelength):
        """
        Initialize a encoder

        :param nf0: number of outmost features
        :param out_channels: 
        :param input_resolution: [description]
        :param output_sidelength: [description]
        """
        super().__init__()

        norm = nn.BatchNorm2d

        num_down_unet = int(math.log2(output_sidelength))
        num_downsampling = int(math.log2(input_resolution)) - num_down_unet

        self.net = nn.Sequential(
            DownsamplingNet([nf0 * (2 ** i) for i in range(num_downsampling)],
                            in_channels=3,
                            use_dropout=False,
                            norm=norm),
            Unet(in_channels=nf0 * (2 ** (num_downsampling-1)),
                 out_channels=out_channels,
                 nf0=nf0 * (2 ** (num_downsampling-1)),
                 use_dropout=False,
                 max_channels=8*nf0,
                 num_down=num_down_unet,
                 norm=norm)
        )
        self.depth_downsampler = DownsamplingNet([1 for i in range(num_downsampling)],
                                                 in_channels=1,
                                                 use_dropout=False,
                                                 norm=norm)

    def forward(self, input, input_depth):
        return self.net(input), torch.round(self.depth_downsampler(input_depth))


class LatentSpaceTransformer(nn.Module):

    def __init__(self, feat_dim: int, cam_params,
                 diopter_of_layers: List[float],
                 view_positions: torch.Tensor):
        """
        Initialize a latent space transformer

        :param feat_dim: dimension of latent space
        :param cam_params: camera parameters
        :param diopter_of_layers: diopter of layers
        :param view_positions: view positions of input sparse light field
        """
        super().__init__()
        self.feat_dim = feat_dim
        self.f_cam = cam_params['f']
        self.view_positions = view_positions
        self.n_views = view_positions.size()[0]
        self.diopter_of_layers = diopter_of_layers
        self.feat_coords = util.MeshGrid(
            (feat_dim, feat_dim)).to(device.GetDevice())

    def forward(self, feats: torch.Tensor,
                feat_depths: torch.Tensor,
                novel_views: torch.Tensor) -> torch.Tensor:
        trans_feats = torch.zeros(novel_views.size()[0],
                                  feats.size()[0], feats.size()[1],
                                  feats.size()[2], feats.size()[3],
                                  device=device.GetDevice())
        for i in range(novel_views.size()[0]):
            for v in range(self.n_views):
                for l in range(len(self.diopter_of_layers)):
                    disparity = self._DisparityFromDepth(novel_views[i],
                                                         self.view_positions[v],
                                                         self.diopter_of_layers[l])
                    src_window = (
                        slice(max(0, -int(disparity[1])),
                              min(feats.size()[2], feats.size()[2] - int(disparity[1]))),
                        slice(max(0, -int(disparity[0])),
                              min(feats.size()[3], feats.size()[3] - int(disparity[0])))
                    )
                    tgt_window = (
                        slice(max(0, int(disparity[1])),
                              min(feats.size()[2], feats.size()[2] + int(disparity[1]))),
                        slice(max(0, int(disparity[0])),
                              min(feats.size()[3], feats.size()[3] + int(disparity[0])))
                    )
                    mask = (feat_depths[v] == l)[:, src_window[0], src_window[1]][0]
                    trans_feats[i, v, :, tgt_window[0], tgt_window[1]][:, mask] = \
                        feats[v, :, src_window[0], src_window[1]][:, mask]
        return trans_feats

    def _DisparityFromDepth(self, tgt_view, src_view, diopter):
        return torch.round((src_view - tgt_view) * diopter * self.f_cam * self.feat_dim)


class Decoder(nn.Module):
    def __init__(self, nf0, in_channels, input_resolution, img_sidelength):
        super().__init__()

        num_down_unet = int(math.log2(input_resolution))
        num_upsampling = int(math.log2(img_sidelength)) - num_down_unet

        self.net = [
            Unet(in_channels=in_channels,
                 out_channels=3 if num_upsampling <= 0 else 4 * nf0,
                 outermost_linear=True if num_upsampling <= 0 else False,
                 use_dropout=True,
                 dropout_prob=0.1,
                 nf0=nf0 * (2 ** num_upsampling),
                 norm=nn.BatchNorm2d,
                 max_channels=8 * nf0,
                 num_down=num_down_unet)
        ]

        if num_upsampling > 0:
            self.net += [
                UpsamplingNet(per_layer_out_ch=num_upsampling * [nf0],
                              in_channels=4 * nf0,
                              upsampling_mode='transpose',
                              use_dropout=True,
                              dropout_prob=0.1),
                Conv2dSame(nf0, out_channels=nf0 // 2,
                           kernel_size=3, bias=False),
                nn.BatchNorm2d(nf0 // 2),
                nn.ReLU(True),
                Conv2dSame(nf0 // 2, 3, kernel_size=3)
            ]

        self.net += [nn.Tanh()]
        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


class TransUnet(nn.Module):

    def __init__(self, cam_params, view_images, view_depths, view_positions, diopter_of_layers):
        super().__init__()
        nf0 = 64                # Number of features to use in the outermost layer of all U-Nets
        nf = 64                 # Number of features in the latent space
        latent_sidelength = 64  # The dimensions of the latent space
        image_sidelength = view_images.size()[2]

        self.view_images = view_images.to(device.GetDevice())
        self.view_depths = view_depths.to(device.GetDevice())
        self.n_views = view_images.size()[0]
        self.encoder = Encoder(nf0=nf0,
                               out_channels=nf,
                               input_resolution=image_sidelength,
                               output_sidelength=latent_sidelength)
        self.latent_space_transformer = LatentSpaceTransformer(feat_dim=latent_sidelength,
                                                               cam_params=cam_params,
                                                               view_positions=view_positions,
                                                               diopter_of_layers=diopter_of_layers)
        self.decoder = Decoder(nf0=nf0,
                               in_channels=nf * 4,
                               input_resolution=latent_sidelength,
                               img_sidelength=image_sidelength)

    def forward(self, novel_views):
        if self.training:
            self.feats, self.feat_depths = self.encoder(self.view_images,
                                                        self.view_depths)
        transformed_feats = self.latent_space_transformer(self.feats,
                                                          self.feat_depths,
                                                          novel_views)
        transformed_feats = torch.flatten(transformed_feats, 1, 2)
        novel_view_images = self.decoder(transformed_feats)
        return novel_view_images
