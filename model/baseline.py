import torch
def var_or_cuda(x):
    if torch.cuda.is_available():
        # x = x.cuda(non_blocking=True)
        x = x.to('cuda:1')
    return x

class residual_block(torch.nn.Module):
    def __init__(self, OUT_CHANNELS_RB, delta_channel_dim,KERNEL_SIZE_RB,RNN=False):
        super(residual_block,self).__init__()
        self.delta_channel_dim = delta_channel_dim
        self.out_channels_rb = OUT_CHANNELS_RB
        self.hidden = None
        self.RNN = RNN
        if self.RNN:
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d((OUT_CHANNELS_RB+delta_channel_dim)*2,OUT_CHANNELS_RB+delta_channel_dim,KERNEL_SIZE_RB,stride=1,padding = 1),
                torch.nn.BatchNorm2d(OUT_CHANNELS_RB+delta_channel_dim),
                torch.nn.ELU()
            )
            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(OUT_CHANNELS_RB+delta_channel_dim,OUT_CHANNELS_RB+delta_channel_dim,KERNEL_SIZE_RB,stride=1,padding = 1),
                torch.nn.BatchNorm2d(OUT_CHANNELS_RB+delta_channel_dim),
                torch.nn.ELU()
            )
        else:
            self.layer1 = torch.nn.Sequential(
                torch.nn.Conv2d(OUT_CHANNELS_RB+delta_channel_dim,OUT_CHANNELS_RB+delta_channel_dim,KERNEL_SIZE_RB,stride=1,padding = 1),
                torch.nn.BatchNorm2d(OUT_CHANNELS_RB+delta_channel_dim),
                torch.nn.ELU()
            )
            self.layer2 = torch.nn.Sequential(
                torch.nn.Conv2d(OUT_CHANNELS_RB+delta_channel_dim,OUT_CHANNELS_RB+delta_channel_dim,KERNEL_SIZE_RB,stride=1,padding = 1),
                torch.nn.BatchNorm2d(OUT_CHANNELS_RB+delta_channel_dim),
                torch.nn.ELU()
            )
        
    def forward(self,input):
        if self.RNN:
            # print("input:",input.shape,"hidden:",self.hidden.shape)
            inp = torch.cat((input,self.hidden),dim=1)
            # print(inp.shape)
            output = self.layer1(inp)
            output = self.layer2(output)
            output = input+output
            self.hidden = output
        else:
            output = self.layer1(input)
            output = self.layer2(output)
            output = input+output
        return output

    def reset_hidden(self, inp):
        size = list(inp.size())
        size[1] = self.delta_channel_dim + self.out_channels_rb
        size[2] = size[2]//2
        size[3] = size[3]//2
        hidden = torch.zeros(*(size))
        self.hidden = var_or_cuda(hidden)

class deinterleave(torch.nn.Module):
    def __init__(self, block_size):
        super(deinterleave, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, d_height, d_width, d_depth) = output.size()
        s_depth = int(d_depth / self.block_size_sq)
        s_width = int(d_width * self.block_size)
        s_height = int(d_height * self.block_size)
        t_1 = output.reshape(batch_size, d_height, d_width, self.block_size_sq, s_depth)
        spl = t_1.split(self.block_size, 3)
        stack = [t_t.reshape(batch_size, d_height, s_width, s_depth) for t_t in spl]
        output = torch.stack(stack,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size, s_height, s_width, s_depth)
        output = output.permute(0, 3, 1, 2)
        return output

class interleave(torch.nn.Module):
    def __init__(self, block_size):
        super(interleave, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, s_height, s_width, s_depth) = output.size()
        d_depth = s_depth * self.block_size_sq
        d_width = int(s_width / self.block_size)
        d_height = int(s_height / self.block_size)
        t_1 = output.split(self.block_size, 2)
        stack = [t_t.reshape(batch_size, d_height, d_depth) for t_t in t_1]
        output = torch.stack(stack, 1)
        output = output.permute(0, 2, 1, 3)
        output = output.permute(0, 3, 1, 2)
        return output

class model(torch.nn.Module):
    def __init__(self,FIRSST_LAYER_CHANNELS,LAST_LAYER_CHANNELS,OUT_CHANNELS_RB,KERNEL_SIZE,KERNEL_SIZE_RB,INTERLEAVE_RATE):
        super(model, self).__init__()
        self.interleave = interleave(INTERLEAVE_RATE)

        self.first_layer = torch.nn.Sequential(
            torch.nn.Conv2d(FIRSST_LAYER_CHANNELS,OUT_CHANNELS_RB,KERNEL_SIZE,stride=1,padding=1),
            torch.nn.BatchNorm2d(OUT_CHANNELS_RB),
            torch.nn.ELU()
        )
        
        self.residual_block1 = residual_block(OUT_CHANNELS_RB,0,KERNEL_SIZE_RB,False)
        self.residual_block2 = residual_block(OUT_CHANNELS_RB,3,KERNEL_SIZE_RB,False)
        self.residual_block3 = residual_block(OUT_CHANNELS_RB,3,KERNEL_SIZE_RB,True)
        self.residual_block4 = residual_block(OUT_CHANNELS_RB,3,KERNEL_SIZE_RB,True)
        self.residual_block5 = residual_block(OUT_CHANNELS_RB,3,KERNEL_SIZE_RB,True)

        self.output_layer = torch.nn.Sequential(
            torch.nn.Conv2d(OUT_CHANNELS_RB+3,LAST_LAYER_CHANNELS,KERNEL_SIZE,stride=1,padding=1),
            torch.nn.BatchNorm2d(LAST_LAYER_CHANNELS),
            torch.nn.Sigmoid()
        )
        self.deinterleave = deinterleave(INTERLEAVE_RATE)

    def reset_hidden(self,inp):
        self.residual_block3.reset_hidden(inp)
        self.residual_block4.reset_hidden(inp)
        self.residual_block5.reset_hidden(inp)

    def forward(self, lightfield_images, focal_length, gazeX, gazeY):
        # lightfield_images: torch.Size([batch_size, channels * D, H, W]) 
        # channels : RGB*D: 3*9, H:256, W:256
        # print("lightfield_images:",lightfield_images.shape)
        input_to_net = self.interleave(lightfield_images)
        # print("after interleave:",input_to_net.shape)
        input_to_rb = self.first_layer(input_to_net)
        
        # print("input_to_rb1:",input_to_rb.shape)
        output = self.residual_block1(input_to_rb)

        depth_layer = torch.ones((input_to_rb.shape[0],1,input_to_rb.shape[2],input_to_rb.shape[3]))
        gazeX_layer = torch.ones((input_to_rb.shape[0],1,input_to_rb.shape[2],input_to_rb.shape[3]))
        gazeY_layer = torch.ones((input_to_rb.shape[0],1,input_to_rb.shape[2],input_to_rb.shape[3]))
        # print("depth_layer:",depth_layer.shape)
        # print("focal_depth:",focal_length," gazeX:",gazeX," gazeY:",gazeY, " gazeX norm:",(gazeX[0] - (-3.333)) / (3.333*2))
        for i in range(focal_length.shape[0]):
            depth_layer[i] *= 1. / focal_length[i]
            gazeX_layer[i] *= (gazeX[i] - (-3.333)) / (3.333*2)
            gazeY_layer[i] *= (gazeY[i] - (-3.333)) / (3.333*2)
            # print(depth_layer.shape)
        depth_layer = var_or_cuda(depth_layer)
        gazeX_layer = var_or_cuda(gazeX_layer)
        gazeY_layer = var_or_cuda(gazeY_layer)

        output = torch.cat((output,depth_layer,gazeX_layer,gazeY_layer),dim=1)
        # output = torch.cat((output,depth_layer),dim=1)
        # print("output to rb2:",output.shape)

        output = self.residual_block2(output)
        # print("output to rb3:",output.shape)
        output = self.residual_block3(output)
        # print("output to rb4:",output.shape)
        output = self.residual_block4(output)
        # print("output to rb5:",output.shape)
        output = self.residual_block5(output)
        # output = output + input_to_net
        output = self.output_layer(output)
        output = self.deinterleave(output)
        return output
