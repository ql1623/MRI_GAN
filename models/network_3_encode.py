import torch
import torch.nn as nn
from torchsummary import summary

"""Have Segmentation Network separately and not inside GAN, take feature layers right before out_conv of fusion"""
"""2 final conv in fusion, final_conv1 and final_conv2, with params: pre_out_features
    sampling and single conv instead of maxpool in unet of modalitiy
    commented out parts with self.use_attn for now, if need just uncomments parts with self.use_attn
    use fusion decoder, and batch norm for modality encoder network
    """
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels): # act_fn="leaky", norm="batch", use_dropout=False
        """Convolution Block for a Double Convolution operation
        
        Parameters:
            in_channels (int)  : the number of channels from input images.
            out_channels (int) : the number of channels in output images.
            act_fn (str)       : the activation function: leakyReLU | ReLU.
            norm (str)         : the type of normalization: batch | instance.
            use_dropout (bool) : if dropout is used
        """
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # self.use_dropout = use_dropout
        # self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv(x)
        # if self.use_dropout:
        #     return self.dropout(x)
        # else:
        return x

class DoubleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):# act_fn="leaky", norm="batch", use_dropout=False
        """Convolution Block for a Double Convolution operation
        
        Parameters:
            in_channels (int)  : the number of channels from input images.
            out_channels (int) : the number of channels in output images.
            act_fn (str)       : the activation function: leakyReLU | ReLU.
            norm (str)         : the type of normalization: batch | instance.
            use_dropout (bool) : if dropout is used
        """
        super(DoubleConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # self.use_dropout = use_dropout
        # self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv(x)
        # if self.use_dropout:
        #     return self.dropout(x)
        # else:
        return x
      
        
class SamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, down=True): # act_fn="leaky", norm="batch", use_dropout=False
        """Convolution Block for upsampling / downsampling
        
        Parameters:
            in_channels (int)  : the number of channels from input images.
            out_channels (int) : the number of channels in output images.
            down (bool)        : if it is a block in downsampling path
            act_fn (str)       : the activation function: leakyReLU | ReLU.
            norm (str)         : the type of normalization: batch | instance.
            use_dropout (bool) : if dropout is used
        """
        super(SamplingBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1) if down
            else nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # self.use_dropout = use_dropout
        # self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = self.conv(x)
        # if self.use_dropout:
        #     return self.dropout(x)
        # else:
        return x

class ResConvBlock_v2(nn.Module):
    def __init__(self, in_channels, out_channels, need_identity=False):  # act_fn="leaky", norm="batch"
        """Convolution Block for a Double Convolution operation
        
        Parameters:
            in_channels (int)  : the number of channels from input images.
            out_channels (int) : the number of channels in output images.
            act_fn (str)       : the activation function: leakyReLU | ReLU.
            norm (str)         : the type of normalization: batch | instance.
            use_dropout (bool) : if dropout is used
        """
        super(ResConvBlock_v2, self).__init__()
        
        self.need_identity = need_identity
        
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(out_channels),
        )
        
        self.activ_func = nn.LeakyReLU(0.2, inplace=True) 
        if self.need_identity:
            self.res = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                # nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.InstanceNorm2d(out_channels),
            )
        
    def forward(self, x):
        # import pdb; pdb.set_trace()
        # residual = x.clone()
        residual = x
        
        x = self.conv(x)
        
        if self.need_identity:
            residual = self.res(residual)
            
        x = torch.add(x, residual)
        x = self.activ_func(x)
        return x
    


class FusionBlock_v2(nn.Module):
    def __init__(self, in_channels, out_channels, groups, num_mod=2, with_concat_cond=False, is_separate=True): # act_fn="leaky", norm="batch"
        """Fuse same-level feature layers of different modalities

        Parameters:
            in_channels (int)  : the number of channels from each modality input images.
            out_channels (int) : the number of channels in fused output images.
            groups (int)       : the number of groups to separate data into and perform conv with.
            act_fn (str)       : the activation function: leakyReLU | ReLU.
            norm (str)         : the type of normalization: batch | instance.
            initial (bool)     : if it is the first fusion block (as first will not have previous fused representation to be merge with).
        """
        super(FusionBlock_v2, self).__init__()
        self.with_concat_cond = with_concat_cond
        self.is_separate = is_separate
        self.num_mod = num_mod
        
        cond_dim = 1 if self.is_separate else 1
        channel_mul = self.num_mod + cond_dim
        
        # conv 3d to merge representation of same-level feature layers from both modality
        if self.with_concat_cond:
            # conv 3d to merge representation of same-level feature layers from both modality
            self.group_conv =  nn.Sequential(
                nn.Conv2d(in_channels*channel_mul, out_channels, kernel_size=3, padding=1, groups=groups) if self.is_separate
                else nn.Conv2d(in_channels*channel_mul, out_channels, kernel_size=3, padding=1, groups=groups),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.group_conv =  nn.Sequential(
                nn.Conv2d(in_channels*self.num_mod, out_channels, kernel_size=3, padding=1, groups=groups),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        
    def forward(self, x1, x2, x3, condition=None): #
        if self.with_concat_cond:
            assert condition != None
            # if self.is_separate:
            #     assert condition.shape[1] == 3
            # else:
            #     assert condition.shape[1] == 1
                
        if self.with_concat_cond:
            x = torch.cat((x1, x2, x3, condition), dim=1) # [B, C, H, W] --> [B, C*stack, H, W] = [B, C=256*2 or C=256*3 or C=256*5, H, W]
            
        else:
            x = torch.cat((x1, x2, x3), dim=1)
        
        # implement the conv3d defined 
        x = self.group_conv(x)

        return x

    

class ConditionalBlock_v3(nn.Module):
    def __init__(self, in_channels, out_channels, feat_img_size): 
        super(ConditionalBlock_v3,self).__init__()
        self.out_channels = out_channels
        self.feat_img_size = feat_img_size
        # self.has_act_fn = has_act_fn
        
        # repeat()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, self.out_channels, kernel_size=(3,3,3), stride=1, padding=(0,1,1)), # out_channels should be 256
            nn.InstanceNorm3d(self.out_channels),
            nn.LeakyReLU(0.2, inplace=True)
            )
        
    def forward(self, condition):
        # condition = [B, C=3] 3 as one hot 
        # [B, C=3] --> [B, C=3, feat_img_size, feat_img_size] --> [B, C=1, D=3, feat_img_size, feat_img_size]
        condition = (condition.unsqueeze(2).unsqueeze(3).repeat(1, 1, self.feat_img_size, self.feat_img_size)).unsqueeze(1).float() 
        condition = self.conv(condition) # [B, C=3, feat_img_size, feat_img_size] --> [B, C=256, feat_img_size, feat_img_size] to be same as fusion feat to stack 
        return torch.squeeze(condition, dim=2)


class Unet_maxpool(nn.Module):
    def __init__(self, in_channels, out_channels, ngf, norm="instance"): 
        super(Unet_maxpool, self).__init__()
        self.in_channels = in_channels
        self.features = ngf
        self.out_channels = out_channels
        self.norm = norm
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down1_mod1 = ConvBlock(in_channels=self.in_channels, out_channels=self.features)
        # self.down1_mod1 = ConvBlock(in_channels=self.in_channels, out_channels=self.features)
        # in_channels=self.input_channels * 2 as adding in labels for cgan
        # self.pool1_mod1 = SamplingBlock(in_channels=self.features, out_channels=self.features, down=True)
        
        self.down2_mod1 = ConvBlock(in_channels=self.features, out_channels=self.features*2)
        # self.down2_mod1 = ConvBlock(in_channels=self.features, out_channels=self.features*2)
        # self.pool2_mod1 = SamplingBlock(in_channels=self.features*2, out_channels=self.features*2, down=True)
        
        self.down3_mod1 = ConvBlock(in_channels=self.features*2, out_channels=self.features*4)
        # self.down3_mod1 = ConvBlock(in_channels=self.features*2, out_channels=self.features*4)
        # self.pool3_mod1 = SamplingBlock(in_channels=self.features*4, out_channels=self.features*4, down=True)
        
        self.down4_mod1 = ConvBlock(in_channels=self.features*4, out_channels=self.features*8)
        # self.down4_mod1 = ConvBlock(in_channels=self.features*4, out_channels=self.features*8)
        # self.pool4_mod1 = SamplingBlock(in_channels=self.features*8, out_channels=self.features*8, down=True)
        
        # bottleneck
        self.bottleneck_mod1 = nn.Sequential(
            nn.Conv2d(self.features*8, self.features*8, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(self.features*8),
            nn.LeakyReLU(inplace=True)
        )
        
        # decoding path
        self.up1_mod1     = SamplingBlock(self.features*8, self.features*8, down=False)
        self.upconv1_mod1 = ConvBlock(self.features*8*2, self.features*8)
        
        self.up2_mod1     = SamplingBlock(self.features*4*2, self.features*4, down=False)
        self.upconv2_mod1 = ConvBlock(self.features*4*2, self.features*4)
        
        self.up3_mod1     = SamplingBlock(self.features*2*2, self.features*2, down=False)
        self.upconv3_mod1 = ConvBlock(self.features*2*2, self.features*2)
        
        self.up4_mod1     = SamplingBlock(self.features*1*2, self.features*1, down=False)
        self.upconv4_mod1 = ConvBlock(self.features*1*2, self.features*1)
        
        # final conv
        self.final_conv_mod1 = nn.Sequential(
            nn.Conv2d(self.features, self.out_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
        
    def forward(self, x):
        skip_connections_1 = []
        down1_x1 = self.down1_mod1(x)
        skip_connections_1.append(down1_x1)
        pool1_x1 = self.pool(down1_x1)
        # pool1_x1 = self.pool1_mod1(down1_x1)
        
        down2_x1 = self.down2_mod1(pool1_x1)
        skip_connections_1.append(down2_x1)
        pool2_x1 = self.pool(down2_x1)
        # pool2_x1 = self.pool2_mod1(down2_x1)
        
        down3_x1 = self.down3_mod1(pool2_x1)
        skip_connections_1.append(down3_x1)
        pool3_x1 = self.pool(down3_x1)
        # pool3_x1 = self.pool3_mod1(down3_x1)
        
        down4_x1 = self.down4_mod1(pool3_x1)
        skip_connections_1.append(down4_x1)
        pool4_x1 = self.pool(down4_x1)
        # pool4_x1 = self.pool4_mod1(down4_x1)
        
        bottleneck_x1 = self.bottleneck_mod1(pool4_x1)
        
        up1_x1 = self.up1_mod1(bottleneck_x1)
        upconv1_x1 = self.upconv1_mod1(torch.cat((up1_x1, skip_connections_1[3]),dim=1))
        
        up2_x1 = self.up2_mod1(upconv1_x1)
        upconv2_x1 = self.upconv2_mod1(torch.cat((up2_x1, skip_connections_1[2]),dim=1))
        
        up3_x1 = self.up3_mod1(upconv2_x1)
        upconv3_x1 = self.upconv3_mod1(torch.cat((up3_x1, skip_connections_1[1]),dim=1))
        
        up4_x1 = self.up4_mod1(upconv3_x1)
        upconv4_x1 = self.upconv4_mod1(torch.cat((up4_x1, skip_connections_1[0]),dim=1))
        
        out_x1 = self.final_conv_mod1(upconv4_x1)
        
        return bottleneck_x1, upconv1_x1, upconv2_x1, upconv3_x1, upconv4_x1, out_x1
        

      
# ---------------------------------------------------------------------------------
# =========== define model architecture ============ #
class datasetGAN(nn.Module):
    """ Defining Generator of the model """
    def __init__(self, input_channels, output_channels, ngf, pre_out_channels, norm="instance"): # input_channels = 1, output_channel = 1, ngf = 64/32 , features = [16,32,64,128]
        """Generator with UNET and Fusion Network 

        Parameters:
            input_channels (int)  : the number of channels from each modality input images.
            output_channels (int) : the number of channels in fused output images.
            ngf (int)       : the number of filter in the first conv layer
        """
        super(datasetGAN,self).__init__()
        
        self.input_channels = input_channels
        self.features = ngf
        self.output_channels = output_channels
        self.pre_output_channels = pre_out_channels
        self.norm = norm    
        
        # -------- Modality Encoder - Modality 1 -------- 
        self.unet1 = Unet_maxpool(self.input_channels, self.output_channels, self.features)
        self.unet2 = Unet_maxpool(self.input_channels, self.output_channels, self.features)
        self.unet3 = Unet_maxpool(self.input_channels, self.output_channels, self.features)
        
        # --------------- fusion network ---------------
       
        self.cond_v2 = ConditionalBlock_v3(in_channels=1, out_channels=self.features*8, feat_img_size=16)
        
        self.fusion0 = FusionBlock_v2(self.features*8, self.features*8, groups=self.features*8, num_mod=3, with_concat_cond=True, is_separate=True,)
        self.fusion_up0 = SamplingBlock(self.features*8, self.features*8, down=False)
        
        self.fusion1 = FusionBlock_v2(self.features*8, self.features*8, groups=self.features*8, num_mod=3, with_concat_cond=False)
        self.fusion_conv1 = ConvBlock(self.features*8*2, self.features*8)
        self.fusion_up1 = SamplingBlock(self.features*8, self.features*4, down=False)
        
        self.fusion2 = FusionBlock_v2(self.features*4, self.features*4, groups=self.features*4, num_mod=3, with_concat_cond=False)
        self.fusion_conv2 = ConvBlock(self.features*4*2, self.features*4)
        self.fusion_up2 = SamplingBlock(self.features*4, self.features*2, down=False)
        
        self.fusion3 = FusionBlock_v2(self.features*2, self.features*2, groups=self.features*2, num_mod=3, with_concat_cond=False)
        self.fusion_conv3 = ConvBlock(self.features*2*2, self.features*2)
        self.fusion_up3 = SamplingBlock(self.features*2, self.features*1, down=False)
        
        self.fusion4 = FusionBlock_v2(self.features*1, self.features*1, groups=self.features*1, num_mod=3, with_concat_cond=False)
        self.fusion_conv4 = ConvBlock(self.features*2*1, self.features*1)

        # whether to add one more conv so input to seg is lower 
        self.fusion_final_conv1 = ConvBlock(self.features, self.pre_output_channels)
        
        # ---- fusion decoding  ----
        self.fusion_decode_conv1 = ResConvBlock_v2(self.pre_output_channels, self.pre_output_channels)
        self.fusion_decode_conv2 = ResConvBlock_v2(self.pre_output_channels, self.pre_output_channels)
        
        self.fusion_final_conv2 = nn.Sequential(
            nn.Conv2d(self.pre_output_channels, self.output_channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    

                
    def forward(self, inputs, condition):

        x1 = inputs[:,0:1,:,:] # modality 1
        x2 = inputs[:,1:2,:,:] # modality 2
        x3 = inputs[:,2:3,:,:] # modality 3
                
        bottleneck_x1, upconv1_x1, upconv2_x1, upconv3_x1, upconv4_x1, out_x1= self.unet1(x1)
        bottleneck_x2, upconv1_x2, upconv2_x2, upconv3_x2, upconv4_x2, out_x2= self.unet2(x2)
        bottleneck_x3, upconv1_x3, upconv2_x3, upconv3_x3, upconv4_x3, out_x3= self.unet3(x3)
        

        # ----- fusion part (for decoding path) -----

        condition_feat = self.cond_v2(condition) 
        fusion_0 = self.fusion0(bottleneck_x1, bottleneck_x2, bottleneck_x3, condition_feat)
        fusion_upsamp0 = self.fusion_up0(fusion_0)
        
        fusion_1 = self.fusion1(upconv1_x1, upconv1_x2, upconv1_x3)
        fusion_merge1 = self.fusion_conv1(torch.cat((fusion_1,fusion_upsamp0),dim=1))
        fusion_upsamp1 = self.fusion_up1(fusion_merge1)
        
        fusion_2 = self.fusion2(upconv2_x1, upconv2_x2, upconv2_x3)
        fusion_merge2 = self.fusion_conv2(torch.cat((fusion_2,fusion_upsamp1),dim=1))
        fusion_upsamp2 = self.fusion_up2(fusion_merge2)
        
        fusion_3 = self.fusion3(upconv3_x1, upconv3_x2, upconv3_x3)
        fusion_merge3 = self.fusion_conv3(torch.cat((fusion_3,fusion_upsamp2),dim=1))
        fusion_upsamp3 = self.fusion_up3(fusion_merge3)
        
        fusion_4 = self.fusion4(upconv4_x1, upconv4_x2, upconv4_x3)
        fusion_merge4 = self.fusion_conv4(torch.cat((fusion_4,fusion_upsamp3),dim=1))
        
        pre_out_fusion = self.fusion_final_conv1(fusion_merge4)
            
        fusion_decode_1 = self.fusion_decode_conv1(pre_out_fusion)
        fusion_decode_2 = self.fusion_decode_conv2(fusion_decode_1)
        out_fusion = self.fusion_final_conv2(fusion_decode_2)
        
        return out_fusion, out_x1, out_x2, out_x3, pre_out_fusion
            


class SegmentationNetwork(nn.Module):
    def __init__(self, input_ngf, output_channels):
        super(SegmentationNetwork, self).__init__()
        
        self.features = input_ngf
        self.output_channels = output_channels
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.seg_conv_1 = ResConvBlock_v2(self.features, self.features*2, need_identity=True)
        self.seg_down_1 = SamplingBlock(self.features*2, self.features*4)

        self.seg_conv_2 = ResConvBlock_v2(self.features*4, self.features*4)
        self.seg_down_2 = SamplingBlock(self.features*4, self.features*8)

        self.seg_conv_3 = ResConvBlock_v2(self.features*8, self.features*8)
        self.seg_down_3 = SamplingBlock(self.features*8, self.features*8)

        self.seg_bottleneck = nn.Sequential(
            nn.Conv2d(self.features*8, self.features*8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.features*8),    
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        self.seg_up_1 = SamplingBlock(self.features*8, self.features*8, down=False)
        self.seg_upconv_1 = DoubleConvBlock(self.features*8*2, self.features*8)

        self.seg_up_2 = SamplingBlock(self.features*8, self.features*4, down=False)
        self.seg_upconv_2 = DoubleConvBlock(self.features*4*2, self.features*4)

        self.seg_up_3 = SamplingBlock(self.features*4, self.features*2, down=False)
        self.seg_upconv_3 = DoubleConvBlock(self.features*2*2, self.features*2)

        self.seg_final_conv = nn.Conv2d(self.features*2, self.output_channels, kernel_size=1, stride=1, padding=0)
        
    
    def forward(self, image_features):
        seg_skip_connection = []
        seg_conv_fu1 = self.seg_conv_1(image_features)
        seg_skip_connection.append(seg_conv_fu1)
        seg_down_fu1 = self.seg_down_1(seg_conv_fu1)
        
        seg_conv_fu2 = self.seg_conv_2(seg_down_fu1)
        seg_skip_connection.append(seg_conv_fu2)
        seg_down_fu2 = self.seg_down_2(seg_conv_fu2)
        
        seg_conv_fu3 = self.seg_conv_3(seg_down_fu2)
        seg_skip_connection.append(seg_conv_fu3)
        seg_down_fu3 = self.seg_down_3(seg_conv_fu3)
        
        seg_bottleneck_fu = self.seg_bottleneck(seg_down_fu3)
        
        seg_up_fu1 = self.seg_up_1(seg_bottleneck_fu)
        seg_upconv_fu1 = self.seg_upconv_1(torch.cat((seg_up_fu1, seg_skip_connection[2]), dim=1))
        
        seg_up_fu2 = self.seg_up_2(seg_upconv_fu1)
        seg_upconv_fu2 = self.seg_upconv_2(torch.cat((seg_up_fu2, seg_skip_connection[1]), dim=1))
        
        seg_up_fu3 = self.seg_up_3(seg_upconv_fu2)
        seg_upconv_fu3 = self.seg_upconv_3(torch.cat((seg_up_fu3, seg_skip_connection[0]), dim=1))
        
        seg_output = self.seg_final_conv(seg_upconv_fu3)
        
        return seg_output


class Discriminator(nn.Module):
    def __init__(self, in_channels=1, features=[32,64,128,256,512]):
        super(Discriminator, self).__init__()
        self.intitial_conv = ConvBlock(in_channels + 2, features[0])
        
        layers = []
        for feat in features[:-1]:
            layers.append(
                ConvBlock(feat, feat*2)
            )
        layers.append(nn.Conv2d(features[-1], in_channels, kernel_size=3, stride=1, padding=1)) 
        self.disc = nn.Sequential(*layers)

    def forward(self, x, input_x1, inputx2):
        x = self.intitial_conv(torch.cat((x, input_x1, inputx2), 1))
        return self.disc(x)   

def test():


    # model = datasetGAN(1,1,32, version=4)
    # model = datasetGAN(1,1,32, 16)
    # summary(model, [(3, 256, 256), (3, )]) 
    
    # model = Discriminator(in_channels=1, features=[32,64,128,256,512])
    # summary(model, [(1, 256, 256),  (1, 256, 256),  (1, 256, 256)]) 
    
    model = SegmentationNetwork(16, 1)
    summary(model, (16, 256, 256)) 
    
