import torch
import torch.nn as nn
from models.resnet import ResnetEncoder
from models.transformer import ViT
from models.recurrency import RecurrencyBlock
from models.utils import FcBlock
from einops import repeat

class ForceEstimator(nn.Module):

    def __init__(self, architecture: str, recurrency: bool, pretrained: bool,
                 att_type: str = None, embed_dim: int = 512,
                 state_size: int = 0) -> None:
        super(ForceEstimator, self).__init__()

        assert architecture in ["cnn", "vit", "fc"], "The resnet encoder must be either a cnn or a vision transformer"
        
        self.architecture = architecture
        self.recurrency = recurrency

        if self.architecture != "fc":
            if self.architecture == "cnn":
                encoder = ResnetEncoder(num_layers=50,
                                            pretrained=pretrained,
                                            att_type=att_type)
                
                self.encoder = nn.Sequential(
                    encoder.encoder.conv1,
                    encoder.encoder.bn1,
                    encoder.encoder.relu,
                    encoder.encoder.maxpool,
                    encoder.encoder.layer1,
                    encoder.encoder.layer2,
                    encoder.encoder.layer3,
                    encoder.encoder.layer4[:2],
                    encoder.encoder.layer4[2].conv1,
                    encoder.encoder.layer4[2].bn1,
                    encoder.encoder.layer4[2].conv2,
                    encoder.encoder.layer4[2].bn2,
                    encoder.encoder.layer4[2].conv3
                )

                self.splitted = nn.Sequential(
                    encoder.encoder.layer4[2].bn3,
                    encoder.encoder.layer4[2].relu,
                    # encoder.encoder.avgpool,
                    # encoder.encoder.fc
                )

            elif self.architecture == "vit":
                encoder = ViT(image_size=256,
                                patch_size=16,
                                dim=1024,
                                depth=6,
                                heads=16,
                                mlp_dim=2048,
                                dropout=0.1,
                                emb_dropout=0.1,
                                channels=3,
                                )
                
                self.cls_token = nn.Parameter(torch.randn(1, 1, 1024))
                self.embeding = encoder.to_patch_embedding
                self.pos_embed = encoder.pos_embedding
                self.dropout = encoder.dropout

                self.encoder = nn.Sequential(
                    encoder.transformer.layers[:5]
                )

                self.last_layer = nn.Sequential(
                    encoder.transformer.layers[5]
                )

                self.splitted = nn.Sequential(
                    encoder.to_latent,
                    encoder.mlp_head
                )
                
            final_ch = 512 if self.architecture=="vit" else (2048 * 8 * 8)
            
            if not self.recurrency:
                self.embed_dim = embed_dim + state_size
            else:
                self.embed_dim = embed_dim

            self.embed_block = FcBlock(final_ch, embed_dim)
            
            if recurrency:
                self.recurrency = RecurrencyBlock(embed_dim=self.embed_dim)
            else:
                if state_size != 0:
                    self.final = nn.Sequential(
                        FcBlock(self.embed_dim, 84),
                        FcBlock(84, 180),
                        FcBlock(180, 50),
                        nn.Linear(50, 3)
                    )
                else:
                    self.final = nn.Linear(self.embed_dim, 3)
                
        else:
            self.encoder = nn.Sequential(
                FcBlock(state_size, 500),
                FcBlock(500, 1000),
                FcBlock(1000, 1000),
                FcBlock(1000, 500),
                FcBlock(500, 50),
                nn.Linear(50, 3)
            )
    
    def forward(self, x, rs = None) -> torch.Tensor:

        if self.architecture != "fc":
            if self.recurrency:
                batch_size = x[0].shape[0]
                rec_size = len(x)

                features = torch.zeros(batch_size, rec_size, self.embed_dim).cuda().float()

                for i in range(batch_size):
                    inp = torch.cat([img[i].unsqueeze(0) for img in x], dim=0)
                    if self.architecture == "vit":
                        inp = self.embeding(inp)
                        b, n, _ = inp.shape

                        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
                        inp = torch.cat((cls_tokens, inp), dim=1)
                        inp += self.pos_embed[:, :(n + 1)]
                        inp = self.dropout(inp)

                    out = self.encoder(inp)
                    # register a hook
                    if out.requires_grad:
                        h = out.register_hook(self.activations_hook)
                    if self.architecture == "vit":
                        out = self.last_layer(out)
                        out = out[:, 0]
                    
                    out = self.splitted(out)
                    out = out.view(rec_size, -1)
                    features[i] = self.embed_block(out)
                
                pred = self.recurrency(features, rs)
            
            else:

                if self.architecture == "vit":
                    x = self.embeding(x)
                    b, n, _ = x.shape

                    cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
                    x = torch.cat((cls_tokens, x), dim=1)
                    x += self.pos_embed[:, :(n + 1)]
                    x = self.dropout(x)

                out = self.encoder(x)
                
                # register a hook
                if out.requires_grad:
                    h = out.register_hook(self.activations_hook)
                if self.architecture == "vit":
                    out = self.last_layer(out)
                    out = out[:, 0]
                    
                out = self.splitted(out)
                out_flatten = out.view(out.shape[0], -1)
                out = self.embed_block(out_flatten)
                
                if rs is not None:
                    out = torch.cat([out, rs], dim=1)
                
                pred = self.final(out)
                
        else:
            pred = self.encoder(x)

        return pred
    

    def activations_hook(self, grad):
        self.gradients = grad
    

    def get_activations_gradient(self):
        return self.gradients
    

    def get_activations(self, x):

        if self.recurrency:
            batch_size = x[0].shape[0]

            for i in range(batch_size):
                inp = torch.cat([img[i].unsqueeze(0) for img in x], dim=0)
                if self.architecture == "vit":
                    inp = self.embeding(inp)
                    b, n, _ = inp.shape

                    cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
                    inp = torch.cat((cls_tokens, inp), dim=1)
                    inp += self.pos_embed[:, :(n + 1)]
                    inp = self.dropout(inp)

                inp = self.encoder(inp)

                if i == 0:
                    if self.architecture == "vit":
                        out = torch.zeros(batch_size, inp.shape[0], inp.shape[1], inp.shape[2]).cuda().float()
                    else:
                        out = torch.zeros(batch_size, inp.shape[0], inp.shape[1], inp.shape[2], inp.shape[3]).cuda().float()
                out[i] = inp
            
            out = torch.mean(out, dim=1)
        
        else:
            if self.architecture == "vit":
                x = self.embeding(x)
                b, n, _ = x.shape

                cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
                x = torch.cat((cls_tokens, x), dim=1)
                x += self.pos_embed[:, :(n + 1)]
                x = self.dropout(x)
            
            out = self.encoder(x)

        return out
