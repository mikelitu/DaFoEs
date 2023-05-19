import torch
import torch.nn as nn
from models.resnet import ResnetEncoder
from models.transformer import ViT
from models.recurrency import RecurrencyBlock
from models.utils import FcBlock

class ForceEstimator(nn.Module):

    def __init__(self, architecture: str, recurrency: bool, pretrained: bool,
                 include_depth: bool, att_type: str = None, embed_dim: int = 512,
                 state_size: int = 0) -> None:
        super(ForceEstimator, self).__init__()

        assert architecture in ["cnn", "vit", "fc"], "The resnet encoder must be either a cnn or a vision transformer"
        
        self.architecture = architecture
        self.recurrency = recurrency

        if self.architecture != "fc":
            if self.architecture == "cnn":
                self.encoder = ResnetEncoder(num_layers=50,
                                            pretrained=pretrained,
                                            include_depth=include_depth,
                                            att_type=att_type)
            elif self.architecture == "vit":
                self.encoder = ViT(image_size=256,
                                patch_size=16,
                                dim=1024,
                                depth=6,
                                heads=16,
                                mlp_dim=2048,
                                dropout=0.1,
                                emb_dropout=0.1,
                                channels=4 if include_depth else 3,
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
                    out = self.encoder(inp)
                    out = out.view(rec_size, -1)
                    features[i] = self.embed_block(out)
                
                pred = self.recurrency(features, rs)
            
            else:
                out = self.encoder(x)
                out_flatten = out.view(out.shape[0], -1)
                out = self.embed_block(out_flatten)

                if rs is not None:
                    out = torch.cat([out, rs], dim=1)
                
                pred = self.final(out)
                
        else:
            pred = self.encoder(x)

        return pred
