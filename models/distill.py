
import torch
import torch.nn.functional as F
from torch import nn
from models.force_estimator_transformers_base import BaseViT

from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

# classes

class DistillMixin:
    def forward(self, img, var = None, rs = None, distill_token = None):
        distilling = exists(distill_token)
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim = 1)
        x += self.pos_embedding[:, :(n + 1)]

        if distilling:
            distill_tokens = repeat(distill_token, '() n d -> b n d', b = b)
            x = torch.cat((x, distill_tokens), dim = 1)

        x = self._attend(x)

        if distilling:
            x, distill_tokens = x[:, :-1], x[:, -1]

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        if rs is not None:
            x = torch.cat((x, rs.squeeze(1)), dim=1)

        out = self.mlp_head(x)

        if distilling:
            return out, distill_tokens

        return out
    

class DistillableViT(DistillMixin, BaseViT):
    def __init__(self, *args, **kwargs):
        super(DistillableViT, self).__init__(*args, **kwargs)
        self.args = args
        self.kwargs = kwargs
        self.dim = kwargs['dim']
        self.num_classes = kwargs['num_classes']

    def to_vit(self):
        v = BaseViT(*self.args, **self.kwargs)
        v.load_state_dict(self.state_dict())
        return v

    def _attend(self, x, var = None, state = None):
        x = self.dropout(x)
        x = self.transformer(x)
        return x


# knowledge distillation wrapper

class DistillWrapper(nn.Module):
    def __init__(
        self,
        *,
        teacher,
        student,
        temperature = 1.,
        alpha = 0.5,
        hard = False
    ):
        super().__init__()
        assert (isinstance(student, (DistillableViT))) , 'student must be a vision transformer'

        self.teacher = teacher
        self.student = student

        dim = student.dim
        num_classes = student.num_classes
        self.temperature = temperature
        self.alpha = alpha
        self.hard = hard

        self.distillation_token = nn.Parameter(torch.randn(1, 1, dim))

        self.distill_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, labels, robot_state = None, temperature = None, alpha = None, **kwargs):
        b, *_ = img.shape
        alpha = alpha if exists(alpha) else self.alpha
        T = temperature if exists(temperature) else self.temperature

        with torch.no_grad():
            if robot_state is not None:
                teacher_logits = self.teacher(img, robot_state.squeeze(1))
            else:
                teacher_logits = self.teacher(img)

        student_logits, distill_tokens = self.student(img, None, robot_state, distill_token = self.distillation_token, **kwargs)
        distill_logits = self.distill_mlp(distill_tokens)

        loss = F.l1_loss(student_logits, labels)

        if not self.hard:
            distill_loss = F.kl_div(
                F.log_softmax(distill_logits / T, dim = -1),
                F.softmax(teacher_logits / T, dim = -1).detach(),
            reduction = 'batchmean')
            distill_loss *= T ** 2

        else:
            teacher_labels = teacher_logits.argmax(dim = -1)
            distill_loss = F.l1_loss(distill_logits, teacher_labels)

        return loss * (1 - alpha) + distill_loss * alpha