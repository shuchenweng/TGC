import torch
import torch.nn as nn
from miscc.utils import l1norm
from miscc.config import cfg

from models.Bmodel import conv1x1, cos_attention, dot_attention

class GlobalAttentionGeneral(nn.Module):
    def __init__(self, idf, cdf):
        super(GlobalAttentionGeneral, self).__init__()
        self.conv_context = conv1x1(cdf, idf)
        self.sm = nn.Softmax(dim=1)
        self.mask = None

    def applyMask(self, mask):
        self.mask = mask  # batch x sourceL

    def forward(self, input, context):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)

        # --> batch x idf x queryL
        target = input.view(batch_size, -1, queryL)
        # --> batch x queryL x idf
        targetT = torch.transpose(target, 1, 2).contiguous()
        # batch x cdf x sourceL --> batch x cdf x sourceL x 1
        sourceT = context.unsqueeze(3)
        # --> batch x idf x sourceL
        sourceT = self.conv_context(sourceT).squeeze(3)
        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = torch.bmm(targetT, sourceT)
        # --> (batch*queryL, sourceL)
        attn = attn.view(batch_size*queryL, sourceL)
        if self.mask is not None:
            # batch_size x sourceL --> batch_size*queryL x sourceL
            mask = self.mask.repeat(queryL, 1)

        attn.data.masked_fill_(mask.byte().data, -float('inf'))
        attn = self.sm(attn)  # Eq. (2)


        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        return sourceT, attn

class AuxAttentionGeneral(nn.Module):
    def __init__(self, cdf, idf):
        super(AuxAttentionGeneral, self).__init__()


    def forward(self, input, context, pooled_sem_seg, word_context=None):
        """
            input: batch x idf x ih x iw (queryL=ihxiw)
            context: batch x cdf x sourceL
        """
        ih, iw = input.size(2), input.size(3)
        queryL = ih * iw
        batch_size, sourceL = context.size(0), context.size(2)
        # --> batch x idf x queryL
        target = input.view(batch_size, -1, queryL)
        # Get attention
        # (batch x queryL x idf)(batch x idf x sourceL)
        # -->batch x queryL x sourceL
        attn = cos_attention(target, context)

        pooled_sem_seg = pooled_sem_seg.view(batch_size, queryL)
        attn[pooled_sem_seg==0] = 0
        # --> (batch*queryL, sourceL)
        attn = attn.view(batch_size*queryL, sourceL)
        attn = l1norm(attn, -1)
        # --> batch x queryL x sourceL
        attn = attn.view(batch_size, queryL, sourceL)
        # --> batch x sourceL x queryL
        attn = torch.transpose(attn, 1, 2).contiguous()

        return context, attn