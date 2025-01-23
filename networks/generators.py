import os
import torch
import torch.nn as nn
import utils
from utils import CONFIG
import torch.nn.functional as F

from networks import decoders
from networks.encoders.MatteFormer import MatteFormer
from networks.DDFE.DDFE import DDFE
from networks.DDFE.DDFE import Refinement


def get_generator(is_train=True):
    generator = Generator_MatteFormer(is_train=is_train)
    return generator


class Generator_MatteFormer(nn.Module):

    def __init__(self, is_train=True):

        super(Generator_MatteFormer, self).__init__()
        # DDFE
        self.ddfe = DDFE()
        # Matting
        self.encoder = MatteFormer(embed_dim=96,
                                   depths=[2, 2, 6, 2],  # tiny-model
                                   num_heads=[3, 6, 12, 24],
                                   window_size=7,
                                   mlp_ratio=4.0,
                                   qkv_bias=True,
                                   qk_scale=None,
                                   drop_rate=0.0,
                                   attn_drop_rate=0.0,
                                   drop_path_rate=0.3,
                                   patch_norm=True,
                                   use_checkpoint=False
                                   )
        # original
        self.decoder = decoders.__dict__['res_shortcut_decoder']()
        # refinement
        self.refinement = Refinement()
        if is_train:
            # self.init_pretrained_weight(pretrained_path=CONFIG.model.imagenet_pretrain_path)  # MatteFormer
            self.init_matteformer_weight(matteformer_weight=CONFIG.model.mateformer_pretrain_path)
    
    def init_matteformer_weight(self, matteformer_weight=None):
        weight = torch.load(matteformer_weight)
        weight = utils.remove_prefix_state_dict(weight['state_dict'])
        # 使用字典解析过滤键值对
        filtered_dict_encoder = {key: value for key, value in weight.items() if 'encoder' in key}
        filtered_dict_decoder = {key: value for key, value in weight.items() if 'decoder' in key}
        new_dict_encoder = {key.replace('encoder.', ''): value for key, value in filtered_dict_encoder.items()}
        new_dict_decoder = {key.replace('decoder.', ''): value for key, value in filtered_dict_decoder.items()}
        self.encoder.load_state_dict(new_dict_encoder, strict=True)
        self.decoder.load_state_dict(new_dict_decoder, strict=True)
    
    def init_pretrained_weight(self, pretrained_path=None):
        if not os.path.isfile(pretrained_path):
            print('Please Check your Pretrained weight path.. file not exist : {}'.format(pretrained_path))
            exit()

        weight = torch.load(pretrained_path)['model']

        # [1] get backbone weights
        weight_ = {}
        for i, (k, v) in enumerate(weight.items()):
            head = k.split('.')[0]
            if head in ['patch_embed', 'layers']:
                if 'attn_mask' in k:
                    print('[{}/{}] {} will be ignored'.format(i, len(weight.items()), k))
                    continue
                weight_.update({k: v})
            else:
                print('[{}/{}] {} will be ignored'.format(i, len(weight.items()), k))

        patch_embed_weight = weight_['patch_embed.proj.weight']
        patch_embed_weight_new = torch.nn.init.xavier_normal_(torch.randn(96, (3 + 3), 4, 4).cuda())
        patch_embed_weight_new[:, :3, :, :].copy_(patch_embed_weight)
        weight_['patch_embed.proj.weight'] = patch_embed_weight_new

        attn_layers = [k for k, v in weight_.items() if 'attn.relative_position_bias_table' in k]
        for layer_name in attn_layers:
            pos_bias = weight_[layer_name]
            n_bias, n_head = pos_bias.shape

            layer_idx, block_idx = int(layer_name.split('.')[1]), int(layer_name.split('.')[3])
            n_prior = block_idx + 1
            pos_bias_new = torch.nn.init.xavier_normal_(torch.randn(n_bias + n_prior * 3, n_head))

            pos_bias_new[:n_bias, :] = pos_bias
            weight_[layer_name] = pos_bias_new

        attn_layers = [k for k, v in weight_.items() if 'attn.relative_position_index' in k]
        for layer_name in attn_layers:
            pos_index = weight_[layer_name]

            layer_idx, block_idx = int(layer_name.split('.')[1]), int(layer_name.split('.')[3])
            n_prior = block_idx + 1

            num_patch = 49
            last_idx = 169
            pos_index_new = torch.ones((num_patch, num_patch + n_prior * 3)).long() * last_idx
            pos_index_new[:num_patch, :num_patch] = pos_index
            for i in range(n_prior):
                for j in range(3):
                    pos_index_new[:, num_patch + i * 3 + j:num_patch + i * 3 + j + 1] = last_idx + i * 3 + j
            weight_[layer_name] = pos_index_new

        self.encoder.load_state_dict(weight_, strict=False)
        print('load pretrained model done')

    def forward(self, image, trimap):
        # DDFE
        image, x_hf = self.ddfe(image, trimap)

        trimap = F.interpolate(trimap, scale_factor=1 / 2, mode='nearest')
        inp = torch.cat((image, trimap), axis=1)

        # Low-resolution Matting
        x = self.encoder(inp, trimap)
        embedding = x[-1]
        outs_l = self.decoder(embedding, x[:-1])

        # High-resolution Refinement
        outs = self.refinement(outs_l, x_hf)
        return outs, x_hf, outs_l


if __name__ == '__main__':
    generator = Generator_MatteFormer().cuda()

    inp1 = torch.Tensor(2, 3, 2048, 2048).cuda()
    inp2 = torch.ones(2, 3, 2048, 2048).cuda()
    out = generator(inp1, inp2)

    print('Done')
