# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

import timm.models.vision_transformer
import torch.nn.functional as F


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
	""" Vision Transformer with support for global average pooling
	"""

	def __init__(self, global_pool=False, nb_classes=7, **kwargs):
		super(VisionTransformer, self).__init__(**kwargs)

		self.global_pool = global_pool
		if self.global_pool:
			norm_layer = kwargs['norm_layer']
			embed_dim = kwargs['embed_dim']
			self.fc_norm = norm_layer(embed_dim)

			del self.norm  # remove the original norm
		self.register_buffer("prototypes", torch.zeros(nb_classes, 768))
		self.proto_m = 0.99

	def forward_features(self, x):
		B = x.shape[0]
		x = self.patch_embed(x)

		cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
		x = torch.cat((cls_tokens, x), dim=1)
		x = x + self.pos_embed
		x = self.pos_drop(x)

		for blk in self.blocks:
			x = blk(x)

		if self.global_pool:
			x = x[:, 1:, :].mean(dim=1)  # global pool without cls token
			outcome = self.fc_norm(x)
		else:
			x = self.norm(x)
			outcome = x[:, 0]

		return outcome

	def forward(self, x, partial_Y=None):
		f = self.forward_features(x)
		if self.head_dist is not None:
			x, x_dist = self.head(f[0]), self.head_dist(f[1])  # x must be a tuple
			if self.training and not torch.jit.is_scripting():
				# during inference, return the average of both classifier predictions
				return x, x_dist
			else:
				return (x + x_dist) / 2
		else:
			x = self.head(f)

		if partial_Y is None:
			return x
		predicetd_scores = torch.softmax(x, dim=1) * partial_Y
		max_scores, pseudo_labels = torch.max(predicetd_scores, dim=1)

		# compute protoypical logits
		prototypes = self.prototypes.clone().detach()
		logits_prot = torch.mm(f, prototypes.t())

		score_prot = torch.softmax(logits_prot, dim=1)

		# update momentum prototypes with pseudo labels
		# for _, (feat, label, max_score) in enumerate(zip(f, pseudo_labels, max_scores)):
		# 	self.prototypes[label] = (
		# 			self.prototypes[label] * self.proto_m + (1 - self.proto_m) * feat
		# 	)
		for _, (feat, label, max_score) in enumerate(zip(f, pseudo_labels, max_scores)):
			prototypes[label] = (
					prototypes[label] * self.proto_m + (1 - self.proto_m) * feat
			)

		# normalize prototypes
		# self.prototypes = F.normalize(self.prototypes, p=2, dim=1)
		self.prototypes = F.normalize(prototypes, p=2, dim=1)

		return x, score_prot


def vit_base_affectnet(nb_classes, **kwargs):
	model = VisionTransformer(
		nb_classes=nb_classes,
		img_size=224,
		patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
		norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
	return model


model = vit_base_affectnet(nb_classes=1000)

x = torch.randn(2, 3, 224, 224)
y = model(x, partial_Y=torch.randn(2, 1000))
print()
