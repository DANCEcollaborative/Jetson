import torch
import torch.nn as nn
from typing import List

from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3
from torchvision.models._api import WeightsEnum
from torch.hub import load_state_dict_from_url


def get_state_dict(self, *args, **kwargs):
    kwargs.pop("check_hash")
    return load_state_dict_from_url(self.url, *args, **kwargs)


WeightsEnum.get_state_dict = get_state_dict


class FullImageModel(nn.Module):
    def __init__(
        self,
        num_hidden_fusion_layers: int = 1,
        dropout_p: float = 0.3,
        hidden_sizes: List = None,
        hidden_size_const: int = 512,
        post_concat_feat_sz: int = 512,
    ):
        super().__init__()
        assert num_hidden_fusion_layers > 0, "Must be at least 1 Hidden Layer"
        self.num_hidden_fusion_layers = num_hidden_fusion_layers
        self.backbone_1_out_feats = 1536
        self.backbone_2_out_feats = 1536
        self.post_concat_feat_sz = post_concat_feat_sz
        if hidden_sizes is not None:
            assert (
                len(hidden_sizes) == num_hidden_fusion_layers
            ), f"List of hidden sizes MUST HAVE {num_hidden_fusion_layers} items"
            self.hidden_sizes = hidden_sizes
        else:
            self.hidden_sizes = [hidden_size_const] * num_hidden_fusion_layers

        # Init backbones
        self.full_image_backbone = efficientnet_b3(
            weights=EfficientNet_B3_Weights.DEFAULT
        )
        self.face_image_backbone = efficientnet_b3(
            weights=EfficientNet_B3_Weights.DEFAULT
        )
        self.full_image_backbone.classifier = nn.Identity()
        self.face_image_backbone.classifier = nn.Identity()

        # Fusion
        # Concat Feats and Project to Hidden
        self.projection = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(
                in_features=(self.backbone_1_out_feats + self.backbone_2_out_feats),
                out_features=self.post_concat_feat_sz,
            ),
            nn.ReLU(),
        )

        layers: List[nn.Module] = [
            nn.Dropout(p=dropout_p),
            nn.Linear(self.post_concat_feat_sz, self.hidden_sizes[0]),
            nn.ReLU(),
        ]
        # Additional Hidden Layers
        for i in range(1, self.num_hidden_fusion_layers):
            layers.append(nn.Dropout(p=dropout_p))
            layers.append(nn.Linear(self.hidden_sizes[i - 1], self.hidden_sizes[i]))
            layers.append(nn.ReLU())
        self.fusion_layers = nn.Sequential(*layers)
        self.regression = nn.Linear(self.hidden_sizes[-1], 3)

    def forward(self, full_tensor: torch.Tensor, face_tensor: torch.Tensor):
        full_image_feats = self.full_image_backbone(full_tensor)
        face_image_feats = self.face_image_backbone(face_tensor)

        # Flatten Feature Maps
        full_image_feats = torch.flatten(full_image_feats, 1)
        face_image_feats = torch.flatten(face_image_feats, 1)

        # Fusion layers
        combined_feats = torch.cat([face_image_feats, full_image_feats], dim=-1)
        projected_feats = self.projection(combined_feats)
        fused_feats = self.fusion_layers(projected_feats)
        reg_out = self.regression(fused_feats)
        return reg_out


class KeypointsModel(nn.Module):
    def __init__(
        self,
        num_hidden_fusion_layers: int = 1,
        dropout_p: float = 0.3,
        hidden_size_const: int = 512,
        post_concat_feat_sz: int = 512,
    ):
        super().__init__()
        assert num_hidden_fusion_layers > 0, "Must be at least 1 Hidden Layer"
        self.num_hidden_fusion_layers = num_hidden_fusion_layers
        self.backbone_1_out_feats = 1536
        self.keypoint_feats = 33
        self.post_concat_feat_sz = post_concat_feat_sz
        self.hidden_size_const = hidden_size_const

        # Init backbones
        self.face_image_backbone = efficientnet_b3(
            weights=EfficientNet_B3_Weights.DEFAULT
        )
        self.face_image_backbone.classifier = nn.Identity()

        # Fusion
        # Concat Feats and Project to Hidden
        self.projection = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(
                in_features=(self.backbone_1_out_feats + self.keypoint_feats),
                out_features=self.post_concat_feat_sz,
            ),
        )

        self.first_fusion_block = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(self.post_concat_feat_sz, self.hidden_size_const),
            nn.ReLU(),
        )
        self.intermediate_fusion_block = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(
                self.hidden_size_const + self.keypoint_feats,
                self.hidden_size_const,
            ),
            nn.ReLU(),
        )
        blocks = [self.first_fusion_block]
        # Additional Hidden Layers
        if self.num_hidden_fusion_layers > 1:
            intermediate_fusion_blocks = [
                self.intermediate_fusion_block
                for _ in range(1, self.num_hidden_fusion_layers)
            ]
            blocks.extend(intermediate_fusion_blocks)

        self.fusion_layers = nn.ModuleList(blocks)
        self.regression = nn.Linear(self.hidden_size_const + self.keypoint_feats, 3)

    def forward(self, keypoint_feats: torch.Tensor, face_tensor: torch.Tensor):
        face_image_feats = self.face_image_backbone(face_tensor)
        face_image_feats = torch.flatten(face_image_feats, 1)
        keypoint_feats = torch.flatten(keypoint_feats, 1)
        combined_feats = torch.cat([face_image_feats, keypoint_feats], dim=-1)
        x = self.projection(combined_feats)
        for i, fusion_block in enumerate(self.fusion_layers):
            prev_feats = x
            if i == 0:
                concat_feats = x
            else:
                concat_feats = torch.cat([x, keypoint_feats], dim=-1)
            x = fusion_block(concat_feats) + prev_feats
        fused_feats = torch.cat([x, keypoint_feats], dim=-1)
        reg_out = self.regression(fused_feats)
        return reg_out


class OnePassModel(nn.Module):
    def __init__(
        self,
        num_hidden_layers: int = 1,
        dropout_p: float = 0.3,
        hidden_sizes: List = None,
        hidden_size_const: int = 512,
    ):
        super().__init__()
        assert num_hidden_layers > 0, "Must be at least 1 Hidden Layer"
        self.num_hidden_layers = num_hidden_layers
        self.backbone_out_feats = 1536
        if hidden_sizes:
            assert (
                len(hidden_sizes) == num_hidden_layers
            ), f"List of hidden sizes MUST HAVE {num_hidden_layers} items"
            self.hidden_sizes = hidden_sizes
        else:
            self.hidden_sizes = [hidden_size_const] * num_hidden_layers
        # Init backbones
        self.image_backbone = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)
        self.image_backbone.classifier = nn.Identity()
        # Project to Hidden
        self.projection = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(
                in_features=self.backbone_out_feats,
                out_features=self.hidden_sizes[0],
            ),
            nn.ReLU(),
        )
        layers: List[nn.Module] = [
            nn.Dropout(p=dropout_p),
            nn.Linear(self.hidden_sizes[0], self.hidden_sizes[0]),
            nn.ReLU(),
        ]
        # Additional Hidden Layers
        for i in range(1, self.num_hidden_layers):
            layers.append(nn.Dropout(p=dropout_p))
            layers.append(nn.Linear(self.hidden_sizes[i - 1], self.hidden_sizes[i]))
            layers.append(nn.ReLU())
        self.linear_layers = nn.Sequential(*layers)
        self.regression = nn.Linear(self.hidden_sizes[-1], 3)

    def forward(self, image_tensor: torch.Tensor):
        image_feats = self.image_backbone(image_tensor)
        image_feats = torch.flatten(image_feats, 1)
        projected_feats = self.projection(image_feats)
        hidden_output = self.linear_layers(projected_feats)
        reg_out = self.regression(hidden_output)
        return reg_out
