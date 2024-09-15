import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
from open_clip import create_model_from_pretrained, get_tokenizer
from ezcolorlog import root_logger as logger
from tome.patch import clip

from .base_encoder import BaseVisionTower
from cambrian.utils import IS_XLA_AVAILABLE


def extract_interp(model_name):
    interp = None
    base_model_name = model_name

    if "interp" in model_name:
        base_model_name = model_name.split('-interp')[0]

    parts = model_name.split("-")
    for part in parts:
        if part.startswith("interp"):
            interp = int(part[6:])

    return base_model_name, interp


class ClipVisionTower(BaseVisionTower):
    def __init__(self, vision_tower_name, args, delay_load=False):
        super(ClipVisionTower, self).__init__(vision_tower_name, args, delay_load)
        base_model_name, interp = extract_interp(vision_tower_name)
        self.vision_tower_name = base_model_name
        self._interp_size = interp 
        self.use_token_merging = args.use_token_merging
        self.token_merging_r = args.token_merging_r
        if not self.delay_load:
            self.load_model()
        elif self.unfreeze_mm_vision_tower:
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            logger.debug(f"{self.vision_tower_name} is already loaded, `load_model` called again, skipping.")
            return
        print(self.vision_tower_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        # token merging patch here
        if self.use_token_merging:
            clip(self.vision_tower)
            # process r to matches the need
            # for now, we temporarily use 576 as the original number of tokens since we use single clip-vit-large-patch14-336
            # later, it should be changed to passing argument in
            # here we define a function to make sure the total number of tokens merged equals to original number of tokens - desired number of tokens
            target_sum = 576 - self._interp_size
            length = len(self.vision_tower.vision_model.encoder.layers) + self.select_layer + 1
            def adjust_list(length, r, target_sum):
                # Step 1: Calculate the original sum and the total reduction needed
                original_sum = length * r
                reduction_needed = original_sum - target_sum
                
                # Step 2: Initialize the list with all elements as r
                result = [r] * length
                
                # Step 3: Distribute the reduction across the list elements
                for i in range(length):
                    # Calculate how much can be reduced from the current element
                    max_reduction = min(r, reduction_needed)
                    result[length - 1 - i] -= max_reduction
                    reduction_needed -= max_reduction
                    
                    # Break early if no more reduction is needed
                    if reduction_needed == 0:
                        break
                
                return result
            r_list = adjust_list(length, self.token_merging_r, target_sum)
            # we need to append 0s to the end of r_list to make it match the number of layers
            r_list += [0] * (len(self.vision_tower.vision_model.encoder.layers) - length)
            print("the list of r for token merging: ", r_list)
            self.vision_tower.r = r_list

        self.vision_tower.requires_grad_(self.unfreeze_mm_vision_tower)
        self.is_loaded = True

        if IS_XLA_AVAILABLE:
            # Very Important for TorchXLA
            from torch_xla.utils.checkpoint import checkpoint
            self.vision_tower.vision_model.encoder._gradient_checkpointing_func = checkpoint

    def _feature_select(self, image_features):
        if self.select_feature == 'patch':
            features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return features

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        return self._feature_select(image_features)

    def interpolate(self, image_features):
        if self._interp_size is None:
            return image_features

        b, num_tokens, dim = image_features.shape

        if num_tokens != self.num_patches:
            target_h = target_w = int(self._interp_size ** 0.5)
            h = w = int(num_tokens ** 0.5)

            image_features = image_features.view(b, h, w, dim)
            image_features = image_features.permute(0, 3, 1, 2).contiguous()

            image_features = F.interpolate(
                image_features.to(torch.float32),
                size=(target_h, target_w),
                mode='bilinear',
                align_corners=False
            ).to(image_features.dtype)

            # Permute the dimensions back to (b, target_h, target_w, dim)
            image_features = image_features.permute(0, 2, 3, 1).contiguous()

            # Flatten the spatial dimensions (target_h, target_w) into a single dimension
            image_features = image_features.flatten(1, 2)

        return image_features

    def _forward(self, images):
        if IS_XLA_AVAILABLE:
            from torch_xla.utils.checkpoint import checkpoint
            self.vision_tower.vision_model.encoder._gradient_checkpointing_func = checkpoint

        with torch.set_grad_enabled(self.unfreeze_mm_vision_tower):
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            interp_features = self.interpolate(image_features)
            return interp_features
