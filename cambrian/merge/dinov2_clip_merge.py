import torch
from transformers import CLIPVisionModel, Dinov2Model
from collections import OrderedDict
from functools import reduce
import argparse
from tqdm import tqdm

def load_models():
    dinov2_model = Dinov2Model.from_pretrained("facebook/dinov2-large")
    clip_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
    return dinov2_model, clip_model

def get_mergeable_modules():
    # Define which modules to merge and their corresponding names in each model
    return {
        "embeddings": {
            "clip": "vision_model.embeddings.patch_embedding",
            "dinov2": "embeddings.patch_embeddings.projection",
        },
        "encoder_layers": {
            "clip": "vision_model.encoder.layers",
            "dinov2": "encoder.layer",
            "submodules": {
                "attention": {
                    "clip": "self_attn",
                    "dinov2": "attention",
                    "q": {"clip": "q_proj", "dinov2": "attention.query"},
                    "k": {"clip": "k_proj", "dinov2": "attention.key"},
                    "v": {"clip": "v_proj", "dinov2": "attention.value"},
                    "out": {"clip": "out_proj", "dinov2": "output.dense"},
                },
                "mlp": {
                    "clip": "mlp",
                    "dinov2": "mlp",
                    "fc1": {"clip": "fc1", "dinov2": "fc1"},
                    "fc2": {"clip": "fc2", "dinov2": "fc2"},
                },
                "layer_norm1": {
                    "clip": "layer_norm1",
                    "dinov2": "norm1",
                },
                "layer_norm2": {
                    "clip": "layer_norm2",
                    "dinov2": "norm2",
                },
            },
        }
    }

def get_module_by_name(model, name):
    return reduce(getattr, name.split('.'), model)

def merge_parameters(merged_module, clip_module, dinov2_module, weight=0.5):
    with torch.no_grad():
        for (name, p_merged), (_, p_clip), (_, p_dinov2) in zip(merged_module.named_parameters(), 
                                                                clip_module.named_parameters(), 
                                                                dinov2_module.named_parameters()):
            if p_clip.shape == p_dinov2.shape:
                p_merged.data = weight * p_clip.data + (1 - weight) * p_dinov2.data
            else:
                # raise an error if the shapes of the parameters are different
                raise ValueError(f"Parameter shapes of {name} are different: {p_clip.shape} and {p_dinov2.shape}")

def create_merged_model(clip_model, dinov2_model, weight=0.5):
    merged_model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
    mergeable_modules = get_mergeable_modules()

    # Count total number of operations for tqdm
    total_ops = sum(1 for module_type, module_info in mergeable_modules.items() 
                    if module_type != "encoder_layers")
    total_ops += sum(len(module_info['submodules']) * min(len(get_module_by_name(clip_model, module_info['clip'])),
                                                          len(get_module_by_name(dinov2_model, module_info['dinov2'])))
                     for module_type, module_info in mergeable_modules.items() 
                     if module_type == "encoder_layers")

    with tqdm(total=total_ops, desc="Merging models", unit="op") as pbar:
        for module_type, module_info in mergeable_modules.items():
            if module_type == "encoder_layers":
                clip_layers = get_module_by_name(clip_model, module_info['clip'])
                dinov2_layers = get_module_by_name(dinov2_model, module_info['dinov2'])
                merged_layers = get_module_by_name(merged_model, module_info['clip'])

                for i in range(min(len(clip_layers), len(dinov2_layers))):
                    for submodule, submodule_info in module_info['submodules'].items():
                        clip_submodule = getattr(clip_layers[i], submodule_info['clip'])
                        dinov2_submodule = getattr(dinov2_layers[i], submodule_info['dinov2'])
                        merged_submodule = getattr(merged_layers[i], submodule_info['clip'])

                        if submodule == "attention":
                            for attn_part in ['q', 'k', 'v', 'out']:
                                clip_attn_part = get_module_by_name(clip_submodule, submodule_info[attn_part]['clip'])
                                dinov2_attn_part = get_module_by_name(dinov2_submodule, submodule_info[attn_part]['dinov2'])
                                merged_attn_part = get_module_by_name(merged_submodule, submodule_info[attn_part]['clip'])
                                merge_parameters(merged_attn_part, clip_attn_part, dinov2_attn_part, weight)
                        elif submodule == "mlp":
                            for mlp_part in ['fc1', 'fc2']:
                                clip_mlp_part = getattr(clip_submodule, submodule_info[mlp_part]['clip'])
                                dinov2_mlp_part = getattr(dinov2_submodule, submodule_info[mlp_part]['dinov2'])
                                merged_mlp_part = getattr(merged_submodule, submodule_info[mlp_part]['clip'])
                                merge_parameters(merged_mlp_part, clip_mlp_part, dinov2_mlp_part, weight)
                        else:  # layer norms
                            merge_parameters(merged_submodule, clip_submodule, dinov2_submodule, weight)
                        pbar.update(1)
            else:
                clip_module = get_module_by_name(clip_model, module_info['clip'])
                dinov2_module = get_module_by_name(dinov2_model, module_info['dinov2'])
                merged_module = get_module_by_name(merged_model, module_info['clip'])
                merge_parameters(merged_module, clip_module, dinov2_module, weight)
                pbar.update(1)

    return merged_model

def parse_args():
    parser = argparse.ArgumentParser(description="Merge CLIP and DINOv2 ViT models")
    parser.add_argument("--weight", type=float, default=0.5, help="Weight for CLIP model (0-1). DINOv2 weight will be 1 - weight.")
    parser.add_argument("--output", type=str, default="merged_dinov2_clip_vit", help="Output directory for the merged model")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.weight < 0 or args.weight > 1:
        raise ValueError("Weight must be between 0 and 1")

    dinov2_model, clip_model = load_models()
    merged_model = create_merged_model(clip_model, dinov2_model, args.weight)

    print(f"\nMerged model structure (CLIP weight: {args.weight}, DINOv2 weight: {1 - args.weight}):")
    print(merged_model)

    merged_model.save_pretrained(args.output + "/" + "dinov2_clip_vit")
    print(f"Merged model saved to '{args.output}'")

if __name__ == "__main__":
    main()