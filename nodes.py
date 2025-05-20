import argparse
import logging
import os
import sys
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

import random
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

import wan
from wan.configs import (MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES,
                         WAN_CONFIGS)
from wan.utils.prompt_extend import QwenPromptExpander
from wan.utils.utils import cache_image, cache_video, str2bool


class LoadMoviiGenModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": ("STRING", {"default": "./MoviiGen1.1"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = "MoviiGen-1.1"

    def load_model(self, model_path):
        model = model_path
        return (model,)


class Prompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {
                    "default": "Inside a smoky, atmospheric private eye office bathed in dramatic film noir lighting, sharp shadows from slatted blinds cut across a cluttered desk and worn surroundings, evoking the classic style by 1940s film. A world-weary detective is sitting behind the desk. He is smoking a cigarette, slowly bringing it to his lips, inhaling, and exhaling a plume of smoke that drifts in the harsh, directional light. The scene is rendered in stark black and white, creating a high-contrast, cinematic mood. The camera holds a static medium shot focused on the detective, emphasizing the gritty texture and oppressive atmosphere.",
                    "multiline": True
                }),
            }
        }

    RETURN_TYPES = ("PROMPT",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "input_text"
    CATEGORY = "MoviiGen-1.1"

    def input_text(self, text):
        prompt = text
        return (prompt,)


class MoviiGen:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "task": ("STRING", {"default": "t2v-14B"}),
                "ckpt_dir": ("MODEL",),
                "prompt": ("PROMPT",),
                "size": ("STRING", {"default": "1280*720"}),
                "frame_num": ("INT", {"default": 81}),
                "ulysses_size": ("INT", {"default": 4}),
                "ring_size": ("INT", {"default": 1}),
                "base_seed": ("INT", {"default": "42"}),
                "sample_solver": ("STRING", {"default": ['unipc', 'dpm++']}),
                "sample_steps": ("INT", {"default": 50}),
                "sample_shift": ("FLOAT", {"default": 5}),
                "sample_guide_scale": ("FLOAT", {"default": 5.0}),
            }
        }

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION = "generate"
    CATEGORY = "MoviiGen-1.1"

    def generate(self, task, size, frame_num, ckpt_dir, offload_model, ulysses_size, ring_size, t5_fsdp, t5_cpu,
                dit_fsdp, prompt, base_seed, sample_solver, sample_steps, sample_shift, sample_guide_scale):
      
        rank = int(os.getenv("RANK", 0))
        world_size = int(os.getenv("WORLD_SIZE", 1))
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        device = local_rank
    
        if offload_model is None:
            offload_model = False if world_size > 1 else True
    
        if world_size > 1:
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                rank=rank,
                world_size=world_size)
        else:
            assert not (
                t5_fsdp or dit_fsdp
            ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
            assert not (
                ulysses_size > 1 or ring_size > 1
            ), f"context parallel are not supported in non-distributed environments."
    
        if ulysses_size > 1 or ring_size > 1:
            assert ulysses_size * ring_size == world_size, f"The number of ulysses_size and ring_size should be equal to the world size."
            from xfuser.core.distributed import (
                init_distributed_environment,
                initialize_model_parallel,
            )
            init_distributed_environment(
                rank=dist.get_rank(), world_size=dist.get_world_size())
    
            initialize_model_parallel(
                sequence_parallel_degree=dist.get_world_size(),
                ring_degree=ring_size,
                ulysses_degree=ulysses_size,
            )
    
        if use_prompt_extend:
            prompt_expander = QwenPromptExpander(
                model_name=args.prompt_extend_model,
                is_vl=False,
                device=rank)
    
        cfg = WAN_CONFIGS[task]
        if ulysses_size > 1:
            assert cfg.num_heads % ulysses_size == 0, f"`{cfg.num_heads=}` cannot be divided evenly by `{ulysses_size=}`."
   
        if dist.is_initialized():
            base_seed = [args.base_seed] if rank == 0 else [None]
            dist.broadcast_object_list(base_seed, src=0)
            base_seed = base_seed[0]
    
        if "t2v" in args.task or "t2i" in task:
            if prompt is None:
                prompt = EXAMPLE_PROMPT[task]["prompt"]
            
            if use_prompt_extend:
                if rank == 0:
                    prompt_output = prompt_expander(
                        prompt,
                        tar_lang=prompt_extend_target_lang,
                        seed=base_seed)
                    if prompt_output.status == False:
                        input_prompt = prompt
                    else:
                        input_prompt = prompt_output.prompt
                    input_prompt = [input_prompt]
                else:
                    input_prompt = [None]
                if dist.is_initialized():
                    dist.broadcast_object_list(input_prompt, src=0)
                prompt = input_prompt[0]
    
            wan_t2v = wan.WanT2V(
                config=cfg,
                checkpoint_dir=ckpt_dir,
                device_id=device,
                rank=rank,
                t5_fsdp=t5_fsdp,
                dit_fsdp=dit_fsdp,
                use_usp=(ulysses_size > 1 or ring_size > 1),
                t5_cpu=t5_cpu,
            )
    
            video = wan_t2v.generate(
                prompt,
                size=SIZE_CONFIGS[size],
                frame_num=frame_num,
                shift=sample_shift,
                sample_solver=sample_solver,
                sampling_steps=sample_steps,
                guide_scale=sample_guide_scale,
                seed=base_seed,
                offload_model=offload_model)
    
        return (video,)

