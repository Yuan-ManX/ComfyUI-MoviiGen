from .nodes import LoadMoviiGenModel, Prompt, MoviiGen

NODE_CLASS_MAPPINGS = {
    "LoadMoviiGenModel": LoadMoviiGenModel,
    "Prompt": Prompt,
    "MoviiGen": MoviiGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadMoviiGenModel": "Load MoviiGen Model",
    "Prompt": "Prompt",
    "MoviiGen": "MoviiGen",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
