from .nodes import LoadMoviiGenModel, Prompt, MoviiGen, SaveMoviiGen

NODE_CLASS_MAPPINGS = {
    "LoadMoviiGenModel": LoadMoviiGenModel,
    "Prompt": Prompt,
    "MoviiGen": MoviiGen,
    "SaveMoviiGen": SaveMoviiGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadMoviiGenModel": "Load MoviiGen Model",
    "Prompt": "Prompt",
    "MoviiGen": "MoviiGen",
    "SaveMoviiGen": "Save MoviiGen",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
