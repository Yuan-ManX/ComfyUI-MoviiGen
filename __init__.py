from .nodes import LoadMoviiGenModel, MoviiGenPrompt, MoviiGen, SaveMoviiGen

NODE_CLASS_MAPPINGS = {
    "LoadMoviiGenModel": LoadMoviiGenModel,
    "MoviiGenPrompt": MoviiGenPrompt,
    "MoviiGen": MoviiGen,
    "SaveMoviiGen": SaveMoviiGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadMoviiGenModel": "Load MoviiGen Model",
    "MoviiGenPrompt": "MoviiGen Prompt",
    "MoviiGen": "MoviiGen",
    "SaveMoviiGen": "Save MoviiGen",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
