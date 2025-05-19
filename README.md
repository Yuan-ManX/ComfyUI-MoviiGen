# ComfyUI-MoviiGen

ComfyUI-MoviiGen is now available in ComfyUI, MoviiGen 1.1 is a cutting-edge video generation model that excels in cinematic aesthetics and visual quality.

[MoviiGen 1.1](https://github.com/ZulutionAI/MoviiGen1.1): Towards Cinematic-Quality Video Generative Models. MoviiGen 1.1 is a cutting-edge video generation model that excels in cinematic aesthetics and visual quality. This model is a fine-tuning model based on the Wan2.1. Based on comprehensive evaluations by 11 professional filmmakers and AIGC creators, including industry experts, across 60 aesthetic dimensions, MoviiGen 1.1 demonstrates superior performance in key cinematic aspects.



## Installation

1. Make sure you have ComfyUI installed

2. Clone this repository into your ComfyUI's custom_nodes directory:
```
cd ComfyUI/custom_nodes
git clone https://github.com/Yuan-ManX/ComfyUI-MoviiGen.git
```

3. Install dependencies:
```
cd ComfyUI-MoviiGen
# Ensure torch >= 2.4.0
pip install -r requirements.txt
```


## Model


### Model Download

T2V-14B  Model: 🤗 [Huggingface](https://huggingface.co/ZuluVision/MoviiGen1.1) 
MoviiGen1.1 model supports both 720P and 1080P. For more cinematic quality, we recommend using 1080P and a 21:9 aspect ratio (1920*832).

Download models using huggingface-cli:
```
pip install "huggingface_hub[cli]"
huggingface-cli download ZuluVision/MoviiGen1.1 --local-dir ./MoviiGen1.1
```


## Prompt Tips:

- **Prompt Length**: The prompt length should be around 100~200.
  
- **Prompt Content**: The prompt should contain **scene description**, **main subject**, **events**, **aesthetics description** and **camera movement**.

- **Example**: 
```
Scene Description: A smoky, atmospheric private eye office bathed in dramatic film noir lighting, sharp shadows from slatted blinds cut across a cluttered desk and worn surroundings, evoking the classic style by 1940s film.
Main Subject: A world-weary detective is sitting behind the desk.
Events: He is smoking a cigarette, slowly bringing it to his lips, inhaling, and exhaling a plume of smoke that drifts in the harsh, directional light.
Aesthetics Description: The scene is rendered in stark black and white, creating a high-contrast, cinematic mood.
Camera Movement: The camera holds a static medium shot focused on the detective, emphasizing the gritty texture and oppressive atmosphere.

Final Prompt:
A smoky, atmospheric private eye office bathed in dramatic film noir lighting, sharp shadows from slatted blinds cut across a cluttered desk and worn surroundings, evoking the classic style by 1940s film. A world-weary detective is sitting behind the desk. He is smoking a cigarette, slowly bringing it to his lips, inhaling, and exhaling a plume of smoke that drifts in the harsh, directional light. The scene is rendered in stark black and white, creating a high-contrast, cinematic mood. The camera holds a static medium shot focused on the detective, emphasizing the gritty texture and oppressive atmosphere.
```
