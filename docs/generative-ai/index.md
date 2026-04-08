# Generative AI

Generative AI creates new content — images, video, audio, text, code, 3D — by learning the underlying data distribution. It's the fastest-moving area of AI.

---

## 2025 Landscape

| Modality | Leading Models | Key Capability |
|---------|---------------|----------------|
| Text | Claude, GPT-4o, Gemini, Llama | Reasoning, code, analysis |
| Image | Flux.1, SD3.5, DALL-E 3, Midjourney v7 | Photorealistic, artistic |
| Video | Sora, Kling 2.0, Runway Gen-3 | Up to 2min HD video |
| Audio | ElevenLabs, Suno v4, MusicGen | Voice cloning, music |
| Code | Claude, GitHub Copilot, Cursor | Full-repo understanding |
| 3D | TripoSG, Shap-E | Mesh + texture generation |
| Multimodal | GPT-4o, Claude, Gemini | See, hear, read, generate |

---

## Generative Model Taxonomy

```
Generative Models
├── Autoregressive     — GPT, LLaMA (token-by-token)
├── Diffusion          — Stable Diffusion, FLUX (noise→image)
├── GANs               — StyleGAN (generator vs discriminator)
├── VAEs               — Latent space interpolation
├── Flow Models        — Exact likelihood, invertible
└── State Space Models — Mamba (linear-time sequence modeling)
```

---

## Topics

| Page | Description |
|------|-------------|
| [Diffusion Models](diffusion-models.md) | Stable Diffusion, ControlNet, GANs, VAEs, text-to-audio |
| [Multimodal AI](multimodal.md) | Vision-language models, CLIP, video, document understanding |

---

## Quick Start — Text-to-Image

```python
from diffusers import StableDiffusion3Pipeline
import torch

pipe = StableDiffusion3Pipeline.from_pretrained(
    "stabilityai/stable-diffusion-3.5-large",
    torch_dtype=torch.bfloat16,
)
pipe = pipe.to("cuda")

image = pipe(
    prompt="A serene Japanese garden at golden hour, photorealistic, 8k",
    negative_prompt="blurry, low quality, watermark",
    num_inference_steps=28,
    guidance_scale=4.5,
    height=1024, width=1024,
).images[0]

image.save("garden.png")
```

---

## Related Topics

- [LLMs & Agents](../llm/index.md)
- [Computer Vision](../computer-vision/index.md)
- [Deep Learning](../deep-learning/index.md)
