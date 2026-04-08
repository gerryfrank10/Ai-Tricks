# Generative AI & Diffusion Models

Generative AI creates new content — images, video, audio, text, code — by learning the underlying data distribution. Diffusion models have become the dominant architecture for high-quality image and video synthesis.

---

## 📖 **Sections**

- [Generative Model Zoo](#generative-model-zoo)
- [Diffusion Models — How They Work](#diffusion-models--how-they-work)
- [Stable Diffusion in Practice](#stable-diffusion-in-practice)
- [ControlNet & Conditioning](#controlnet--conditioning)
- [GANs](#gans)
- [VAEs (Variational Autoencoders)](#vaes-variational-autoencoders)
- [Text-to-Audio](#text-to-audio)
- [Evaluation Metrics](#evaluation-metrics)

---

## 🗂️ **Generative Model Zoo**

| Model Type | Best For | Examples |
|-----------|---------|---------|
| Diffusion | High-quality images/video/audio | DALL-E 3, Stable Diffusion, Sora |
| GANs | Fast generation, face synthesis | StyleGAN3, CycleGAN |
| VAEs | Latent space exploration | VQVAE, DALL-E 1 |
| Flow Models | Exact likelihood, invertible | Glow, RealNVP |
| Autoregressive | Text, code, discrete tokens | GPT, LLaMA, MusicGen |

---

## 🌊 **Diffusion Models — How They Work**

Diffusion models work in two phases:

```
FORWARD PROCESS (Training)
Clean Image → Add Gaussian Noise (T steps) → Pure Noise
x₀ ──noise──► x₁ ──noise──► x₂ ──...──► xT ~ N(0,I)

REVERSE PROCESS (Inference)
Pure Noise → Remove Noise (T steps) → Clean Image
xT ──denoise──► xT-1 ──denoise──► ... ──► x₀
```

The neural network (U-Net) learns to predict the noise added at each step.

### Simple DDPM Implementation

```python
import torch
import torch.nn as nn
import numpy as np

class SimpleDDPM:
    """Denoising Diffusion Probabilistic Model (simplified)."""

    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = timesteps

        # Linear noise schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward process: add noise to image at timestep t."""
        noise = torch.randn_like(x0)
        alpha_bar = self.alphas_cumprod[t].view(-1, 1, 1, 1)

        # q(xₜ | x₀) = N(√ᾱₜ·x₀, (1-ᾱₜ)·I)
        noisy = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise
        return noisy, noise

    def denoise_step(self, model: nn.Module, xt: torch.Tensor, t: int) -> torch.Tensor:
        """One step of reverse process."""
        t_tensor = torch.full((xt.shape[0],), t, dtype=torch.long)

        with torch.no_grad():
            predicted_noise = model(xt, t_tensor)

        alpha = self.alphas[t]
        alpha_bar = self.alphas_cumprod[t]
        beta = self.betas[t]

        # Compute x_{t-1}
        mean = (1 / torch.sqrt(alpha)) * (
            xt - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise
        )

        if t > 0:
            noise = torch.randn_like(xt)
            variance = torch.sqrt(beta) * noise
            return mean + variance
        return mean

    @torch.no_grad()
    def sample(self, model: nn.Module, shape: tuple, device: str = "cpu") -> torch.Tensor:
        """Generate new samples via reverse diffusion."""
        x = torch.randn(shape).to(device)

        for t in reversed(range(self.T)):
            x = self.denoise_step(model, x, t)

        return torch.clamp(x, -1, 1)

# Training step
ddpm = SimpleDDPM()

def train_step(model, optimizer, batch):
    x0 = batch.to(device)

    # Random timestep
    t = torch.randint(0, ddpm.T, (x0.shape[0],), device=device)

    # Add noise
    xt, noise = ddpm.add_noise(x0, t)

    # Predict noise
    predicted_noise = model(xt, t)
    loss = nn.MSELoss()(predicted_noise, noise)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

---

## 🖼️ **Stable Diffusion in Practice**

### Text-to-Image with Diffusers

```python
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from PIL import Image

# Load pipeline
model_id = "stabilityai/stable-diffusion-2-1"

pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")

# Memory optimization
pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()

# Generate image
result = pipe(
    prompt="A serene Japanese garden at sunset, cherry blossoms falling, "
           "koi pond reflecting golden light, photorealistic, 8k",
    negative_prompt="blurry, low quality, deformed, ugly, oversaturated",
    num_inference_steps=25,    # More steps = higher quality but slower
    guidance_scale=7.5,         # Higher = more prompt-adherent, less creative
    height=768,
    width=768,
    num_images_per_prompt=4,    # Generate multiple options
    generator=torch.Generator().manual_seed(42),  # Reproducible
)

for i, img in enumerate(result.images):
    img.save(f"output_{i}.png")
```

### Image-to-Image

```python
from diffusers import StableDiffusionImg2ImgPipeline

pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16
).to("cuda")

init_image = Image.open("sketch.jpg").convert("RGB").resize((768, 768))

result = pipe_img2img(
    prompt="Professional oil painting in the style of Van Gogh, vibrant colors, thick brushstrokes",
    image=init_image,
    strength=0.75,       # 0 = keep original, 1 = completely regenerate
    guidance_scale=7.5,
    num_inference_steps=30,
)
result.images[0].save("stylized.png")
```

### Inpainting

```python
from diffusers import StableDiffusionInpaintPipeline

pipe_inpaint = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

# Load image and mask (white = area to regenerate)
image = Image.open("photo.jpg").resize((512, 512))
mask = Image.open("mask.png").resize((512, 512))  # Black=keep, White=regenerate

result = pipe_inpaint(
    prompt="A beautiful garden with colorful flowers",
    image=image,
    mask_image=mask,
    num_inference_steps=50,
)
result.images[0].save("inpainted.png")
```

---

## 🎮 **ControlNet & Conditioning**

ControlNet adds structural control to diffusion — guide generation with edges, poses, depth maps, etc.

```python
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import cv2
import numpy as np

# Load Canny edge ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
).to("cuda")

# Extract edge map from control image
control_image = load_image("building.jpg")
control_np = np.array(control_image)
edges = cv2.Canny(control_np, 100, 200)
edges_rgb = Image.fromarray(np.stack([edges, edges, edges], axis=2))

# Generate conditioned on edges
result = pipe(
    prompt="A futuristic cyberpunk building, neon lights, rain, cinematic lighting",
    image=edges_rgb,
    controlnet_conditioning_scale=0.8,  # 0=ignore control, 1=strict control
    num_inference_steps=30,
    guidance_scale=7.5,
)
result.images[0].save("controlled_output.png")
```

---

## 🤖 **GANs**

Generative Adversarial Networks pit a generator against a discriminator.

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim: int = 100, img_channels: int = 3, features: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, features * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.ReLU(True),
            # 512 x 4 x 4
            nn.ConvTranspose2d(features * 8, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.ReLU(True),
            # 256 x 8 x 8
            nn.ConvTranspose2d(features * 4, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.ReLU(True),
            # 128 x 16 x 16
            nn.ConvTranspose2d(features * 2, features, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(True),
            # 64 x 32 x 32
            nn.ConvTranspose2d(features, img_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 3 x 64 x 64
        )

    def forward(self, z):
        return self.model(z.view(z.size(0), -1, 1, 1))

class Discriminator(nn.Module):
    def __init__(self, img_channels: int = 3, features: int = 64):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(img_channels, features, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features, features * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 2, features * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 4, features * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1)

# Training loop
def train_gan(generator, discriminator, dataloader, n_epochs: int = 100):
    opt_G = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(discriminator.parameters(), lr=2e-4, betas=(0.5, 0.999))
    criterion = nn.BCELoss()
    latent_dim = 100

    for epoch in range(n_epochs):
        for real_imgs in dataloader:
            batch_size = real_imgs.size(0)
            real_labels = torch.ones(batch_size).to(device)
            fake_labels = torch.zeros(batch_size).to(device)

            # --- Train Discriminator ---
            opt_D.zero_grad()
            d_real_loss = criterion(discriminator(real_imgs), real_labels)

            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z).detach()
            d_fake_loss = criterion(discriminator(fake_imgs), fake_labels)

            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            opt_D.step()

            # --- Train Generator ---
            opt_G.zero_grad()
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_imgs = generator(z)
            g_loss = criterion(discriminator(fake_imgs), real_labels)  # Fool discriminator
            g_loss.backward()
            opt_G.step()

        print(f"Epoch [{epoch}/{n_epochs}] D_loss: {d_loss:.4f}, G_loss: {g_loss:.4f}")
```

---

## 🧬 **VAEs (Variational Autoencoders)**

VAEs learn a structured latent space — enabling interpolation between concepts.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim: int = 784, latent_dim: int = 20):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * std"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple:
        mu, logvar = self.encode(x.view(x.size(0), -1))
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

def vae_loss(recon_x, x, mu, logvar, beta: float = 1.0):
    """ELBO loss: reconstruction + KL divergence."""
    recon_loss = F.binary_cross_entropy(recon_x, x.view(x.size(0), -1), reduction="sum")
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_loss  # beta-VAE: beta > 1 for disentanglement

# Latent space interpolation
def interpolate(vae, img1, img2, n_steps: int = 10) -> list:
    """Smoothly interpolate between two images in latent space."""
    mu1, _ = vae.encode(img1.view(1, -1))
    mu2, _ = vae.encode(img2.view(1, -1))

    interpolated = []
    for alpha in torch.linspace(0, 1, n_steps):
        z = (1 - alpha) * mu1 + alpha * mu2
        img = vae.decode(z)
        interpolated.append(img)

    return interpolated
```

---

## 🎵 **Text-to-Audio**

```python
# pip install transformers scipy
from transformers import pipeline
import scipy.io.wavfile

# AudioLDM2: text-to-sound
audio_pipe = pipeline(
    "text-to-audio",
    model="cvssp/audioldm2",
    device="cuda",
)

prompts = [
    "A rainy day in a coffee shop with jazz music playing softly",
    "Thunderstorm with heavy rain on a metal roof",
    "Busy city street with cars and birds chirping",
]

for i, prompt in enumerate(prompts):
    audio = audio_pipe(
        prompt,
        forward_params={
            "num_inference_steps": 200,
            "audio_length_in_s": 10.0,
        }
    )
    scipy.io.wavfile.write(
        f"audio_{i}.wav",
        rate=audio["sampling_rate"],
        data=audio["audio"]
    )
    print(f"Generated: audio_{i}.wav | Prompt: {prompt[:50]}")
```

---

## 📏 **Evaluation Metrics**

```python
import torch
from torchvision.models import inception_v3
from scipy.linalg import sqrtm
import numpy as np

def calculate_fid(real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
    """
    Frechet Inception Distance (FID) — lower is better.
    Measures distribution similarity between real and generated images.
    """
    inception = inception_v3(pretrained=True, transform_input=False).eval()

    def get_features(images):
        with torch.no_grad():
            features = inception(images)
        return features.numpy()

    real_features = get_features(real_images)
    fake_features = get_features(fake_images)

    mu_r, sigma_r = real_features.mean(0), np.cov(real_features, rowvar=False)
    mu_f, sigma_f = fake_features.mean(0), np.cov(fake_features, rowvar=False)

    diff = mu_r - mu_f
    covmean = sqrtm(sigma_r @ sigma_f)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean)
    return float(fid)

# Reference FID scores (lower = better)
# DDPM on CIFAR-10:   3.17
# StyleGAN2 on FFHQ:  2.84
# Stable Diffusion:   ~8-12 (text-to-image task is harder)
```

---

## 💡 **Tips & Tricks**

1. **Classifier-free guidance (CFG)**: Scale 7-9 for most prompts; lower for creative freedom, higher for adherence
2. **Negative prompts**: Use `"blurry, low quality, deformed, watermark, text"` as a baseline negative prompt
3. **DDIM vs DDPM sampling**: DDIM is 10-50x faster with minimal quality loss — use it for inference
4. **VAE latent space**: For 512×512 images, Stable Diffusion works in 64×64 latent space (8x compression)
5. **Seed consistency**: Fix the seed for reproducible results; vary seed for diversity
6. **Mode collapse in GANs**: If the generator produces only one type of image, increase diversity loss or use minibatch discrimination

---

## 🔗 **Related Topics**

- [Deep Learning](../Deep%20Learning/Introduction.md)
- [Computer Vision](../Computer%20Vision/Image-Classification.md)
- [Multimodal AI](../Multimodal-AI/README.md)
- [Fine-Tuning LLMs](../Fine-Tuning/README.md)
