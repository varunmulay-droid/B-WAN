import torch, numpy as np, gc, sys, os, random, imageio
from PIL import Image

sys.path.insert(0, './ComfyUI')

from nodes import (
    CLIPLoader, CLIPTextEncode,
    VAEDecode, VAELoader, KSampler
)

from custom_nodes.ComfyUI_GGUF.nodes import UnetLoaderGGUF
from comfy_extras.nodes_model_advanced import ModelSamplingSD3

# =========================
# SETTINGS
# =========================
useQ6 = False

# =========================
# INIT NODES
# =========================
clip_loader = CLIPLoader()
clip_encode = CLIPTextEncode()
vae_loader = VAELoader()
ksampler = KSampler()
vae_decode = VAEDecode()
unet_loader = UnetLoaderGGUF()
model_sampling = ModelSamplingSD3()

# =========================
# UTILS
# =========================
def clear():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def save_video(frames, fps=16):
    os.makedirs("output", exist_ok=True)
    path = "output/result.mp4"

    frames = [(f.cpu().numpy()*255).astype(np.uint8) for f in frames]

    with imageio.get_writer(path, fps=fps) as w:
        for f in frames:
            w.append_data(f)

    print("Saved:", path)

# =========================
# PIPELINE
# =========================
def generate_video(
    prompt="A cinematic drone shot flying over mountains at sunset",
    negative="low quality, blurry",
    width=512,
    height=512,
    frames=24,
    steps=20,
    cfg=4,
    fps=16,
    seed=None
):

    seed = seed or random.randint(0,2**32-1)
    print("Seed:", seed)

    # TEXT ENCODING
    clip = clip_loader.load_clip(
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "wan","default"
    )[0]

    pos = clip_encode.encode(clip, prompt)[0]
    neg = clip_encode.encode(clip, negative)[0]

    del clip; clear()

    # LOAD VAE
    vae = vae_loader.load_vae("wan_2.1_vae.safetensors")[0]

    # RANDOM LATENT VIDEO
    latent = torch.randn(
        (frames, 4, height//8, width//8),
        device="cuda"
    )

    # LOAD UNET
    model_name = (
        "wan2.1-i2v-14b-480p-Q6_K.gguf"
        if useQ6 else
        "wan2.1-i2v-14b-480p-Q4_0.gguf"
    )

    model = unet_loader.load_unet(model_name)[0]
    model = model_sampling.patch(model,8)[0]

    # SAMPLING
    sampled = ksampler.sample(
        model=model,
        seed=seed,
        steps=steps,
        cfg=cfg,
        sampler_name="uni_pc",
        scheduler="simple",
        positive=pos,
        negative=neg,
        latent_image=latent
    )[0]

    decoded = vae_decode.decode(vae, sampled)[0]

    save_video(decoded, fps)
    clear()

if __name__ == "__main__":
    generate_video()
