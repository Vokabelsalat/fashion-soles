"""Synthetic fabric swatches with folds, noise, blur, and JPEG, used to calibrate and test texture.py."""
import io, math, random
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

W, H = 300, 400

def finish(img, seed):
    rng = np.random.default_rng(seed)
    arr = np.asarray(img).astype(float)
    y, x = np.mgrid[0:H, 0:W]
    folds = 1 + .12 * np.sin(x / 37 + y / 61) + .08 * np.sin(y / 23 - x / 90)  # Soft shading from folds.
    arr = arr * folds[..., None] + rng.normal(0, 4, arr.shape)
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(.6))
    buf = io.BytesIO(); out.save(buf, "JPEG", quality=85)
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")

def mask():
    m = Image.new("L", (W, H), 0)
    ImageDraw.Draw(m).rounded_rectangle((20, 20, W - 20, H - 20), radius=40, fill=255)
    return m

def plain(c=(70, 90, 140)):
    return Image.new("RGB", (W, H), c)

def stripes(angle=0, width=12, a=(240, 240, 235), b=(30, 50, 110)):
    img = Image.new("RGB", (W * 2, H * 2), a); d = ImageDraw.Draw(img)
    for y in range(0, H * 2, width * 2):
        d.rectangle((0, y, W * 2, y + width - 1), fill=b)
    return img.rotate(angle, resample=Image.BICUBIC).crop((W // 2, H // 2, W // 2 + W, H // 2 + H))

def gingham(size=14):
    arr = np.full((H, W, 3), 245.0)
    y, x = np.mgrid[0:H, 0:W]
    v = (x // size) % 2 == 0; h = (y // size) % 2 == 0
    red = np.array([200, 40, 50.0])
    arr[v | h] = arr[v | h] * .45 + red * .55
    arr[v & h] = red
    return Image.fromarray(arr.astype(np.uint8))

def dots(r=6, step=28):
    img = Image.new("RGB", (W, H), (25, 30, 60)); d = ImageDraw.Draw(img)
    for row, y in enumerate(range(0, H + step, step)):
        for x in range((row % 2) * step // 2, W + step, step):
            d.ellipse((x - r, y - r, x + r, y + r), fill=(240, 240, 240))
    return img

def floral(seed=1):
    rnd = random.Random(seed)
    img = Image.new("RGB", (W, H), (245, 235, 215)); d = ImageDraw.Draw(img)
    for _ in range(45):
        cx, cy = rnd.randrange(W), rnd.randrange(H); petal = rnd.choice([(200, 60, 90), (230, 150, 40), (150, 70, 160)])
        for k in range(5):
            ang = k * 72 + rnd.random() * 20
            px, py = cx + 11 * math.cos(math.radians(ang)), cy + 11 * math.sin(math.radians(ang))
            d.ellipse((px - 8, py - 6, px + 8, py + 6), fill=petal)
        d.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=(240, 200, 60))
        lx, ly = cx + rnd.randint(-25, 25), cy + rnd.randint(15, 30)
        d.ellipse((lx - 10, ly - 4, lx + 10, ly + 4), fill=(60, 120, 60))
    return img

def geometric(seed=2):
    rnd = random.Random(seed)
    img = Image.new("RGB", (W, H), (235, 230, 220)); d = ImageDraw.Draw(img)
    for _ in range(40):
        x, y, s = rnd.randrange(W), rnd.randrange(H), rnd.randint(15, 40)
        color = rnd.choice([(20, 20, 20), (210, 60, 40), (40, 90, 170)])
        if rnd.random() < .5:
            d.rectangle((x, y, x + s, y + s // 2), fill=color)
        else:
            d.polygon([(x, y), (x + s, y), (x + s // 2, y + s)], fill=color)
    return img

def graphic():
    img = Image.new("RGB", (W, H), (235, 228, 215)); d = ImageDraw.Draw(img)
    cx, cy = W // 2, H // 2 - 30
    d.ellipse((cx - 45, cy - 45, cx + 45, cy + 45), outline=(60, 50, 50), width=4)
    d.line((cx - 60, cy + 60, cx + 60, cy + 60), fill=(60, 50, 50), width=4)
    d.text((cx - 30, cy - 8), "ANGEL", fill=(60, 50, 50))
    for k in range(6):
        d.arc((cx - 70 + k * 6, cy - 70 + k * 4, cx + 70 - k * 6, cy + 20), 200, 340, fill=(80, 70, 70), width=2)
    return img

def knit(seed=3):
    rng = np.random.default_rng(seed)
    base = np.full((H, W, 3), (140, 110, 90), float)
    y, x = np.mgrid[0:H, 0:W]
    ribs = 10 * np.sin(x * 2 * np.pi / 5) + rng.normal(0, 10, (H, W))
    return Image.fromarray(np.clip(base + ribs[..., None], 0, 255).astype(np.uint8))

SWATCHES = {"plain": plain(), "stripes horizontal": stripes(0), "stripes diagonal": stripes(30),
            "pinstripe": stripes(0, width=4, a=(40, 40, 50), b=(200, 200, 200)), "gingham": gingham(),
            "polka dots": dots(), "floral": floral(), "geometric": geometric(), "graphic": graphic(), "knit ribs": knit()}
