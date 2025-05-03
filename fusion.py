import time
import io
import serial
import mss
import numpy as np
import torch
import sounddevice as sd
from PIL import Image
import matplotlib.pyplot as plt
from transformers import AutoProcessor, AutoModel
import clip
from pynput import mouse
from threading import Lock

# === Serial Communication ===
SERIAL_PORT_OUTPUT = 'COM14'
BAUD_RATE = 115200
ser_output = serial.Serial(SERIAL_PORT_OUTPUT, BAUD_RATE, timeout=1)

# === CLIP (Image) ===
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

player_violence_level_prompts_dict = {
    0: [
        "A first-person view showing no player-initiated violence. The player is not performing any aggressive actions.",
        "A calm scene from the player's perspective with no signs of the player engaging in combat or violent interactions.",
        "No violence is caused by the player; the scene is peaceful from the player's own actions."
    ],
    1: [
        "A very light player interaction: a brief, non-damaging tap or a subtle gesture that is minimally aggressive, initiated by the player.",
        "The player shows a minor aggressive action, such as a light punch or a slight trigger pull, with negligible harmful effect.",
        "A minimal player-initiated action that barely qualifies as violence, with almost no damage inflicted."
    ],
    2: [
        "A moderate player-initiated act of violence, such as a solid punch or a controlled gunshot that causes minor damage.",
        "The player engages in a moderate level of aggression, delivering a noticeable punch or firing a single shot that causes some harm.",
        "A player action resulting in moderate violence, like a well-timed punch or a controlled firearm use, causing limited damage."
    ],
    3: [
        "A high level of violence initiated by the player, such as a powerful, damaging punch or a forceful gunshot that inflicts significant harm.",
        "The player actively engages in combat with aggressive actions—rapid firing or a strong melee attack—that cause notable damage.",
        "Intense player-initiated violence where the player's actions result in substantial harm, such as a series of strong gunshots or a forceful strike."
    ],
    4: [
        "An extremely violent action initiated by the player, such as triggering a large explosion or unleashing overwhelming firepower that causes massive destruction.",
        "The player engages in extreme violence, for example causing an explosion or using heavy weaponry in a way that results in catastrophic damage.",
        "A top-tier display of player-initiated violence, where the player's actions lead to devastating impact—such as an explosive attack or relentless assault."
    ]
}

interaction_violence_embeddings = []
for level in sorted(player_violence_level_prompts_dict.keys()):
    tokens = clip.tokenize(player_violence_level_prompts_dict[level]).to(device)
    with torch.no_grad():
        embeds = clip_model.encode_text(tokens)
        embeds /= embeds.norm(dim=-1, keepdim=True)
        avg_embed = embeds.mean(dim=0, keepdim=True)
        avg_embed /= avg_embed.norm(dim=-1, keepdim=True)
    interaction_violence_embeddings.append(avg_embed)
interaction_violence_embeddings = torch.cat(interaction_violence_embeddings, dim=0)

def query_clip_image_score(img_pil):
    image_input = clip_preprocess(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embed = clip_model.encode_image(image_input)
        image_embed /= image_embed.norm(dim=-1, keepdim=True)
        sim = (100.0 * image_embed @ interaction_violence_embeddings.T).softmax(dim=-1)
        categories = torch.arange(0, 5).float().to(device)
        score = (sim * categories).sum().item()
    return score

# === CLAP (Audio) ===
model_id = "laion/clap-htsat-unfused"
processor = AutoProcessor.from_pretrained(model_id)
clap_model = AutoModel.from_pretrained(model_id).to(device)

violence_prompts = [
    "player gunshot", "player weapon fire", "close-up gunshot", "shooting weapon",
    "machine gun fire nearby", "player reloading weapon", "weapon handling sounds",
    "player melee attack sound", "player grenade explosion", "close explosion",
    "player ability activation sound", "bullet impact sound nearby",
    "gunshot", "gunfire", "explosion", "loud bang",
    "intense combat sounds", "battle sounds",
    "player character grunt during combat", "aggressive player shout"
]

with torch.no_grad():
    text_inputs = processor(text=violence_prompts, return_tensors="pt", padding=True).to(device)
    text_embeds = clap_model.get_text_features(**text_inputs)
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)


# === Fusion Weights ===
IMAGE_WEIGHT = 0.4
AUDIO_WEIGHT = 0.4
INTERACTION_WEIGHT = 0.2

# === History Buffers ===
history_time = []
history_fused_score = []
history_image_score = []
history_audio_score = []
history_bio_score = []

# === Screen & Audio Capture ===
def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb)
        return img

def record_audio_segment(duration=0.5, samplerate=48000):
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    return np.squeeze(audio)

def query_clap(audio_segment, samplerate=48000):
    audio_inputs = processor(audios=[audio_segment], sampling_rate=samplerate, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        audio_features = clap_model.get_audio_features(**audio_inputs)
        audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        similarity = torch.matmul(audio_features, text_embeds.T)
        violence_score = torch.max(similarity).item()
    return violence_score

# === Mouse Fire Tracker ===
fire_count = 0
fire_duration = 0.0
_current_press_time = None
_lock = Lock()

def on_click(x, y, button, pressed):
    global fire_count, fire_duration, _current_press_time
    if button == mouse.Button.left:
        with _lock:
            if pressed:
                fire_count += 1
                _current_press_time = time.time()
            else:
                if _current_press_time is not None:
                    fire_duration += time.time() - _current_press_time
                    _current_press_time = None

def reset_mouse_stats():
    global fire_count, fire_duration
    with _lock:
        count = fire_count
        duration = fire_duration
        fire_count = 0
        fire_duration = 0.0
    return count, duration

def compute_interaction_score(fire_count, fire_duration):
    count_score = min(fire_count / 5, 1.0)
    duration_score = min(fire_duration / 0.9, 1.0)
    return (count_score + duration_score) / 2 * 4

def start_mouse_listener():
    listener = mouse.Listener(on_click=on_click)
    listener.daemon = True
    listener.start()

start_mouse_listener()
print("🖱️ Mouse fire listener started.")

# === Main Loop ===
print("🎮 Starting real-time violence estimation (4-modality)...")

bpm_prev, gsr_prev = None, None
t0 = time.time()

try:
    while True:
        start_time = time.time()

        # 1. Image
        img = capture_screen()
        image_score = query_clip_image_score(img)

        # 2. Audio
        audio_segment = record_audio_segment(duration=0.5)
        audio_score = query_clap(audio_segment)

        # 3. Interaction
        fire_cnt, fire_dur = reset_mouse_stats()
        interaction_score = compute_interaction_score(fire_cnt, fire_dur)

        # 4. Fusion
        fused_score = (
            IMAGE_WEIGHT * image_score +
            AUDIO_WEIGHT * audio_score +
            INTERACTION_WEIGHT * interaction_score
        )

        # 6. Trigger hardware if needed
        if fused_score >= 2.0:
            ser_output.write(b"T\n")

        # 7. Log & Print
        now = time.time() - t0
        history_time.append(now)
        history_image_score.append(image_score)
        history_audio_score.append(audio_score)
        history_fused_score.append(fused_score)

        print(f"[{now:.1f}s] Image: {image_score:.2f} | Audio: {audio_score:.2f} | Fire: {interaction_score:.2f} --> Fused: {fused_score:.2f}")

        elapsed = time.time() - start_time
        time.sleep(max(0.5 - elapsed, 0.01))

except KeyboardInterrupt:
    print("\n🛑 Program stopped.")
    ser_output.close()
