import time
import mss
import numpy as np
import torch
from PIL import Image
import clip
import cv2
import mediapipe as mp
import sounddevice as sd
from pynput import mouse
from threading import Lock
from transformers import AutoProcessor, AutoModel
import matplotlib.pyplot as plt
from datetime import datetime
import os

# === [GAME] CLIP (Image) [GAME] ===
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

def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", (screenshot.width, screenshot.height), screenshot.rgb)
        return img

# === [GAME] CLAP (Audio) [GAME] ===
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


# === [PLAYER] Mediapipe (Facial Expression) [PLAYER] ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Landmark indices for mouth and eyebrows
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_BROW = 70
LEFT_EYE = 159

def analyze_face_expression(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if not results.multi_face_landmarks:
        return 0  # No face detected

    face_landmarks = results.multi_face_landmarks[0]
    h, w, _ = image.shape

    # Get coordinates
    upper_lip = face_landmarks.landmark[UPPER_LIP]
    lower_lip = face_landmarks.landmark[LOWER_LIP]
    mouth_gap = abs((lower_lip.y - upper_lip.y) * h)

    left_brow = face_landmarks.landmark[LEFT_BROW]
    left_eye = face_landmarks.landmark[LEFT_EYE]
    brow_eye_gap = abs((left_brow.y - left_eye.y) * h)

    mouth_open = mouth_gap > 15  # Adjust threshold as needed
    brow_lifted = brow_eye_gap > 20  # Adjust threshold as needed

    expression_score = 0
    if mouth_open:
        expression_score += 2
    if brow_lifted:
        expression_score += 2

    return expression_score  # Range: 0–4


# === [PLAYER] Sounddevice (Audio) [PLAYER] ===
DURATION = 0.5  # seconds
SAMPLERATE = 16000

def capture_audio_energy():
    audio = sd.rec(int(DURATION * SAMPLERATE), samplerate=SAMPLERATE, channels=1)
    sd.wait()
    volume = np.linalg.norm(audio) 
    return volume

def map_audio_to_emotion_score(energy):
    if energy < 1:
        return 0
    elif energy < 5:
        return 1
    elif energy < 10:
        return 2
    elif energy < 20:
        return 3
    else:
        return 4


# === [PLAYER] Mouse Fire Tracker [PLAYER] ===
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


# === Fusion Weights ===
# GAME 游戏场景的暴力程度
IMAGE_WEIGHT = 0.6
AUDIO_WEIGHT = 0.4
# PLAYER 玩家的反应程度
FACE_WEIGHT = 0.4
VOICE_WEIGHT = 0.3
INTERACTION_WEIGHT = 0.3

# === History Buffers ===
history_time = []
history_game_fused_score = []
history_player_fused_score = []
history_image_score = []
history_audio_score = []
history_voice_score = []
history_expression_score = []
history_interaction_score = []


def save_plot_auto_named(
    time_series,
    image_scores,
    expression_scores,
    voice_scores,
    interaction_scores,
    game_scores,
    player_scores,
    save_dir="plots"
):
    # Create directory if not exists
    os.makedirs(save_dir, exist_ok=True)

    # Timestamp-based filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_path = os.path.join(save_dir, f"violence_plot_{timestamp}.png")

    # Plotting
    plt.figure(figsize=(14, 7))

    plt.plot(time_series, image_scores, label="Image Score (CLIP)", linestyle="--")
    plt.plot(time_series, expression_scores, label="Face Score (MediaPipe)", linestyle="--")
    plt.plot(time_series, voice_scores, label="Voice Score (Sound)", linestyle="--")
    plt.plot(time_series, interaction_scores, label="Mouse Score", linestyle="--")

    plt.plot(time_series, game_scores, label="Game Violence Level", linewidth=2)
    plt.plot(time_series, player_scores, label="Player Reaction Level", linewidth=2)

    plt.axhline(3.0, color='r', linestyle=':', label="Warning Threshold (3.0)")

    plt.xlabel("Time (s)")
    plt.ylabel("Score (0–4)")
    plt.title("Violence Scene vs Player Reaction Over Time")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()

    # Save & show
    plt.savefig(save_path)
    print(f"✅ Plot saved to: {save_path}")
    plt.show()


# === Main Loop ===
print("🎮 Starting real-time violence estimation (4-modality)...")

t0 = time.time()

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

try:
    while True:
        start_time = time.time()

        # 1.1 Image
        img = capture_screen()
        image_score = query_clip_image_score(img)

        # 1.2 Audio
        audio_segment = record_audio_segment(duration=0.5)
        audio_score = query_clap(audio_segment)

        # [Game] Fusion
        game_fused_score = (
            IMAGE_WEIGHT * image_score +
            AUDIO_WEIGHT * audio_score
        )

        # 2.1 Read webcam image
        ret, frame = cap.read()
        expression_score = analyze_face_expression(frame) if ret else 0

        # 2.2 Voice
        energy = capture_audio_energy()
        voice_score = map_audio_to_emotion_score(energy)

        # 2.3 Interaction
        fire_cnt, fire_dur = reset_mouse_stats()
        interaction_score = compute_interaction_score(fire_cnt, fire_dur)

        # [PLAYER] Fusion
        player_fused_score = (
            FACE_WEIGHT * expression_score +
            VOICE_WEIGHT * voice_score +
            INTERACTION_WEIGHT * interaction_score
        )

        # 7. Log & Print
        now = time.time() - t0
        history_time.append(now)
        history_image_score.append(image_score)
        history_expression_score.append(expression_score)
        history_voice_score.append(voice_score)
        history_interaction_score.append(interaction_score)
        history_game_fused_score.append(game_fused_score)
        history_player_fused_score.append(player_fused_score)

        print(f"[{now:.1f}s] GAME VIOLENCE LEVEL: {game_fused_score:.2f} <--- | Image(0.6): {image_score:.2f} | AUDIO(0.4): {audio_score:.2f}")
        print(f"[{now:.1f}s] PLAYER REACTION LEVEL: {player_fused_score:.2f} <--- | FACE(0.4): {expression_score:.2f} | VOICE(0.3): {voice_score:.2f} | MOUSE(0.3): {interaction_score:.2f}")

        reaction_gap = player_fused_score - game_fused_score

        if abs(reaction_gap) < 1.5:
            status = "✅ 正常反应"
        elif reaction_gap > 1.5 and game_fused_score > 2.0:
            status = "⚠️ 过度兴奋，建议警告"
            # ser_output.write(b"T\n")
        elif reaction_gap < -1.5 and game_fused_score > 2.0:
            status = "⚠️ 情绪缺失，建议警告"
            # ser_output.write(b"T\n")
        else:
            status = "✅ 无需警告（低暴力场景）"

        print(f"[{now:.1f}s] GAP: {reaction_gap:.2f} → {status}")

        elapsed = time.time() - start_time
        time.sleep(max(0.5 - elapsed, 0.01))

        cv2.imshow("Webcam - Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Program stopped.")
    cap.release()
    cv2.destroyAllWindows()

save_plot_auto_named(
    time_series=history_time,
    image_scores=history_image_score,
    expression_scores=history_expression_score,
    voice_scores=history_voice_score,
    interaction_scores=history_interaction_score,
    game_scores=history_game_fused_score,
    player_scores=history_player_fused_score
)

