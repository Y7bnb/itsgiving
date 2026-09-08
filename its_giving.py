#!/usr/bin/env python3
"""
its_giving.py — meme reactions on top of your face, live in Zoom / Meet.

MediaPipe (Tasks API) tracks your face and hands; when a pose matches, the matching
image/GIF is pasted over your face and the frame goes out through a virtual camera.

Poses (checked in this order, first match wins):
  time_out         T sign: one hand flat on top, other hand vertical underneath it
  heart            heart hands: index fingertips together, thumb tips together
  cover_nose       both hands over your nose/mouth
  crashing_out     elbows raised (hands on your head) + mouth open
  dance            elbows raised (hands behind your head) + mouth closed
  nose_closed      one hand pinching your nose
  flirty           one index fingertip touching your mouth
  hand_up          one open palm raised beside your head
  tongue_out       mouth open + tongue out (pink fills the mouth opening)
  open_mouth       mouth wide open
  disgusted        scrunch your nose (or brows down + frown)
  talking_to_wall  hands visible and gesturing (moving around)
  suspicious       head turned to the side + squint
  spin             face, hands and body gone from the frame

Assets go in ./assets as <pose>.gif / .png / .jpg / .jpeg  (e.g. assets/heart.jpeg).
A leading "something_" prefix on the filename is ignored, so 123_heart.jpeg works too.

Keys (preview window focused):  q quit   d toggle HUD   1-9 0 - = [ ] force-show a pose (test)
Run:  python its_giving.py [--camera 1] [--no-vcam] [--size 640x480] [--no-flip]
"""
import argparse
import os
import platform
import subprocess
import sys
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

# ------------------------------------------------------------------ tuning
POSES = ["time_out", "heart", "cover_nose", "crashing_out", "dance", "nose_closed", "flirty", "hand_up",
         "tongue_out", "open_mouth", "disgusted", "talking_to_wall", "suspicious", "spin"]
TEST_KEYS = "1234567890-=[]"        # key i forces POSES[i] for 2 s

FACE_SCALE = 2.0      # overlay height = this × your face height
HOLD_FRAMES = 10      # keep showing after the pose ends (anti-flicker)
ARM = {               # frames a pose must persist before it fires (default 3 ≈ 0.1 s)
    "spin": 15, "suspicious": 8, "talking_to_wall": 6, "dance": 6, "crashing_out": 4,
    "open_mouth": 4, "tongue_out": 5, "disgusted": 5,
}
T = dict(             # thresholds — press 'd' to watch the live values and tune
    jaw_open=0.5,     # blendshape 0..1: "open_mouth"
    scream_jaw=0.3,   # mouth open at least this much with elbows up = crashing_out, below = dance
    tongue_jaw=0.3,   # mouth must be at least this open before we even look for a tongue
    tongue=0.5,       # fraction of the mouth opening that is pink (0..1)
    sneer=0.12,       # noseSneer alone; a nose scrunch fires on this by itself (neutral face reads ~0.02)
    disgust=0.6,      # ...or 2*noseSneer + browDown + mouthFrown + upperLipRaise for the full face
    head_turn=0.15,   # 0 = facing camera, ~0.4 = full profile
    squint=0.3,       # eyeSquint / half-closed eyes
    gesture=0.035,    # hand speed, in face-widths per frame
)
INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}
HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ models
def ensure_models():
    mdir = os.path.join(HERE, "models")
    os.makedirs(mdir, exist_ok=True)
    paths = {}
    for name, url in MODELS.items():
        path = os.path.join(mdir, name)
        if not os.path.exists(path):
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(url, path)
        paths[name] = path
    return paths


def preflight(model_path):
    """Open a detector in a throwaway subprocess first: bad macOS MediaPipe builds
    (0.10.30+, Metal compiled in but no GPU service) abort() on this, and abort()
    cannot be caught in-process."""
    code = (
        "import sys\n"
        "from mediapipe.tasks import python as t\n"
        "from mediapipe.tasks.python import vision\n"
        "vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(\n"
        "    base_options=t.BaseOptions(model_asset_path=sys.argv[1]),\n"
        "    running_mode=vision.RunningMode.VIDEO, num_faces=1,\n"
        "    output_face_blendshapes=True)).close()\n"
    )
    proc = subprocess.run([sys.executable, "-c", code, model_path], capture_output=True, text=True)
    if proc.returncode == 0:
        return
    err = (proc.stderr or "") + (proc.stdout or "")
    print(f"\nMediaPipe cannot start a detector here (python {platform.python_version()}, "
          f"mediapipe {getattr(mp, '__version__', '?')}, exit {proc.returncode}).\n")
    if "Service is unavailable" in err or "MetalHelper" in err or proc.returncode == -6:
        print("Cause: mediapipe 0.10.30+ macOS wheels abort on startup. Use Python 3.11/3.12 with:\n"
              '  pip install "mediapipe==0.10.21" --no-deps\n'
              '  pip install "numpy<2" "protobuf>=4.25.3,<5" absl-py attrs flatbuffers sounddevice \\\n'
              '              sentencepiece matplotlib "opencv-contrib-python<5" "opencv-python<5" pyvirtualcam pillow\n')
    else:
        print(err[-1500:])
    sys.exit(1)


def build_detectors(model_paths):
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["face_landmarker.task"]),
        running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True))
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["hand_landmarker.task"]),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["pose_landmarker_lite.task"]),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    return face, hand, pose


# ------------------------------------------------------------------ assets
class Asset:
    """One reaction: a list of BGRA frames plus per-frame durations (ms) for GIFs."""

    def __init__(self, frames, durations):
        self.frames = frames
        self.durations = durations
        self.cum = np.cumsum(durations)
        self.total = int(self.cum[-1])
        h, w = frames[0].shape[:2]
        self.aspect = w / h
        self._cache = {}

    def frame_at(self, ms):
        if len(self.frames) == 1:
            return 0
        return int(np.searchsorted(self.cum, ms % self.total, side="right"))

    def scaled(self, idx, height):
        key = (idx, height)
        if key not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            w = max(1, int(round(height * self.aspect)))
            self._cache[key] = cv2.resize(self.frames[idx], (w, height), interpolation=cv2.INTER_AREA)
        return self._cache[key]


def to_bgra(img):
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    return img


def placeholder(label):
    img = np.zeros((300, 300, 4), np.uint8)
    cv2.circle(img, (150, 150), 140, (0, 0, 255, 220), -1)
    cv2.putText(img, label, (12, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255, 255), 2)
    return Asset([img], [100])


def find_asset_file(pose):
    adir = os.path.join(HERE, "assets")
    if not os.path.isdir(adir):
        return None
    exts = (".gif", ".png", ".jpg", ".jpeg")
    for fn in sorted(os.listdir(adir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in exts and (stem == pose or stem.endswith("_" + pose)):
            return os.path.join(adir, fn)
    return None


def load_asset(pose):
    path = find_asset_file(pose)
    if path is None:
        print(f"  {pose:16s} missing -> placeholder")
        return placeholder(pose)
    frames, durations = [], []
    if path.lower().endswith(".gif"):
        from PIL import Image, ImageSequence
        with Image.open(path) as im:
            for f in ImageSequence.Iterator(im):
                frames.append(cv2.cvtColor(np.array(f.convert("RGBA")), cv2.COLOR_RGBA2BGRA))
                durations.append(max(20, int(f.info.get("duration", 100))))
    else:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None:
            frames, durations = [to_bgra(img)], [100]
    if not frames:
        print(f"  {pose:16s} could not read {os.path.basename(path)} -> placeholder")
        return placeholder(pose)
    print(f"  {pose:16s} {os.path.basename(path)}  ({len(frames)} frame{'s' if len(frames) > 1 else ''})")
    return Asset(frames, durations)


def overlay(frame, sprite, x, y):
    """Alpha-composite BGRA sprite onto BGR frame at top-left (x, y), clipped to the frame."""
    H, W = frame.shape[:2]
    h, w = sprite.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x0 >= x1 or y0 >= y1:
        return frame
    s = sprite[y0 - y:y1 - y, x0 - x:x1 - x]
    a = s[:, :, 3:4].astype(np.float32) / 255.0
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = (a * s[:, :, :3] + (1 - a) * roi).astype(np.uint8)
    return frame


# ------------------------------------------------------------------ geometry
def dist(a, b):
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


class Face:
    def __init__(self, lms, blendshapes, W, H):
        p = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
        self.pts = p
        x0, y0 = p.min(0)
        x1, y1 = p.max(0)
        self.box = (int(x0), int(y0), int(x1), int(y1))
        self.w, self.h = float(x1 - x0), float(y1 - y0)
        self.center = ((x0 + x1) / 2, (y0 + y1) / 2)
        self.nose, self.chin, self.top = p[1], p[152], p[10]
        self.mouth = (p[13] + p[14]) / 2
        self.eye_y = float((p[33][1] + p[263][1]) / 2)
        cl, cr = p[234], p[454]                       # left / right edge of the face
        self.turn = abs((self.nose[0] - cl[0]) / max(cr[0] - cl[0], 1e-3) - 0.5)
        self.bs = {c.category_name: c.score for c in (blendshapes or [])}

    def b(self, name):
        return self.bs.get(name, 0.0)


class Hand:
    def __init__(self, lms, W, H):
        p = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
        self.palm = p[[0, 5, 9, 13, 17]].mean(0)
        self.thumb, self.index, self.middle = p[4], p[8], p[12]
        d = p[9] - p[0]                               # wrist -> middle knuckle
        self.horizontal = abs(d[0]) > 1.5 * abs(d[1])
        self.vertical = abs(d[1]) > 1.5 * abs(d[0])
        ext = [dist(p[0], p[t]) > 1.2 * dist(p[0], p[t - 2]) for t in (8, 12, 16, 20)]
        self.open = sum(ext) >= 3


class Body:
    """Upper-body pose: shoulders 11/12, elbows 13/14, wrists 15/16."""

    def __init__(self, lms, W, H):
        p = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
        self.shoulders, self.elbows, self.wrists = p[[11, 12]], p[[13, 14]], p[[15, 16]]
        vis = [getattr(lms[i], "visibility", 1.0) for i in (11, 12, 13, 14)]
        self.seen = min(vis) > 0.5
        shoulder_y = float(self.shoulders[:, 1].mean())
        self.elbows_up = self.seen and bool((self.elbows[:, 1] < shoulder_y).all())



def tongue_score(frame, face, hands):
    """Fraction of the mouth opening that is tongue-colored (pink/red, saturated, lit).
    Teeth are unsaturated, the cavity is dark, and the lips sit outside the polygon.
    Only evaluated when the jaw is clearly open and no hand is near the mouth, since
    lips and skin are pink too."""
    if face.b("jawOpen") < T["tongue_jaw"]:
        return 0.0
    if any(dist(h.palm, face.mouth) < 0.7 * face.w for h in hands):
        return 0.0
    poly = face.pts[INNER_LIPS].astype(np.int32)
    x0, y0 = poly.min(0)
    x1, y1 = poly.max(0)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0.0
    x0, y0 = max(x0, 0), max(y0, 0)
    roi = frame[y0:y1 + 1, x0:x1 + 1]
    if roi.size == 0:
        return 0.0
    mask = np.zeros(roi.shape[:2], np.uint8)
    cv2.fillPoly(mask, [poly - [x0, y0]], 255)
    k = max(3, int(0.15 * (y1 - y0)))                  # shave 15% off the edges: no lip pixels
    mask = cv2.erode(mask, np.ones((k, k), np.uint8))
    n = int(np.count_nonzero(mask))
    if n < 40:
        return 0.0
    h, s, v = cv2.split(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV))
    pink = ((h < 12) | (h > 160)) & (s > 70) & (v > 110)
    return float(np.count_nonzero(pink & (mask > 0)) / n)


class Motion:
    """Smoothed hand speed across frames, in face-widths per frame."""

    def __init__(self):
        self.prev, self.energy, self.fw = [], 0.0, 200.0

    def update(self, hands, face):
        if face is not None:
            self.fw = max(face.w, 1.0)
        cur = [h.palm for h in hands]
        speed = 0.0
        if cur and self.prev:
            moved = [min(dist(c, p) for p in self.prev) for c in cur]
            moved = [m for m in moved if m < self.fw]         # ignore re-detection jumps
            if moved:
                speed = max(moved) / self.fw
        self.energy = 0.8 * self.energy + 0.2 * speed
        self.prev = cur
        return self.energy


def decide(face, hands, body, tongue, gesture):
    """Return (pose or None, debug dict)."""
    d = {"hands": len(hands)}
    if face is None:
        gone = not hands and (body is None or not body.seen)
        return ("spin" if gone else None), d          # face hidden but you're still there: keep holding

    fw = face.w
    near = lambda a, b, k: dist(a, b) < k * fw        # distances in units of face width
    jaw = face.b("jawOpen")
    elbows_up = bool(body and body.elbows_up)
    pair = lambda n: (face.b(n + "Left") + face.b(n + "Right")) / 2
    sneer, brow_down, frown, lip_up = pair("noseSneer"), pair("browDown"), pair("mouthFrown"), pair("mouthUpperUp")
    disgust = 2 * sneer + brow_down + frown + lip_up      # each alone is weak; a real scrunch moves all four
    squint = max((face.b("eyeSquintLeft") + face.b("eyeSquintRight")) / 2,
                 (face.b("eyeBlinkLeft") + face.b("eyeBlinkRight")) / 2)
    d.update(jaw=jaw, tongue=tongue, disgust=disgust, sneer=sneer, brow=brow_down, frown=frown, lip=lip_up,
             turn=face.turn, squint=squint, gesture=gesture, elbows_up=elbows_up)

    if len(hands) >= 2:
        a, b = hands[0], hands[1]
        for top, under in ((a, b), (b, a)):           # time-out T
            if top.horizontal and under.vertical and top.palm[1] < under.palm[1] \
                    and near(under.middle, top.palm, 0.6):
                return "time_out", d
        if near(a.index, b.index, 0.3) and near(a.thumb, b.thumb, 0.3) \
                and (a.index[1] + b.index[1]) < (a.thumb[1] + b.thumb[1]):
            return "heart", d
        if near(a.palm, face.mouth, 0.6) and near(b.palm, face.mouth, 0.6):
            return "cover_nose", d
        on_head = lambda h: (h.palm[1] < face.eye_y and abs(h.palm[0] - face.nose[0]) < 1.1 * fw
                             and h.palm[1] > face.top[1] - 0.8 * face.h)
        if on_head(a) and on_head(b) and jaw > T["scream_jaw"]:
            return "crashing_out", d

    # hands on / behind the head: the hand model often can't see them, but the pose
    # model sees the raised elbows. Mouth open = crashing out, mouth closed = dance.
    near_head = lambda h: abs(h.palm[0] - face.nose[0]) < 1.3 * fw and h.palm[1] < face.eye_y + 0.3 * face.h
    if elbows_up and all(near_head(h) for h in hands):
        return ("crashing_out" if jaw > T["scream_jaw"] else "dance"), d

    for h in hands:
        if near(h.thumb, face.nose, 0.35) and near(h.index, face.nose, 0.35) and near(h.thumb, h.index, 0.3):
            return "nose_closed", d
        if near(h.index, face.mouth, 0.22) and not near(h.palm, face.mouth, 0.3):
            return "flirty", d                        # fingertip on the lips, palm not covering the mouth
        if h.open and h.palm[1] < face.nose[1] and abs(h.palm[0] - face.nose[0]) > 0.8 * fw:
            return "hand_up", d

    if tongue > T["tongue"]:
        return "tongue_out", d
    if jaw > T["jaw_open"]:
        return "open_mouth", d
    if sneer > T["sneer"] or disgust > T["disgust"]:
        return "disgusted", d
    if hands and gesture > T["gesture"]:
        return "talking_to_wall", d
    if face.turn > T["head_turn"] and squint > T["squint"]:
        return "suspicious", d
    return None, d


def draw_hud(img, shown, raw, d, face, hands, body):
    if face:
        x0, y0, x1, y1 = face.box
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 1)
    for h in hands:
        cv2.circle(img, (int(h.palm[0]), int(h.palm[1])), 6, (0, 200, 255), -1)
    if body and body.seen:
        for pt in np.vstack([body.shoulders, body.elbows]):
            cv2.circle(img, (int(pt[0]), int(pt[1])), 6, (255, 120, 0), -1)
    lines = [
        f"showing: {shown or '-'}   raw: {raw or '-'}   hands: {d.get('hands', 0)}   elbows up: {'Y' if d.get('elbows_up') else 'n'}",
        f"jaw {d.get('jaw', 0):.2f}  tongue {d.get('tongue', 0):.2f}  turn {d.get('turn', 0):.2f}  squint {d.get('squint', 0):.2f}  gesture {d.get('gesture', 0):.3f}",
        f"disgust {d.get('disgust', 0):.2f} = 2x sneer {d.get('sneer', 0):.2f} + brow {d.get('brow', 0):.2f} + frown {d.get('frown', 0):.2f} + lip {d.get('lip', 0):.2f}",
        "keys: q quit  d hud  1-9 0 - = [ ] test poses",
    ]
    for i, t in enumerate(lines):
        y = 24 + 22 * i
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="webcam index (try 1 if 0 is your iPhone)")
    ap.add_argument("--no-vcam", action="store_true", help="preview only; don't start the virtual camera")
    ap.add_argument("--skip-check", action="store_true", help="skip the MediaPipe startup check")
    ap.add_argument("--size", default="1280x720", help="capture size, e.g. 1280x720 or 640x480 (lower = faster)")
    ap.add_argument("--no-flip", action="store_true", help="don't mirror the image")
    args = ap.parse_args()

    model_paths = ensure_models()
    if not args.skip_check:
        preflight(model_paths["face_landmarker.task"])
    print("Assets:")
    assets = {pose: load_asset(pose) for pose in POSES}

    cap = cv2.VideoCapture(args.camera)
    if cap.isOpened() and "x" in args.size:
        w, h = args.size.lower().split("x")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
    ok, frame = False, None
    if cap.isOpened():
        for _ in range(5):                            # let the size change settle
            ok, frame = cap.read()
            if not ok:
                break
    if not ok:
        sys.exit(f"Could not read from camera {args.camera}.\n"
                 "  - try --camera 1\n"
                 "  - System Settings > Privacy & Security > Camera: allow your terminal app, then re-run")
    H, W = frame.shape[:2]
    print(f"Camera {args.camera}: {W}x{H}")

    vcam = None
    if not args.no_vcam:
        try:
            import pyvirtualcam
            vcam = pyvirtualcam.Camera(width=W, height=H, fps=30, fmt=pyvirtualcam.PixelFormat.BGR)
            print(f"Virtual camera: '{vcam.device}'  <- pick this camera in Zoom / Meet")
        except Exception as e:
            print(f"Virtual camera unavailable ({e}). Preview-only.")

    face_det, hand_det, pose_det = build_detectors(model_paths)
    motion = Motion()
    shown, hold, show_hud = None, 0, True
    arm = {p: 0 for p in POSES}
    shown_since = 0.0
    forced, forced_until = None, 0.0
    sm_center, sm_h = np.array([W / 2, H / 2], np.float32), H * 0.45   # smoothed face box
    t_start, last_ts = time.monotonic(), -1
    print("Running. Focus the preview window: q quit, d HUD, 1-9 0 - = [ ] test a pose")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera stopped returning frames.")
                break
            if frame.shape[0] != H or frame.shape[1] != W:
                frame = cv2.resize(frame, (W, H))
            if not args.no_flip:
                frame = cv2.flip(frame, 1)

            ts = int((time.monotonic() - t_start) * 1000)
            ts = last_ts = max(ts, last_ts + 1)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fr = face_det.detect_for_video(mp_img, ts)
            hr = hand_det.detect_for_video(mp_img, ts)
            pr = pose_det.detect_for_video(mp_img, ts)
            face = Face(fr.face_landmarks[0], fr.face_blendshapes[0] if fr.face_blendshapes else None, W, H) \
                if fr.face_landmarks else None
            hands = [Hand(h, W, H) for h in hr.hand_landmarks]
            body = Body(pr.pose_landmarks[0], W, H) if pr.pose_landmarks else None

            tongue = tongue_score(frame, face, hands) if face is not None else 0.0
            gesture = motion.update(hands, face)
            raw, dbg = decide(face, hands, body, tongue, gesture)

            # arm (must persist) then hold (linger after it ends)
            fired = None
            for p in POSES:
                arm[p] = arm[p] + 1 if raw == p else 0
                if raw == p and arm[p] >= ARM.get(p, 3):
                    fired = p
            now = time.monotonic()
            if forced and now < forced_until:
                fired = forced
            if fired:
                if fired != shown:
                    shown_since = now
                shown, hold = fired, HOLD_FRAMES
            elif hold > 0:
                hold -= 1
            else:
                shown = None

            # follow the face (smoothed); when the face is gone, stay where it was
            if face is not None:
                sm_center = 0.7 * sm_center + 0.3 * np.array(face.center, np.float32)
                sm_h = 0.7 * sm_h + 0.3 * face.h * FACE_SCALE

            if shown:
                asset = assets[shown]
                idx = asset.frame_at(int((now - shown_since) * 1000))
                h = int(min(sm_h, H * 0.98, (W * 0.98) / asset.aspect)) // 8 * 8
                sprite = asset.scaled(idx, max(h, 8))
                sh, sw = sprite.shape[:2]
                overlay(frame, sprite, int(sm_center[0] - sw / 2), int(sm_center[1] - sh / 2 - 0.05 * sh))

            if vcam:
                vcam.send(frame)
                vcam.sleep_until_next_frame()

            preview = frame
            if show_hud:
                preview = frame.copy()
                draw_hud(preview, shown, raw, dbg, face, hands, body)
            cv2.imshow("Reaction Cam  (q quit, d HUD, 1-9 0 - = [ ] test)", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_hud = not show_hud
            elif 0 < key < 256 and chr(key) in TEST_KEYS:
                forced, forced_until = POSES[TEST_KEYS.index(chr(key))], now + 2.0
    finally:
        cap.release()
        face_det.close()
        hand_det.close()
        pose_det.close()
        if vcam:
            vcam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
