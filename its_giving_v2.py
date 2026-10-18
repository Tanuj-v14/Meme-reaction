#!/usr/bin/env python3
"""
its_giving_v2.py — meme reactions on top of your face, live in Zoom / Meet.

Same as its_giving.py, but expression thresholds are measured in standard
deviations above your own resting face rather than against fixed constants.
Seven seconds of calibration; see README.md.

  python its_giving_v2.py --calibrate
  python its_giving_v2.py [--camera 1] [--no-vcam] [--size 640x480] [--no-flip]

Keys:  q quit   d toggle HUD   c recalibrate   1-9 0 - = [ ] \\ e r t y u i o p a test poses
"""
import argparse
import json
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

POSES = [
    "time_out", "heart", "cover_nose", "crashing_out", "nose_closed", "flirty",
    "open_mouth", "disgusted", "talking_to_wall", "suspicious", "spin", "son",
    "mind", "actually", "shit", "me", "intellectual", "cinema", "soldier", "idea", "mew"
]
TEST_KEYS = "1234567890-=[]\\ertyui"

FACE_SCALE = 2.0
HOLD_FRAMES = 10
ARM = {
    "spin": 15, "suspicious": 8, "talking_to_wall": 12, "crashing_out": 4,
    "open_mouth": 4, "disgusted": 5, "son": 3, "cover_nose": 2,
    "shit": 3, "cinema": 3, "mind": 3, "actually": 3, "idea": 3, "intellectual": 3,
    "me": 2, "mew": 3, "soldier": 1,
}

Z = dict(
    jaw_open=6.0,
    scream_jaw=3.5,
    sneer=4.5,
    disgust=14.0,
    squint=4.0,
    smile=2.5,
)
Z_CAP = 8.0
FLOOR = dict(
    jaw_open=0.30,
    scream_jaw=0.18,
    sneer=0.06,
    squint=0.18,
    smile=0.25,
)
T = dict(
    head_turn=0.15,
    gesture=0.075,
)

CALIB_FILE = "calibration.json"
CALIB_SECONDS = 7.0
CALIB_WARMUP = 1.5
CALIB_MIN_SAMPLES = 30
SIGMA_FLOOR = 0.015
SIGMA_CEIL = 0.080
CALIB_VERSION = 1

GENERIC_SIGMA = 0.035
GENERIC_MEAN = {
    "jawOpen": 0.08, "eyeSquintLeft": 0.10, "eyeSquintRight": 0.10,
    "eyeBlinkLeft": 0.10, "eyeBlinkRight": 0.10, "noseSneerLeft": 0.03, "noseSneerRight": 0.03,
    "browDownLeft": 0.06, "browDownRight": 0.06, "mouthFrownLeft": 0.05, "mouthFrownRight": 0.05,
    "mouthUpperUpLeft": 0.05, "mouthUpperUpRight": 0.05,
    "mouthSmileLeft": 0.05, "mouthSmileRight": 0.05,
}

INNER_LIPS = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

MODELS = {
    "face_landmarker.task": "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
}
HERE = os.path.dirname(os.path.abspath(__file__))


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
    """Open a detector in a throwaway subprocess: bad macOS builds abort() uncatchably."""
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
        print("Cause: mediapipe 0.10.30+ ships macOS wheels that abort on startup.\n"
              "Fix (Python 3.11 or 3.12) — install the pinned set:\n"
              "  pip install -r requirements.txt\n"
              "If you already installed something newer by hand, force it back:\n"
              '  pip install "mediapipe==0.10.21" "numpy<2" "opencv-python<5" "opencv-contrib-python<5"\n')
    else:
        print(err[-1500:])
    sys.exit(1)


def build_detectors(model_paths):
    face = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["face_landmarker.task"]),
        running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True))
    hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["hand_landmarker.task"]),
        running_mode=vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3))
    pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["pose_landmarker_lite.task"]),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    return face, hand, pose


class Clock:
    """Strictly increasing timestamps for the life of a detector, recalibrations included."""

    def __init__(self):
        self.t0, self.last = time.monotonic(), -1

    def next(self):
        self.last = max(int((time.monotonic() - self.t0) * 1000), self.last + 1)
        return self.last


class Baseline:
    """Your resting face: a mean and a wobble for every channel."""

    def __init__(self, mean=None, sigma=None, samples=0, made=None):
        self.mean = mean or {}
        self.sigma = sigma or {}
        self.samples = samples
        self.made = made
        self.generic = not self.mean

    def z(self, name, value):
        """How far above your neutral this channel is, in standard deviations."""
        if self.generic:
            return (value - GENERIC_MEAN.get(name, 0.02)) / GENERIC_SIGMA
        m = self.mean.get(name)
        if m is None:
            return (value - GENERIC_MEAN.get(name, 0.02)) / GENERIC_SIGMA
        return (value - m) / self.sigma.get(name, SIGMA_CEIL)

    @property
    def neutral_turn(self):
        return self.mean.get("turn_signed", 0.0) if not self.generic else 0.0

    def save(self, path):
        with open(path, "w") as fh:
            json.dump({"version": CALIB_VERSION, "made": self.made, "samples": self.samples,
                       "mean": self.mean, "sigma": self.sigma}, fh, indent=1, sort_keys=True)

    @staticmethod
    def load(path):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            return Baseline()
        if data.get("version") != CALIB_VERSION or not data.get("mean"):
            return Baseline()
        return Baseline(data["mean"], data.get("sigma", {}), data.get("samples", 0), data.get("made"))


class Collector:
    """Running mean and standard deviation per channel, over the calibration window."""

    def __init__(self):
        self.n, self.s, self.ss = 0, {}, {}

    def add(self, face):
        self.n += 1
        for name, v in list(face.bs.items()) + [("turn_signed", face.turn_signed)]:
            self.s[name] = self.s.get(name, 0.0) + v
            self.ss[name] = self.ss.get(name, 0.0) + v * v

    def finish(self):
        mean, sigma = {}, {}
        for name, total in self.s.items():
            m = total / self.n
            var = max(self.ss[name] / self.n - m * m, 0.0)
            mean[name] = round(m, 5)
            sigma[name] = round(min(max(var ** 0.5, SIGMA_FLOOR), SIGMA_CEIL), 5)
        sigma["turn_signed"] = min(max(sigma.get("turn_signed", 0.02), 0.01), 0.10)
        return Baseline(mean, sigma, self.n, time.strftime("%Y-%m-%d %H:%M"))


def calibration_warnings(base):
    """Catch the two ways a calibration goes wrong: mid-expression, or fidgeting."""
    out = []
    if base.mean.get("jawOpen", 0) > 0.30:
        out.append("your mouth looks like it was open — don't talk during calibration")
    if max(base.mean.get("noseSneerLeft", 0), base.mean.get("noseSneerRight", 0)) > 0.15:
        out.append("your nose was scrunched — hold a bored face, not a reaction")
    if max(base.mean.get("browInnerUp", 0), base.mean.get("browOuterUpLeft", 0)) > 0.35:
        out.append("your eyebrows were up — relax them")
    pinned = sum(1 for k, v in base.sigma.items() if v >= SIGMA_CEIL)
    if pinned > 12:
        out.append("you moved a lot — sit still and try again for a tighter baseline")
    return out


def run_calibration(cap, face_det, clock, args, W, H, window):
    """Watch a bored face for CALIB_SECONDS and learn what its channels rest at."""
    print(f"\nCalibrating for {CALIB_SECONDS:.0f}s. Sit how you normally sit, look at the "
          "camera, hold a bored face.\nBlinking is fine. Don't talk, smile or raise your eyebrows.")
    col, start, seen_face = Collector(), time.monotonic(), 0
    while True:
        elapsed = time.monotonic() - start
        if elapsed > CALIB_SECONDS:
            break
        ok, frame = cap.read()
        if not ok:
            break
        if frame.shape[0] != H or frame.shape[1] != W:
            frame = cv2.resize(frame, (W, H))
        if not args.no_flip:
            frame = cv2.flip(frame, 1)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        fr = face_det.detect_for_video(mp_img, clock.next())
        face = Face(fr.face_landmarks[0], fr.face_blendshapes[0] if fr.face_blendshapes else None, W, H) \
            if fr.face_landmarks else None
        if face is not None:
            seen_face += 1
            if elapsed > CALIB_WARMUP and face.bs:
                col.add(face)
        draw_calibration(frame, elapsed, col.n, face)
        cv2.imshow(window, frame)
        if (cv2.waitKey(1) & 0xFF) == ord("q"):
            print("Calibration cancelled.")
            return None

    if col.n < CALIB_MIN_SAMPLES:
        print(f"Calibration failed: only {col.n} usable frames"
              f"{' — your face was never detected' if not seen_face else ''}.\n"
              "  - light your face from the front, sit head-and-shoulders in frame, and try again")
        return None
    base = col.finish()
    print(f"Calibrated on {base.samples} frames. Your neutral face:")
    for name in ("jawOpen", "noseSneerLeft", "browDownLeft", "mouthFrownLeft", "eyeSquintLeft"):
        if name in base.mean:
            print(f"  {name:16s} {base.mean[name]:.3f} ± {base.sigma[name]:.3f}")
    for w in calibration_warnings(base):
        print(f"  ! {w}")
    return base


def draw_calibration(img, elapsed, samples, face):
    H, W = img.shape[:2]
    left = max(0.0, CALIB_SECONDS - elapsed)
    cv2.rectangle(img, (0, 0), (W, 96), (0, 0, 0), -1)
    cv2.putText(img, "CALIBRATING - hold a bored face", (16, 34), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (255, 255, 255), 2)
    cv2.putText(img, f"{left:0.1f}s   {samples} frames" + ("" if face is not None else "   NO FACE"),
                (16, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255) if face is not None else (0, 140, 255), 2)
    done = int(W * min(elapsed / CALIB_SECONDS, 1.0))
    cv2.rectangle(img, (0, 84), (done, 96), (0, 220, 0), -1)
    if face is not None:
        x0, y0, x1, y1 = face.box
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 220, 0), 1)


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
    aliases = {
        "soldier": ["soldier", "cat meme", "cat_meme"]
    }
    targets = aliases.get(pose, [pose])
    for fn in sorted(os.listdir(adir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() in exts:
            for t in targets:
                if stem == t or stem.endswith("_" + t):
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
        cl, cr = p[234], p[454]
        self.turn_signed = float((self.nose[0] - cl[0]) / max(cr[0] - cl[0], 1e-3) - 0.5)
        self.bs = {c.category_name: c.score for c in (blendshapes or [])}
        self.temple_r = p[[54, 103, 162]].mean(0)
        self.temple_l = p[[284, 332, 389]].mean(0)
        self.jaw_pts_r = p[[234, 132, 58, 172, 136, 150, 149, 176, 148, 152]]
        self.jaw_pts_l = p[[152, 377, 400, 378, 379, 365, 397, 288, 361, 454]]

    def b(self, name):
        return self.bs.get(name, 0.0)


class Hand:
    def __init__(self, lms, W, H):
        p = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
        self.p = p
        self.wrist = p[0]
        self.palm = p[[0, 5, 9, 13, 17]].mean(0)
        self.thumb, self.index, self.middle = p[4], p[8], p[12]
        self.ring, self.pinky = p[16], p[20]
        self.index_mcp, self.index_pip = p[5], p[6]
        d = p[9] - p[0]
        self.horizontal = abs(d[0]) > 1.5 * abs(d[1])
        self.vertical = abs(d[1]) > 1.5 * abs(d[0])
        ext = [dist(p[0], p[t]) > 1.2 * dist(p[0], p[t - 2]) for t in (8, 12, 16, 20)]
        self.ext = ext
        self.index_ext = ext[0] or (dist(p[0], p[8]) > 1.15 * dist(p[0], p[12]))
        self.middle_ext = ext[1]
        self.ring_ext = ext[2]
        self.pinky_ext = ext[3]
        self.num_ext = sum(ext)
        self.open = self.num_ext >= 3
        self.fist = self.num_ext <= 1
        v_idx = p[8] - p[5]
        v_len = float(np.hypot(v_idx[0], v_idx[1]))
        self.index_up = self.index_ext and (v_idx[1] < -0.3 * max(v_len, 1.0)) and (abs(v_idx[1]) > 1.1 * abs(v_idx[0]))


class MewTracker:
    """Tracks index fingertip motion along the jawline across frames to detect MEW."""

    def __init__(self):
        self.history = []

    def reset(self):
        self.history.clear()

    def update(self, tip, jaw_pts, fw, now):
        self.history = [(t, pt) for (t, pt) in self.history if now - t <= 0.8]
        dists = [dist(tip, jp) for jp in jaw_pts]
        min_d = min(dists) if dists else 999.0
        if min_d > 0.30 * fw:
            return False

        self.history.append((now, np.array(tip, dtype=np.float32)))
        if len(self.history) < 5:
            return False

        t_span = self.history[-1][0] - self.history[0][0]
        if t_span < 0.12:
            return False

        p_first = self.history[0][1]
        p_last = self.history[-1][1]
        disp = float(np.hypot(p_last[0] - p_first[0], p_last[1] - p_first[1]))
        if 0.18 * fw < disp < 0.90 * fw:
            return True
        return False

    def has_movement(self, fw):
        if len(self.history) < 2:
            return False
        p_first = self.history[0][1]
        p_last = self.history[-1][1]
        disp = float(np.hypot(p_last[0] - p_first[0], p_last[1] - p_first[1]))
        return disp > 0.08 * fw


class InactivityTracker:
    """Tracks continuous user inactivity for the SOLDIER reaction (> 5 seconds)."""

    def __init__(self, timeout=5.0):
        self.timeout = timeout
        self.last_active = time.monotonic()
        self.prev_center = None
        self.prev_turn = None
        self.active_soldier = False

    def reset(self, now=None):
        self.last_active = now if now is not None else time.monotonic()
        self.active_soldier = False

    def update(self, face, hands, gesture, raw_intentional, now):
        if face is None:
            self.reset(now)
            return False

        if raw_intentional is not None:
            self.reset(now)
            return False

        if hands and gesture > 0.06:
            self.reset(now)
            return False

        fw = max(face.w, 1.0)
        center = np.array(face.center, np.float32)
        turn = face.turn_signed

        if self.prev_center is not None and self.prev_turn is not None:
            shift = float(np.hypot(center[0] - self.prev_center[0], center[1] - self.prev_center[1])) / fw
            turn_diff = abs(turn - self.prev_turn)
            if shift > 0.08 or turn_diff > 0.08:
                self.reset(now)
                self.prev_center = center
                self.prev_turn = turn
                return False

        self.prev_center = center
        self.prev_turn = turn

        inactive_time = now - self.last_active
        if inactive_time >= self.timeout:
            if not self.active_soldier:
                self.active_soldier = True
                return True
            return False

        return False


class Body:
    """Upper-body pose: shoulders 11/12, elbows 13/14, wrists 15/16."""

    def __init__(self, lms, W, H):
        p = np.array([[l.x * W, l.y * H] for l in lms], np.float32)
        self.shoulders, self.elbows, self.wrists = p[[11, 12]], p[[13, 14]], p[[15, 16]]
        vis = [getattr(lms[i], "visibility", 1.0) for i in (11, 12, 13, 14)]
        self.seen = min(vis) > 0.4
        wrists_vis = [getattr(lms[i], "visibility", 1.0) for i in (15, 16)]
        self.wrists_seen = min(wrists_vis) > 0.3
        shoulder_y = float(self.shoulders[:, 1].mean())
        self.elbows_up = self.seen and bool((self.elbows[:, 1] < shoulder_y).all())


def tongue_score(frame, face, hands, jaw_ready):
    """Fraction of the mouth opening that reads pink: saturated and lit, unlike teeth or throat."""
    if not jaw_ready:
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
    k = max(3, int(0.15 * (y1 - y0)))
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
            moved = [m for m in moved if m < self.fw]
            if moved:
                speed = max(moved) / self.fw
        self.energy = 0.8 * self.energy + 0.2 * speed
        self.prev = cur
        return self.energy


def measure(face, base):
    """Every expression channel, raw and in sigma above your own neutral."""
    zpair = lambda n: (base.z(n + "Left", face.b(n + "Left")) + base.z(n + "Right", face.b(n + "Right"))) / 2
    pair = lambda n: (face.b(n + "Left") + face.b(n + "Right")) / 2
    m = {
        "jaw": face.b("jawOpen"), "z_jaw": base.z("jawOpen", face.b("jawOpen")),
        "sneer": pair("noseSneer"), "z_sneer": zpair("noseSneer"),
        "z_brow": zpair("browDown"), "z_frown": zpair("mouthFrown"), "z_lip": zpair("mouthUpperUp"),
        "squint": max(pair("eyeSquint"), pair("eyeBlink")),
        "z_squint": max(zpair("eyeSquint"), zpair("eyeBlink")),
        "smile": pair("mouthSmile"), "z_smile": zpair("mouthSmile"),
        "turn": abs(face.turn_signed - base.neutral_turn),
    }
    cap = lambda v: min(v, Z_CAP)
    m["z_disgust"] = 2 * cap(m["z_sneer"]) + cap(m["z_brow"]) + cap(m["z_frown"]) + cap(m["z_lip"])
    return m


def over(key, m, zkey, rawkey):
    """Sigma above your neutral AND a raw floor, so a tiny sigma can't become a hair trigger."""
    return m[zkey] >= Z[key] and m[rawkey] >= FLOOR[key]


def decide(face, hands, body, tongue, gesture, m, mew_tracker=None, now=None, last_choice=None, smooth_smile=0.0):
    """Return (pose or None, debug dict) using centralized disambiguation."""
    d = {"hands": len(hands)}
    if face is None:
        gone = not hands and (body is None or not body.seen)
        return ("spin" if gone else None), d

    fw = face.w
    near = lambda a, b, k: dist(a, b) < k * fw
    elbows_up = bool(body and body.elbows_up)
    d.update(m, gesture=gesture, elbows_up=elbows_up)
    screaming = over("scream_jaw", m, "z_jaw", "jaw")
    turn = m.get("turn", abs(face.turn_signed))
    head_turned = turn > 0.05

    # Chest location estimation
    if body and body.seen:
        chest = body.shoulders.mean(0) + np.array([0, 0.25 * dist(body.shoulders[0], body.shoulders[1])])
    else:
        chest = np.array([face.chin[0], face.chin[1] + 0.65 * fw])

    temple_r = face.temple_r
    temple_l = face.temple_l
    jaw_pts_r = face.jaw_pts_r
    jaw_pts_l = face.jaw_pts_l

    # 1. Two-hand / Pose-assisted gestures
    if len(hands) >= 2:
        a, b = hands[0], hands[1]
        for top, under in ((a, b), (b, a)):
            if top.horizontal and under.vertical and top.palm[1] < under.palm[1] \
                    and near(under.middle, top.palm, 0.6):
                return "time_out", d
        if near(a.index, b.index, 0.3) and near(a.thumb, b.thumb, 0.3) \
                and (a.index[1] + b.index[1]) < (a.thumb[1] + b.thumb[1]):
            return "heart", d
        on_head = lambda h: (h.palm[1] < face.eye_y and abs(h.palm[0] - face.nose[0]) < 1.1 * fw
                             and h.palm[1] > face.top[1] - 0.8 * face.h)
        if on_head(a) and on_head(b) and screaming:
            return "crashing_out", d

        # SHIT vs CINEMA: Both hands raised
        hands_raised = (a.palm[1] < face.eye_y + 0.20 * fw) and (b.palm[1] < face.eye_y + 0.20 * fw)
        if hands_raised:
            hands_dist = dist(a.palm, b.palm) / fw
            lat_a = abs(a.palm[0] - face.center[0]) / fw
            lat_b = abs(b.palm[0] - face.center[0]) / fw
            dist_top_a = dist(a.palm, face.top) / fw
            dist_top_b = dist(b.palm, face.top) / fw

            # SHIT: hands on or clutching the head (monkey meme pose)
            # Hands are on top/sides of head, not held wide out in the air
            hands_on_head = (dist_top_a < 0.75 or dist_top_b < 0.75) and (hands_dist < 1.05) and (lat_a < 0.68 and lat_b < 0.68)
            if hands_on_head:
                return "shit", d

            # CINEMA: both arms raised high in the air, separated from head (Scorsese meme pose)
            hands_away_from_head = dist_top_a > 0.55 and dist_top_b > 0.55
            hands_wide = hands_dist >= 0.95 and (lat_a > 0.45 and lat_b > 0.45)
            if hands_away_from_head and hands_wide and (a.palm[1] < face.eye_y and b.palm[1] < face.eye_y):
                return "cinema", d

    # cover_nose: both hands over mouth and nose (fast, reliable multi-hand and pose tracking)
    hands_covering = len(hands) >= 2 and near(hands[0].palm, face.mouth, 0.70) and near(hands[1].palm, face.mouth, 0.70) and (hands[0].palm[1] > face.eye_y and hands[1].palm[1] > face.eye_y)
    wrists_at_mouth = bool(body and body.seen and body.wrists_seen and
                           (body.wrists[:, 1] > face.eye_y).all() and
                           dist(body.wrists[0], face.mouth) < 0.65 * fw and dist(body.wrists[1], face.mouth) < 0.65 * fw)
    one_hand_wrists_covering = bool(len(hands) >= 1 and near(hands[0].palm, face.mouth, 0.70) and
                                    body and body.seen and body.wrists_seen and
                                    (body.wrists[:, 1] > face.eye_y).all() and
                                    any(dist(w, face.mouth) < 0.65 * fw for w in body.wrists))
    if hands_covering or wrists_at_mouth or one_hand_wrists_covering:
        return "cover_nose", d

    # SHIT fallback: 1 hand clutching top of head (MediaPipe often merges or drops 1 hand on head)
    if len(hands) == 1:
        h = hands[0]
        hand_on_head = (dist(h.palm, face.top) < 0.65 * fw) and (h.palm[1] < face.eye_y)
        not_temple = not (head_turned and (near(h.index, temple_r, 0.32) or near(h.index, temple_l, 0.32)))
        pose_confirms = (body is None or not body.seen or body.elbows_up or
                         any(w[1] < face.eye_y + 0.25 * fw for w in body.wrists))
        if hand_on_head and not_temple and pose_confirms and not screaming:
            return "shit", d

    # Pose-assisted fallback for SHIT & CINEMA if HandLandmarker misses hands on head or in air
    if body and body.seen and body.wrists_seen:
        w0, w1 = body.wrists[0], body.wrists[1]
        wrists_dist = dist(w0, w1) / fw
        lat_w0 = abs(w0[0] - face.center[0]) / fw
        lat_w1 = abs(w1[0] - face.center[0]) / fw
        dist_top_w0 = dist(w0, face.top) / fw
        dist_top_w1 = dist(w1, face.top) / fw
        wrists_up = (w0[1] < face.eye_y + 0.20 * fw and w1[1] < face.eye_y + 0.20 * fw)

        if wrists_up:
            # SHIT via pose: wrists on or clutching the head
            if (dist_top_w0 < 0.75 or dist_top_w1 < 0.75) and wrists_dist < 1.05 and (lat_w0 < 0.68 and lat_w1 < 0.68):
                return "shit", d

            # CINEMA via pose: wrists high in the air, wide apart, away from head
            if dist_top_w0 > 0.55 and dist_top_w1 > 0.55 and wrists_dist >= 0.95 and (lat_w0 > 0.45 and lat_w1 > 0.45):
                return "cinema", d

    # Crashing out if elbows up, screaming, and hands near head
    near_head = lambda h: abs(h.palm[0] - face.nose[0]) < 1.3 * fw and h.palm[1] < face.eye_y + 0.3 * face.h
    if elbows_up and screaming and all(near_head(h) for h in hands):
        return "crashing_out", d

    # 2. Single-hand gestures
    for h in hands:
        # nose_closed: thumb and index pinching the nose bridge shut (fingers at nose, above mouth)
        pinch_dist = dist(h.thumb, h.index)
        near_nose = near(h.thumb, face.nose, 0.28) and near(h.index, face.nose, 0.28)
        at_nose_height = min(h.index[1], h.thumb[1]) < face.mouth[1] - 0.02 * fw
        if near_nose and (pinch_dist < 0.22 * fw) and at_nose_height:
            return "nose_closed", d

        if near(h.index, face.mouth, 0.22) and not near(h.palm, face.mouth, 0.3):
            return "flirty", d

        # son: closed fist directly in front of the mouth (head straight, not pinching nose)
        at_mouth_height = (face.nose[1] + 0.02 * fw < h.palm[1] < face.chin[1] + 0.05 * fw)
        not_pinching = not (near_nose and pinch_dist < 0.22 * fw)
        if not head_turned and near(h.palm, face.mouth, 0.36) and h.fist and at_mouth_height and not_pinching:
            return "son", d

        # MIND: Finger approaching temple + head turn
        near_temple = near(h.index, temple_r, 0.32) or near(h.index, temple_l, 0.32)
        temple_height = h.index[1] < face.eye_y + 0.18 * fw
        not_scalp = dist(h.index, face.top) > 0.20 * fw
        if head_turned and h.index_ext and (h.num_ext <= 2) and near_temple and temple_height and not_scalp:
            return "mind", d

        # MEW: Finger moving along jawline + head turn (requires extended index and sliding motion)
        if head_turned and mew_tracker and now is not None:
            active_jaw = jaw_pts_r if (face.turn_signed > 0) else jaw_pts_l
            if h.index_ext and (h.num_ext <= 2) and mew_tracker.update(h.index, active_jaw, fw, now):
                return "mew", d

        # INTELLECTUAL: Hand resting near chin + head turn (stationary thinking pose)
        near_chin = near(h.palm, face.chin, 0.48) or near(h.index, face.chin, 0.45) or near(h.wrist, face.chin, 0.55)
        hand_chin_pt = h.palm if near(h.palm, face.chin, 0.48) else h.index
        chin_height = face.mouth[1] - 0.10 * fw < hand_chin_pt[1] < face.chin[1] + 0.45 * fw
        is_moving_jaw = (mew_tracker.has_movement(fw) if mew_tracker else False) or (gesture > 0.045)
        if head_turned and near_chin and chin_height and not h.index_up and not is_moving_jaw:
            return "intellectual", d

        # ME: Pointing index finger toward chest
        chest_y_min = face.chin[1] + 0.05 * fw
        chest_y_max = face.chin[1] + 2.00 * fw
        chest_x_min = face.center[0] - 0.50 * fw
        chest_x_max = face.center[0] + 0.50 * fw
        in_chest_box = (chest_x_min < h.index[0] < chest_x_max) and (chest_y_min < h.index[1] < chest_y_max)
        in_chest_palm = (chest_x_min < h.palm[0] < chest_x_max) and (chest_y_min < h.palm[1] < chest_y_max)
        not_pointing_up = not h.index_up
        not_at_chin = dist(h.index, face.chin) > 0.16 * fw and dist(h.palm, face.chin) > 0.16 * fw
        is_pointing_finger = h.index_ext and (h.num_ext <= 2 or dist(h.wrist, h.index) > 1.20 * dist(h.wrist, h.middle))
        closer_to_chest = dist(h.index, chest) <= dist(h.wrist, chest) + 0.10 * fw
        if is_pointing_finger and (in_chest_box or in_chest_palm) and not_pointing_up and not_at_chin and closer_to_chest:
            return "me", d

        # IDEA vs ACTUALLY: Raised index finger vertically
        at_head_or_upper = face.top[1] - 0.6 * fw < h.index[1] < face.chin[1] + 1.2 * fw
        not_temple = not (head_turned and (near(h.index, temple_r, 0.28) or near(h.index, temple_l, 0.28)))
        not_chest = dist(h.index, chest) > 0.30 * fw
        if h.index_up and (h.num_ext <= 2) and at_head_or_upper and not_temple and not_chest:
            if smooth_smile > 0.26:
                return "idea", d
            elif smooth_smile < 0.16:
                return "actually", d
            else:
                if last_choice == "idea" and smooth_smile >= 0.20:
                    return "idea", d
                return "actually", d

    # 3. Facial / Motion expressions
    if over("jaw_open", m, "z_jaw", "jaw"):
        return "open_mouth", d
    if over("sneer", m, "z_sneer", "sneer") or m["z_disgust"] >= Z["disgust"]:
        return "disgusted", d
    # talking_to_wall: animated gesturing / explaining with hands, not just holding hand still in frame
    if hands and gesture > T["gesture"]:
        not_face = all(dist(h.palm, face.center) > 0.40 * fw for h in hands)
        if not_face:
            return "talking_to_wall", d
    # suspicious: side-eye with head turn and squint, WITHOUT hand at face
    hand_near_face = any(near(h.palm, face.center, 0.75) or near(h.index, face.center, 0.75) for h in hands)
    if not hand_near_face and m["turn"] > T["head_turn"] and over("squint", m, "z_squint", "squint"):
        return "suspicious", d
    return None, d


def draw_hud(img, shown, raw, d, face, hands, body, base):
    if face:
        x0, y0, x1, y1 = face.box
        cv2.rectangle(img, (x0, y0), (x1, y1), (0, 255, 0), 1)
    for h in hands:
        cv2.circle(img, (int(h.palm[0]), int(h.palm[1])), 6, (0, 200, 255), -1)
    if body and body.seen:
        for pt in np.vstack([body.shoulders, body.elbows]):
            cv2.circle(img, (int(pt[0]), int(pt[1])), 6, (255, 120, 0), -1)
    g = d.get
    lines = [
        (f"showing: {shown or '-'}   raw: {raw or '-'}   hands: {g('hands', 0)}"
         f"   elbows up: {'Y' if g('elbows_up') else 'n'}", (0, 255, 0)),
        (f"jaw {g('jaw', 0):.2f} = {g('z_jaw', 0):+.1f}s/{Z['jaw_open']:.0f}   "
         f"squint {g('squint', 0):.2f} = {g('z_squint', 0):+.1f}s/{Z['squint']:.0f}   "
         f"turn {g('turn', 0):.2f}   gesture {g('gesture', 0):.3f}", (0, 255, 0)),
        (f"disgust {g('z_disgust', 0):+.1f}s/{Z['disgust']:.0f} = 2x sneer {g('z_sneer', 0):+.1f} "
         f"+ brow {g('z_brow', 0):+.1f} + frown {g('z_frown', 0):+.1f} + lip {g('z_lip', 0):+.1f}", (0, 255, 0)),
        (("NOT CALIBRATED - generic baseline, everything is harder to trigger. press 'c'"
          if base.generic else
          f"calibrated {base.made} on {base.samples} frames   (s = sigma above your neutral)"),
         (0, 140, 255) if base.generic else (200, 200, 200)),
        ("keys: q quit  d hud  c recalibrate  1-9 0 - = [ ] \\ e r t y u i test poses", (0, 255, 0)),
    ]
    for i, (t, colour) in enumerate(lines):
        y = 24 + 22 * i
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
        cv2.putText(img, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=0, help="webcam index (try 1 if 0 is your iPhone)")
    ap.add_argument("--calibrate", action="store_true", help="learn your neutral face, save it, and exit")
    ap.add_argument("--no-calibration", action="store_true", help="ignore calibration.json; use the generic baseline")
    ap.add_argument("--no-vcam", action="store_true", help="preview only; don't start the virtual camera")
    ap.add_argument("--skip-check", action="store_true", help="skip the MediaPipe startup check")
    ap.add_argument("--size", default="1280x720", help="capture size, e.g. 1280x720 or 640x480 (lower = faster)")
    ap.add_argument("--no-flip", action="store_true", help="don't mirror the image")
    args = ap.parse_args()

    calib_path = os.path.join(HERE, CALIB_FILE)
    base = Baseline() if args.no_calibration else Baseline.load(calib_path)
    if base.generic and not args.calibrate and not args.no_calibration:
        print(f"No usable {CALIB_FILE}. Running on the generic baseline — everything is harder to\n"
              f"trigger than it should be. Run:  python {os.path.basename(__file__)} --calibrate")

    model_paths = ensure_models()
    if not args.skip_check:
        preflight(model_paths["face_landmarker.task"])

    cap = cv2.VideoCapture(args.camera)
    if cap.isOpened() and "x" in args.size:
        w, h = args.size.lower().split("x")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(w))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(h))
    ok, frame = False, None
    if cap.isOpened():
        for _ in range(5):
            ok, frame = cap.read()
            if not ok:
                break
    if not ok:
        sys.exit(f"Could not read from camera {args.camera}.\n"
                 "  - try --camera 1\n"
                 "  - System Settings > Privacy & Security > Camera: allow your terminal app, then re-run")
    H, W = frame.shape[:2]
    print(f"Camera {args.camera}: {W}x{H}")

    clock = Clock()
    window = "it's giving v2  (q quit, d HUD, c recalibrate, 1-9 0 - = [ ] \\ e-i test)"

    if args.calibrate:
        face_det = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
            base_options=mp_tasks.BaseOptions(model_asset_path=model_paths["face_landmarker.task"]),
            running_mode=vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True))
        try:
            new = run_calibration(cap, face_det, clock, args, W, H, window)
        finally:
            face_det.close()
            cap.release()
            cv2.destroyAllWindows()
        if new is None:
            sys.exit(1)
        new.save(calib_path)
        print(f"Saved {CALIB_FILE}. Now run:  python {os.path.basename(__file__)}")
        return

    print("Assets:")
    assets = {pose: load_asset(pose) for pose in POSES}

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
    mew_tracker = MewTracker()
    inactivity_tracker = InactivityTracker(timeout=5.0)
    smooth_smile = 0.0
    last_reaction = None
    shown, hold, show_hud = None, 0, True
    arm = {p: 0 for p in POSES}
    shown_since = 0.0
    forced, forced_until = None, 0.0
    sm_center, sm_h = np.array([W / 2, H / 2], np.float32), H * 0.45
    print("Running. Focus the preview window: q quit, d HUD, c recalibrate, 1-9 0 - = [ ] \\ e-i test a pose")

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

            ts = clock.next()
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            fr = face_det.detect_for_video(mp_img, ts)
            hr = hand_det.detect_for_video(mp_img, ts)
            pr = pose_det.detect_for_video(mp_img, ts)
            face = Face(fr.face_landmarks[0], fr.face_blendshapes[0] if fr.face_blendshapes else None, W, H) \
                if fr.face_landmarks else None
            hands = [Hand(h, W, H) for h in hr.hand_landmarks]
            body = Body(pr.pose_landmarks[0], W, H) if pr.pose_landmarks else None

            m = measure(face, base) if face is not None else {}
            tongue = 0.0
            gesture = motion.update(hands, face)
            now = time.monotonic()
            raw_smile = m.get("smile", 0.0) if face is not None else 0.0
            smooth_smile = 0.8 * smooth_smile + 0.2 * raw_smile
            raw, dbg = decide(face, hands, body, tongue, gesture, m,
                              mew_tracker=mew_tracker, now=now,
                              last_choice=last_reaction, smooth_smile=smooth_smile)

            if raw is None and face is not None:
                if inactivity_tracker.update(face, hands, gesture, raw, now):
                    raw = "soldier"
            else:
                inactivity_tracker.reset(now)

            if raw is not None:
                last_reaction = raw

            fired = None
            for p in POSES:
                arm[p] = arm[p] + 1 if raw == p else 0
                if raw == p and arm[p] >= ARM.get(p, 3):
                    fired = p
            if forced and now < forced_until:
                fired = forced
            if fired:
                if fired != shown:
                    shown_since = now
                shown, hold = fired, (45 if fired == "soldier" else HOLD_FRAMES)
            elif hold > 0:
                hold -= 1
            else:
                shown = None

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
                draw_hud(preview, shown, raw, dbg, face, hands, body, base)
            cv2.imshow(window, preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("d"):
                show_hud = not show_hud
            elif key == ord("c"):
                new = run_calibration(cap, face_det, clock, args, W, H, window)
                if new is not None:
                    base = new
                    base.save(calib_path)
                    print(f"Saved {CALIB_FILE}.")
                motion, shown, hold = Motion(), None, 0
                mew_tracker.reset()
                inactivity_tracker.reset(time.monotonic())
                arm = {p: 0 for p in POSES}
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
