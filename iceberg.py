"""
 What They See vs What It Took  -  made of particles
 pip install numpy opencv-python
 python iceberg.py            (live window)
 python iceberg.py --save     (writes out/iceberg.mp4)
"""
import os
import subprocess
import sys
import time
import numpy as np
import cv2

W, H = 600, 830          # canvas
FPS = 60
SLOW = 14.0 / 11.0       # >1 = slower
DURATION = 16.0          # seconds
N = 15000                # particles
IMAGE = "iceberg.png"
WATER_Y = 268

TOP = np.array([245, 255, 130], np.float32)    # BGR  electric cyan
MID = np.array([255, 175, 70], np.float32)     # BGR  sky blue
BOT = np.array([255, 120, 195], np.float32)    # BGR  violet
ORB = np.array([255, 120, 255], np.float32)    # BGR  pink spark

WALK_PATH = np.array([
    [68, 267], [142, 267], [214, 267],
    [242, 232], [272, 204], [286, 192],
], np.float32)

CLIMB_PATH = np.array([
    [296, 742], [262, 684], [222, 608], [190, 554],
    [166, 484], [136, 428], [120, 396], [148, 330],
    [178, 304], [214, 267], [242, 232], [272, 204], [286, 192],
], np.float32)


def load_lines(path):
    here = os.path.dirname(os.path.abspath(__file__))
    full = path if os.path.isabs(path) else os.path.join(here, path)
    img = cv2.imdecode(np.fromfile(full, np.uint8), cv2.IMREAD_GRAYSCALE)
    ink = (255 - img).astype(np.float32) / 255.0
    s = min((W - 40) / ink.shape[1], (H - 40) / ink.shape[0])
    ink = cv2.resize(ink, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    avg = cv2.GaussianBlur(ink, (0, 0), 3.0)
    ink = np.clip(ink * 1.35 - avg * .45, 0, 1)
    ink = np.clip((ink - .08) / .92, 0, 1) ** .85
    out = np.zeros((H, W), np.float32)
    oy, ox = (H - ink.shape[0]) // 2, (W - ink.shape[1]) // 2
    out[oy:oy + ink.shape[0], ox:ox + ink.shape[1]] = ink
    return out


def gradient(y):
    t = np.clip(y / H, 0, 1)[..., None]
    a = np.clip(t * 2, 0, 1)
    b = np.clip(t * 2 - 1, 0, 1)
    return TOP * (1 - a) + MID * a * (1 - b) + BOT * b


def ease(t):
    t = np.clip(t, 0, 1)
    return np.where(t < .5, 4 * t ** 3, 1 - (-2 * t + 2) ** 3 / 2)


def sample_targets(lines, n, rng):
    ys, xs = np.nonzero(lines > 0.25)
    w = lines[ys, xs] ** 2
    idx = rng.choice(len(xs), n, p=w / w.sum())
    return np.stack([xs[idx], ys[idx]], 1).astype(np.float32)


def sphere(n):
    i = np.arange(n) + .5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** .5) * i
    return np.stack([np.cos(th) * np.sin(phi),
                     np.cos(phi),
                     np.sin(th) * np.sin(phi)], 1).astype(np.float32)


def splat(canvas, pts, col, alpha):
    x = pts[:, 0].astype(np.int32)
    y = pts[:, 1].astype(np.int32)
    ok = (x >= 0) & (x < W) & (y >= 0) & (y < H) & (alpha > 0)
    idx = y[ok] * W + x[ok]
    vals = col[ok] * alpha[ok, None]
    flat = canvas.reshape(-1, 3)
    for c in range(3):
        flat[:, c] += np.bincount(idx, weights=vals[:, c], minlength=W * H)


def path_sample(pts, u):
    segs = np.diff(pts, axis=0)
    lens = np.linalg.norm(segs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    d = float(np.clip(u, 0.0, 1.0)) * cum[-1]
    i = int(np.clip(np.searchsorted(cum, d, side="right") - 1, 0, len(segs) - 1))
    local = (d - cum[i]) / max(lens[i], 1e-6)
    pos = pts[i] + segs[i] * local
    tx, ty = segs[i] / max(lens[i], 1e-6)
    return pos, np.array([tx, ty], np.float32)


def draw_walker(canvas, foot, tangent, stride, celebrate=0.0, fade=1.0, climb=0.0):
    """Draw a glowing articulated stick figure on the canvas."""
    if fade <= 0.02:
        return
    fx, fy = float(foot[0]), float(foot[1])
    dir_x = 1.0 if tangent[0] >= -0.2 else -1.0
    slope = max(0.0, -float(tangent[1]))
    lean = (0.14 + 0.28 * slope + 0.14 * climb) * (1.0 - celebrate) * dir_x
    bob = abs(np.sin(stride)) * 2.4 * (1.0 - celebrate)

    hip = np.array([fx, fy - 10.5 + bob * .7], np.float32)
    sh = np.array([hip[0] + np.sin(lean) * 10.0, hip[1] - np.cos(lean) * 10.0], np.float32)
    head = np.array([sh[0] + np.sin(lean) * 4.8, sh[1] - 4.8], np.float32)

    base_col = gradient(np.array([fy], np.float32))[0] * .45 + np.float32(255.0) * .55
    col = tuple((base_col * fade).tolist())
    bright = tuple((np.minimum(255.0, base_col * 1.25) * fade).tolist())

    def seg(a, b, th=2):
        cv2.line(canvas, (int(round(a[0])), int(round(a[1]))),
                 (int(round(b[0])), int(round(b[1]))), col, th, cv2.LINE_AA)

    seg(hip, sh, 2)
    cv2.circle(canvas, (int(round(head[0])), int(round(head[1]))), 3, bright, -1, cv2.LINE_AA)

    # Legs with knee bend
    for sgn in (-1.0, 1.0):
        ph = stride + (0.0 if sgn > 0 else np.pi)
        swing = np.sin(ph) * (1.0 - celebrate)
        lift = max(0.0, -np.cos(ph)) * (2.6 + 2.2 * climb) * (1.0 - celebrate)
        knee = hip + np.array([dir_x * (2.2 + 3.4 * swing), 5.5 - lift * .7], np.float32)
        ft = np.array([fx + dir_x * (5.2 * swing + 1.4 * sgn * celebrate),
                       fy - lift - slope * max(0.0, swing) * 3.5], np.float32)
        seg(hip, knee, 2)
        seg(knee, ft, 2)

    # Arms: swing opposite to legs when walking/climbing, raise into V on celebration
    for sgn in (-1.0, 1.0):
        ph = stride + (np.pi if sgn > 0 else 0.0)
        asw = np.sin(ph) * (1.0 - celebrate)
        walk_el = sh + np.array([dir_x * (1.6 + 3.2 * asw) + climb * dir_x * 2.2,
                                 4.6 - climb * (1.8 + 2.2 * max(0.0, asw))], np.float32)
        walk_hd = walk_el + np.array([dir_x * (2.8 * asw + 2.0 * climb),
                                      3.4 - climb * 3.6], np.float32)
        cel_el = sh + np.array([sgn * 4.2, -3.8], np.float32)
        cel_hd = sh + np.array([sgn * 7.0, -8.5], np.float32)
        el = walk_el * (1.0 - celebrate) + cel_el * celebrate
        hd = walk_hd * (1.0 - celebrate) + cel_hd * celebrate
        seg(sh, el, 2)
        seg(el, hd, 2)


class Iceberg:
    def __init__(self, path=IMAGE, seed=7):
        rng = np.random.default_rng(seed)
        self.lines = load_lines(path)[..., None]
        self.core = (self.lines ** 1.8) * np.float32(65.0)
        self.yy = np.arange(H, dtype=np.float32)
        self.grad_y = gradient(self.yy)[:, None, :]
        self.target = sample_targets(self.lines[..., 0], N, rng)

        # Above-waterline tip + surface ripples + "WHAT THEY SEE / SUCCESS"
        tip_mask = np.clip((WATER_Y + 14.0 - self.yy) / 14.0, 0.0, 1.0)[:, None, None]
        self.tip_lines = self.lines * tip_mask
        self.tip_core = self.core * tip_mask
        self.tip_src = sample_targets(self.tip_lines[..., 0], N, rng)
        self.tip_col = gradient(self.tip_src[:, 1])

        self.ball = sphere(N)[rng.permutation(N)]
        self.ball *= (1 + rng.normal(0, .07, N).astype(np.float32))[:, None]
        self.delay = (self.target[:, 1] / H) * 1.3 + rng.random(N) * .4
        self.swirl = rng.normal(0, 60, N).astype(np.float32)
        self.tcol = gradient(self.target[:, 1])
        self.center = np.array([W / 2, H * .44], np.float32)
        self.R = 210.0
        self.ones = np.ones(N, np.float32)
        self.white = np.full((N, 3), 255, np.float32)
        self.bcol = self.white * .45 + self.tcol * .55
        self.cv = np.zeros((H, W, 3), np.float32)
        self.small = np.zeros((H // 2, W // 2, 3), np.float32)
        self.glow = np.zeros((H, W, 3), np.float32)

        g = np.zeros((H, W), np.float32)
        cx, cy = map(int, self.center)
        cv2.circle(g, (cx, cy), 18, 1.0, -1)
        self.orb_glow = cv2.GaussianBlur(g, (0, 0), 22)[..., None] * 9 * ORB

    def ball_pos(self, t, grow=1.0):
        a, tilt = t * 1.6, .35
        p = self.ball
        x = p[:, 0] * np.cos(a) + p[:, 2] * np.sin(a)
        z = -p[:, 0] * np.sin(a) + p[:, 2] * np.cos(a)
        y = p[:, 1] * np.cos(tilt) - z * np.sin(tilt)
        z = p[:, 1] * np.sin(tilt) + z * np.cos(tilt)
        xy = self.center + np.stack([x, y], 1) * self.R * grow
        depth = (z + 1.6) / 2.6
        return xy, depth

    def draw_art(self, lines, core, reveal=1.0, gain=1.0):
        band, pos = 160, reveal * (H + 160)
        y = self.yy[:, None, None]
        mask = np.clip((pos - y) / band, 0, 1)
        edge = np.exp(-((pos - band * .5 - y) / 18) ** 2)
        row = (self.grad_y * mask * .95 + 255 * edge * .85) * gain
        np.multiply(lines, row, out=self.cv)
        self.cv += core * (mask * gain)

    def draw_trail(self, pts, u_now, climb=0.0):
        # Trailing particles behind the moving stick figure
        sparks, cols, alphas = [], [], []
        for k in range(28):
            uk = u_now - k * 0.0055
            if uk < 0.0:
                break
            p, _ = path_sample(pts, uk)
            ox = np.sin(k * 7.3 + u_now * 40.0) * (2.0 + 3.5 * climb)
            oy = -k * 0.65 - np.abs(np.cos(k * 5.1)) * 5.0
            sparks.append([p[0] + ox, p[1] + oy])
            c = gradient(np.array([p[1]], np.float32))[0] * .5 + np.float32(255.0) * .5
            cols.append(c)
            alphas.append((1.0 - k / 28.0) * 1.1)
        if sparks:
            splat(self.cv, np.array(sparks, np.float32),
                  np.array(cols, np.float32), np.array(alphas, np.float32))

    def frame(self, t):
        cv = self.cv
        ones, bcol = self.ones, self.bcol

        # 0  above-water tip + stick figure walking easily to the summit
        if t < 2.15:
            self.draw_art(self.tip_lines, self.tip_core, min(1.0, (t + .25) / .6))
            u = float(ease(t / 1.65))
            pos, tan = path_sample(WALK_PATH, u)
            cel = float(ease((t - 1.58) / .35))
            self.draw_trail(WALK_PATH, u, climb=0.2)
            draw_walker(cv, pos, tan, u * 16.0 * np.pi, celebrate=cel)

        # 1  surface illusion shatters into the spinning bluish sphere
        elif t < 3.05:
            cv.fill(0)
            u = float(ease((t - 2.15) / .9))
            xy1, d = self.ball_pos(t)
            xy = self.tip_src * (1 - u) + xy1 * u
            col = self.tip_col * (1 - u) + bcol * u
            alpha = ones * 1.4 * (1 - u) + d * d * 1.6 * u
            splat(cv, xy, col, alpha)
            cv2.GaussianBlur(cv, (0, 0), .6, dst=cv)

        # 2  the sphere spins + flash at center
        elif t < 4.3:
            cv.fill(0)
            xy, d = self.ball_pos(t)
            splat(cv, xy, bcol, d * d * 1.6)
            cv2.GaussianBlur(cv, (0, 0), .6, dst=cv)

        # 3  particles swirl outward into the full hidden iceberg
        elif t < 6.3:
            cv.fill(0)
            xy0, d = self.ball_pos(4.3)
            u = ease((t - 4.3 - self.delay * .65) / 1.2)
            mx = u[:, None]
            dx = self.target[:, 0] - xy0[:, 0]
            dy = self.target[:, 1] - xy0[:, 1]
            side = np.stack([-dy, dx], 1)
            side /= np.linalg.norm(side, axis=1, keepdims=True) + 1e-6
            arc = np.sin(np.pi * u)[:, None] * self.swirl[:, None] * side
            xy = xy0 * (1 - mx) + self.target * mx + arc
            col = bcol * (1 - mx) + self.tcol * mx
            splat(cv, xy, col, ones * (d * d * (1 - u) + u) * 1.5)
            cv2.GaussianBlur(cv, (0, 0), .6, dst=cv)

        # 4  light reveals the colossal iceberg + climber ascends from the deep base
        else:
            reveal = (t - 6.3) / 1.9
            fade = max(0.0, 1.0 - float(ease(reveal)))
            if fade > .01:
                cv.fill(0)
                twinkle = .6 + .4 * np.sin(t * 6 + self.swirl)
                splat(cv, self.target, self.tcol, ones * fade * twinkle * 1.4)
                cv2.GaussianBlur(cv, (0, 0), .6, dst=cv)
                particles = cv.copy()
                self.draw_art(self.lines, self.core, reveal)
                cv += particles
            else:
                self.draw_art(self.lines, self.core, reveal)

            if t > 6.65:
                u = float(ease((t - 6.65) / 5.32))
                pos, tan = path_sample(CLIMB_PATH, u)
                cel = float(ease((t - 11.82) / .4))
                appear = float(ease((t - 6.65) / .35))
                self.draw_trail(CLIMB_PATH, u, climb=1.0)
                draw_walker(cv, pos, tan, u * 30.0 * np.pi,
                            celebrate=cel, fade=appear, climb=1.0 - cel)

        orb_k = np.exp(-((t - 3.85) / .52) ** 2)
        if orb_k > .01:
            cv += self.orb_glow * orb_k

        cv2.resize(cv, (W // 2, H // 2), dst=self.small,
                   interpolation=cv2.INTER_AREA)
        cv2.GaussianBlur(self.small, (0, 0), 3.5, dst=self.small)
        cv2.resize(self.small, (W, H), dst=self.glow,
                   interpolation=cv2.INTER_LINEAR)
        k = .8 - .52 * float(ease((t - 6.3) / 1.8)) if t > 2.15 else .28
        cv2.addWeighted(cv, 1.0, self.glow, k, 0.0, dst=cv)
        return cv2.convertScaleAbs(cv)

    def frames(self):
        for i in range(int(DURATION * FPS)):
            yield self.frame(i / FPS / SLOW)


if __name__ == "__main__":
    z = Iceberg()
    if "--save" in sys.argv:
        here = os.path.dirname(os.path.abspath(__file__))
        os.makedirs(os.path.join(here, "out"), exist_ok=True)
        out_path = os.path.join(here, "out", "iceberg.mp4")
        ff = subprocess.Popen([
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
            "-c:v", "libx264", "-preset", "slow", "-crf", "16",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", out_path,
        ], stdin=subprocess.PIPE)
        for f in z.frames():
            ff.stdin.write(f.tobytes())
        ff.stdin.close()
        ff.wait()
    else:
        t0 = time.perf_counter()
        for i, f in enumerate(z.frames()):
            cv2.imshow("iceberg", f)
            ms = max(1, int((t0 + (i + 1) / FPS - time.perf_counter()) * 1000))
            if cv2.waitKey(ms) == 27:
                break
        cv2.destroyAllWindows()
