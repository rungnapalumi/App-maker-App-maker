# src/cv_proxy_pose.py
# ------------------------------------------------------------
# Classical CV "Proxy Pose" (NO ML MODEL)
# - Extracts silhouette-based proxy metrics from video
# - Works well when: camera is static, subject is full-body, background is simple
# - Returns features per sampled frame:
#     - centroid (x,y) normalized
#     - bbox (x,y,w,h) normalized
#     - area ratio
#     - upright angle (PCA major-axis vs vertical)  [proxy posture]
#     - bottom width ratio (proxy stance base)
#     - motion energy (proxy dynamics)
#
# Dependencies: opencv-python-headless, numpy
# ------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None  # type: ignore

try:
    import numpy as np  # type: ignore
except Exception:
    np = None  # type: ignore


# -------------------------
# Data structures
# -------------------------

@dataclass
class ProxyFrameFeature:
    t_sec: float
    centroid_xy: Tuple[float, float]          # normalized (0..1)
    bbox_xywh: Tuple[float, float, float, float]  # normalized (0..1)
    area_ratio: float                         # silhouette area / frame area
    upright_angle_deg: float                  # 0=perfect vertical, 90=horizontal (proxy)
    bottom_width_ratio: float                 # bottom width / bbox width (proxy)
    motion_energy: float                      # mean abs diff (proxy)
    ok: bool
    reason: str = ""


@dataclass
class ProxyExtractResult:
    engine: str
    video_path: str
    sample_fps: float
    processed_frames: int
    ok_frames: int
    duration_sec: float
    frame_w: int
    frame_h: int
    features: List[ProxyFrameFeature]
    debug: Dict[str, Any]


@dataclass
class CVScore:
    score_0_100: int
    level: str  # High/Moderate/Low (or Thai if you map outside)
    details: Dict[str, Any]


def _require_libs():
    if cv2 is None:
        raise RuntimeError("cv2 is not available. Install opencv-python-headless.")
    if np is None:
        raise RuntimeError("numpy is not available. Install numpy.")


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _safe_mean(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return float(np.mean(vals))  # type: ignore


def _safe_std(vals: List[float]) -> Optional[float]:
    if not vals:
        return None
    return float(np.std(vals))  # type: ignore


def _level_from_score(score: int, lang: str = "en") -> str:
    s = int(score)
    if (lang or "").lower().startswith("th"):
        if s >= 75:
            return "สูง"
        if s >= 45:
            return "ปานกลาง"
        return "ต่ำ"
    else:
        if s >= 75:
            return "High"
        if s >= 45:
            return "Moderate"
        return "Low"
def _largest_contour(binary_mask) -> Optional[Any]:
    # Return largest contour by area
    cnts, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def _pca_upright_angle_deg(points_xy: "np.ndarray") -> float:
    """
    points_xy: Nx2 float
    Returns angle from vertical axis (0=vertical, 90=horizontal)
    """
    if points_xy.shape[0] < 10:
        return 90.0

    mean = points_xy.mean(axis=0)
    centered = points_xy - mean
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eig(cov)
    # major axis = eigenvector with max eigenvalue
    idx = int(np.argmax(eigvals))
    v = eigvecs[:, idx]
    vx, vy = float(v[0]), float(v[1])

    # angle to vertical: vertical vector is (0,1)
    # Compute absolute angle between v and vertical
    # cos(theta) = |v·vertical| / (|v||vertical|) = |vy|/|v|
    norm = math.sqrt(vx * vx + vy * vy) + 1e-9
    cos_t = abs(vy) / norm
    cos_t = max(-1.0, min(1.0, cos_t))
    theta = math.degrees(math.acos(cos_t))  # 0..90
    return float(theta)


def _bottom_width_ratio(binary_mask, bbox: Tuple[int, int, int, int]) -> float:
    """
    Estimate stance base width proxy:
    Take a horizontal slice near the bottom of bbox and measure foreground span.
    Return span / bbox_width (0..1)
    """
    x, y, w, h = bbox
    if w <= 1 or h <= 1:
        return 0.0

    # slice at 90-95% height of bbox (near feet)
    y_slice = int(y + h * 0.93)
    y_slice = max(0, min(binary_mask.shape[0] - 1, y_slice))

    row = binary_mask[y_slice, x:x+w]
    if row is None or row.size == 0:
        return 0.0

    fg = np.where(row > 0)[0]
    if fg.size == 0:
        return 0.0

    span = int(fg.max() - fg.min() + 1)
    return float(span / float(w + 1e-9))


def extract_proxy_features_from_video(
    video_path: str,
    sample_fps: float = 6.0,
    max_frames: int = 1200,
    downscale_to_width: int = 640,
    bg_history: int = 300,
    bg_var_threshold: int = 16,
    min_area_ratio: float = 0.01,
) -> ProxyExtractResult:
    """
    Classical CV extraction:
    - Background subtractor (MOG2)
    - Largest contour = person silhouette (proxy)
    - PCA angle = uprightness proxy
    - bottom width slice = stance proxy
    - frame diff energy = motion proxy
    """
    _require_libs()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return ProxyExtractResult(
            engine="cv_proxy_pose",
            video_path=video_path,
            sample_fps=sample_fps,
            processed_frames=0,
            ok_frames=0,
            duration_sec=0.0,
            frame_w=0,
            frame_h=0,
            features=[],
            debug={"ok": False, "reason": "cannot_open_video"},
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = float(frame_count / fps) if fps > 0 else 0.0
    if fps <= 0:
        fps = 25.0

    sample_every = int(max(1, round(fps / max(1e-6, sample_fps))))

    # Background subtractor
    bg = cv2.createBackgroundSubtractorMOG2(history=bg_history, varThreshold=bg_var_threshold, detectShadows=False)

    features: List[ProxyFrameFeature] = []
    prev_gray = None

    processed = 0
    ok_frames = 0

    # Read first frame to get size
    ok, frame0 = cap.read()
    if not ok:
        cap.release()
        return ProxyExtractResult(
            engine="cv_proxy_pose",
            video_path=video_path,
            sample_fps=sample_fps,
            processed_frames=0,
            ok_frames=0,
            duration_sec=duration_sec,
            frame_w=0,
            frame_h=0,
            features=[],
            debug={"ok": False, "reason": "empty_video"},
        )

    # rewind to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    orig_h, orig_w = frame0.shape[:2]
    scale = 1.0
    if downscale_to_width and orig_w > downscale_to_width:
        scale = downscale_to_width / float(orig_w)

    frame_w = int(round(orig_w * scale))
    frame_h = int(round(orig_h * scale))

    idx = 0
    while processed < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if idx % sample_every != 0:
            continue

        if scale != 1.0:
            frame = cv2.resize(frame, (frame_w, frame_h), interpolation=cv2.INTER_AREA)

        t_sec = float((idx - 1) / fps)

        # motion energy (frame diff)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is None:
            motion_energy = 0.0
        else:
            diff = cv2.absdiff(gray, prev_gray)
            motion_energy = float(np.mean(diff) / 255.0)
        prev_gray = gray

        # silhouette via background subtractor
        fg = bg.apply(frame)

        # clean mask
        fg = cv2.medianBlur(fg, 5)
        _, fg = cv2.threshold(fg, 200, 255, cv2.THRESH_BINARY)

        kernel = np.ones((5, 5), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, kernel, iterations=2)

        cnt = _largest_contour(fg)
        processed += 1

        if cnt is None:
            features.append(ProxyFrameFeature(
                t_sec=t_sec,
                centroid_xy=(0.5, 0.5),
                bbox_xywh=(0.0, 0.0, 0.0, 0.0),
                area_ratio=0.0,
                upright_angle_deg=90.0,
                bottom_width_ratio=0.0,
                motion_energy=motion_energy,
                ok=False,
                reason="no_contour",
            ))
            continue

        area = float(cv2.contourArea(cnt))
        frame_area = float(frame_w * frame_h)
        area_ratio = float(area / (frame_area + 1e-9))

        if area_ratio < min_area_ratio:
            features.append(ProxyFrameFeature(
                t_sec=t_sec,
                centroid_xy=(0.5, 0.5),
                bbox_xywh=(0.0, 0.0, 0.0, 0.0),
                area_ratio=area_ratio,
                upright_angle_deg=90.0,
                bottom_width_ratio=0.0,
                motion_energy=motion_energy,
                ok=False,
                reason="too_small_area",
            ))
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        # centroid
        M = cv2.moments(cnt)
        if abs(M.get("m00", 0.0)) < 1e-9:
            cx, cy = x + w / 2.0, y + h / 2.0
        else:
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])

        # PCA upright angle using contour points
        pts = cnt.reshape(-1, 2).astype(np.float32)
        upright_angle = _pca_upright_angle_deg(pts)

        # stance base proxy: bottom width ratio
        bw = _bottom_width_ratio(fg, (x, y, w, h))

        # normalize
        cx_n = float(cx / (frame_w + 1e-9))
        cy_n = float(cy / (frame_h + 1e-9))
        bbox_n = (
            float(x / (frame_w + 1e-9)),
            float(y / (frame_h + 1e-9)),
            float(w / (frame_w + 1e-9)),
            float(h / (frame_h + 1e-9)),
        )

        features.append(ProxyFrameFeature(
            t_sec=t_sec,
            centroid_xy=(cx_n, cy_n),
            bbox_xywh=bbox_n,
            area_ratio=area_ratio,
            upright_angle_deg=float(upright_angle),
            bottom_width_ratio=float(bw),
            motion_energy=float(motion_energy),
            ok=True,
            reason="",
        ))
        ok_frames += 1

    cap.release()

    debug = {
        "ok": True,
        "fps": float(fps),
        "frame_count": int(frame_count),
        "duration_sec": duration_sec,
        "sample_every": int(sample_every),
        "min_area_ratio": float(min_area_ratio),
        "scale": float(scale),
    }

    return ProxyExtractResult(
        engine="cv_proxy_pose",
        video_path=video_path,
        sample_fps=sample_fps,
        processed_frames=int(processed),
        ok_frames=int(ok_frames),
        duration_sec=float(duration_sec),
        frame_w=int(frame_w),
        frame_h=int(frame_h),
        features=features,
        debug=debug,
    )
def score_uprightness_cv(extract: ProxyExtractResult, lang: str = "en") -> CVScore:
    """
    Uprightness proxy:
    - Use upright_angle_deg (0 is best)
    - Penalize high angle variability
    """
    ok_feats = [f for f in extract.features if f.ok]
    if len(ok_feats) < 10:
        return CVScore(0, _level_from_score(0, lang), {"reason": "insufficient_ok_frames", "ok_frames": len(ok_feats)})

    angles = [float(f.upright_angle_deg) for f in ok_feats]
    a_mean = float(np.mean(angles))
    a_std = float(np.std(angles))

    # Map mean angle: 0..35 good, >55 poor
    mean_score = 1.0 - _clamp01((a_mean - 10.0) / 45.0)  # 10->1, 55->0
    var_score = 1.0 - _clamp01((a_std - 3.0) / 15.0)     # stable is better

    combined = 0.75 * mean_score + 0.25 * var_score
    score = int(round(_clamp01(combined) * 100))
    return CVScore(score, _level_from_score(score, lang), {
        "angle_mean_deg": a_mean,
        "angle_std_deg": a_std,
        "ok_frames": len(ok_feats),
        "processed_frames": extract.processed_frames,
    })


def score_stance_cv(extract: ProxyExtractResult, lang: str = "en") -> CVScore:
    """
    Stance proxy:
    - base width stability: bottom_width_ratio (0..1)
    - body sway: centroid x std
    """
    ok_feats = [f for f in extract.features if f.ok]
    if len(ok_feats) < 10:
        return CVScore(0, _level_from_score(0, lang), {"reason": "insufficient_ok_frames", "ok_frames": len(ok_feats)})

    bw = [float(f.bottom_width_ratio) for f in ok_feats]
    cx = [float(f.centroid_xy[0]) for f in ok_feats]

    bw_mean = float(np.mean(bw))
    bw_std = float(np.std(bw))
    sway_std = float(np.std(cx))

    # Target base width ratio ~0.45-0.75 is typical silhouette feet span within bbox (very rough)
    # Too small -> feet together; too large -> might be noise/arm swing included
    target = 0.60
    bw_center_score = 1.0 - _clamp01(abs(bw_mean - target) / 0.35)
    bw_consistency = 1.0 - _clamp01((bw_std - 0.05) / 0.20)
    sway_score = 1.0 - _clamp01((sway_std - 0.01) / 0.05)

    combined = 0.40 * bw_center_score + 0.35 * bw_consistency + 0.25 * sway_score
    score = int(round(_clamp01(combined) * 100))

    return CVScore(score, _level_from_score(score, lang), {
        "bottom_width_mean": bw_mean,
        "bottom_width_std": bw_std,
        "centroid_x_sway_std": sway_std,
        "ok_frames": len(ok_feats),
        "processed_frames": extract.processed_frames,
    })


def score_motion_dynamics_cv(extract: ProxyExtractResult, lang: str = "en") -> CVScore:
    """
    Motion dynamics proxy:
    - motion_energy average + variability (not Effort/Shape classification)
    - Use as baseline for "activity level" / "expressiveness"
    """
    ok_feats = [f for f in extract.features if f.ok]
    if len(ok_feats) < 10:
        return CVScore(0, _level_from_score(0, lang), {"reason": "insufficient_ok_frames", "ok_frames": len(ok_feats)})

    me = [float(f.motion_energy) for f in ok_feats]
    m_mean = float(np.mean(me))
    m_std = float(np.std(me))

    # Map mean energy: 0.00..0.05 low, 0.08..0.18 moderate, >0.25 high (very rough)
    mean_component = _clamp01((m_mean - 0.04) / 0.20)
    var_component = _clamp01((m_std - 0.01) / 0.10)

    combined = 0.70 * mean_component + 0.30 * var_component
    score = int(round(_clamp01(combined) * 100))
    return CVScore(score, _level_from_score(score, lang), {
        "motion_energy_mean": m_mean,
        "motion_energy_std": m_std,
        "ok_frames": len(ok_feats),
        "processed_frames": extract.processed_frames,
    })


def analyze_cv_proxy(
    video_path: str,
    lang: str = "en",
    sample_fps: float = 6.0,
    max_frames: int = 1200,
) -> Dict[str, Any]:
    """
    One-shot helper you can call from report_worker.py
    Returns a JSON-serializable dict.
    """
    extract = extract_proxy_features_from_video(
        video_path=video_path,
        sample_fps=sample_fps,
        max_frames=max_frames,
    )

    up = score_uprightness_cv(extract, lang=lang)
    st = score_stance_cv(extract, lang=lang)
    mo = score_motion_dynamics_cv(extract, lang=lang)

    return {
        "engine": extract.engine,
        "ok": True,
        "video_path": video_path,
        "meta": {
            "sample_fps": extract.sample_fps,
            "processed_frames": extract.processed_frames,
            "ok_frames": extract.ok_frames,
            "duration_sec": extract.duration_sec,
            "frame_w": extract.frame_w,
            "frame_h": extract.frame_h,
            "debug": extract.debug,
        },
        "uprightness": {
            "score_0_100": up.score_0_100,
            "level": up.level,
            "details": up.details,
        },
        "stance": {
            "score_0_100": st.score_0_100,
            "level": st.level,
            "details": st.details,
        },
        "motion_dynamics": {
            "score_0_100": mo.score_0_100,
            "level": mo.level,
            "details": mo.details,
        },
    }
