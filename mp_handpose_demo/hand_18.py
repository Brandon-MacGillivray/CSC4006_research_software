#!/usr/bin/env python3
"""
Standalone gesture-driven demonstration application.

This artefact presents two precomputed image-processing demos through live hand
interaction:
- segmentation overlay selection on microscopy imagery
- tone-mapping comparison via a hand-controlled spotlight

Runtime dependencies:
- Python 3.10
- ZED SDK Python API (`pyzed`)
- PySide6, OpenCV, MediaPipe, Pillow, NumPy

Run this file from the repository root so that relative `data/` paths resolve
correctly.
"""

import sys
import time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image  # for loading mask as numpy

from PySide6.QtCore import Qt, QThread, Signal, QPoint, QTimer
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QMouseEvent,
    QColor, QImage, QPainterPath, QPixmap   # <-- added QPixmap
)
from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QStackedLayout, QSizePolicy, QHBoxLayout
)

from pyzed import sl
import mediapipe as mp

# ------------ Config ------------

MIRROR_X = True          # mirror horizontally if needed

# Camera rotation so portrait is "natural"
# Options: None, "CW" (clockwise), "CCW" (counterclockwise)
CAM_ROTATE = "CW"

# Target portrait screen (e.g. 1080x1920)
TARGET_WIDTH = 1080
TARGET_HEIGHT = 1920
FULLSCREEN = True        # set False for a 1080x1920 window during dev

HAND_TIMEOUT_S = 5.0
HAND_RADIUS_SCALE = 2.0  # size of spotlight relative to palm width

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

# ------------ Helpers ------------

def palm_center(lm):
    idxs = [0,1,5,9,13,17]
    pts = np.array([[lm[i].x, lm[i].y] for i in idxs], dtype=np.float32)
    return pts.mean(axis=0)

def palm_width(lm):
    a = np.array([lm[0].x, lm[0].y])
    b = np.array([lm[9].x, lm[9].y])
    return float(np.linalg.norm(a - b)) + 1e-6

def palm_width_uv(lm_uv):
    if not lm_uv or len(lm_uv) < 10:
        return 0.1
    a = np.array(lm_uv[0])
    b = np.array(lm_uv[9])
    return float(np.linalg.norm(a - b)) + 1e-6

def tip_mcp_norm(lm, tip, mcp):
    pw = palm_width(lm)
    tipv = np.array([lm[tip].x, lm[tip].y])
    mcpv = np.array([lm[mcp].x, lm[mcp].y])
    return float(np.linalg.norm(tipv - mcpv) / pw)

def is_fist(lm, t=0.58):
    return (
        tip_mcp_norm(lm, 8, 5)  < t and
        tip_mcp_norm(lm,12, 9) < t and
        tip_mcp_norm(lm,16,13) < t and
        tip_mcp_norm(lm,20,17) < t
    )

# ------------ Vision Thread ------------

class VisionThread(QThread):
    # u, v in [0,1], hand_present, lm_uv list or None, click bool
    pose = Signal(float, float, bool, object, bool)
    camera_size = Signal(int, int)  # width, height (after rotation)

    def __init__(self):
        super().__init__()
        self.running = True

    def run(self):
        zed = sl.Camera()
        init = sl.InitParameters(
            camera_resolution=sl.RESOLUTION.HD720,
            camera_fps=60,
            depth_mode=sl.DEPTH_MODE.NONE
        )
        if zed.open(init) != sl.ERROR_CODE.SUCCESS:
            print("Failed to open ZED.")
            return

        info = zed.get_camera_information()
        res = info.camera_configuration.resolution
        cam_w, cam_h = int(res.width), int(res.height)

        # Adjust reported camera size for portrait logical orientation
        if CAM_ROTATE in ("CW", "CCW"):
            self.camera_size.emit(cam_h, cam_w)  # swap
        else:
            self.camera_size.emit(cam_w, cam_h)

        runtime = sl.RuntimeParameters()
        image = sl.Mat()

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        last_u, last_v = 0.5, 0.5
        prev_fist = False
        last_click = 0.0
        CLICK_DEBOUNCE = 0.15

        while self.running:
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                continue

            zed.retrieve_image(image, sl.VIEW.LEFT)
            bgra = image.get_data()
            if bgra is None:
                continue

            # Convert to BGR
            bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)

            # Rotate so portrait is natural
            if CAM_ROTATE == "CCW":
                bgr = cv2.rotate(bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif CAM_ROTATE == "CW":
                bgr = cv2.rotate(bgr, cv2.ROTATE_90_CLOCKWISE)

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            hand_present = False
            lm_uv = None
            click = False
            u, v = last_u, last_v

            if result.multi_hand_landmarks:
                lm = result.multi_hand_landmarks[0].landmark

                pc = palm_center(lm)
                u, v = float(pc[0]), float(pc[1])
                last_u, last_v = u, v

                lm_uv = [(p.x, p.y) for p in lm]
                hand_present = True

                now = time.time()
                cur = is_fist(lm)
                if cur and not prev_fist:
                    if now - last_click > CLICK_DEBOUNCE:
                        click = True
                        last_click = now
                prev_fist = cur
            else:
                prev_fist = False

            self.pose.emit(u, v, hand_present, lm_uv, click)

        hands.close()
        zed.close()

    def stop(self):
        self.running = False
        self.wait(500)

# ------------ Overlay ------------

class CursorOverlay(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WA_TranslucentBackground, True)

        self.cursor_pos = QPoint(100, 100)
        self.valid = False
        self.lm_px = None
        self.res_label = ""

    def set_cursor(self, p, valid):
        self.cursor_pos = p
        self.valid = valid
        self.update()

    def set_landmarks(self, pts):
        self.lm_px = pts
        self.update()

    def set_res_label(self, t):
        self.res_label = t
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        if self.res_label:
            p.setPen(QPen(QColor(148, 163, 184), 1))
            p.drawText(10, 20, self.res_label)

        if self.lm_px:
            p.setPen(QPen(QColor(96, 165, 250), 2))
            for a, b in HAND_CONNECTIONS:
                p.drawLine(self.lm_px[a], self.lm_px[b])
            p.setBrush(QBrush(QColor(56, 189, 248)))
            p.setPen(Qt.NoPen)
            for pt in self.lm_px:
                p.drawEllipse(pt, 3, 3)

        alpha = 255 if self.valid else 120
        p.setPen(QPen(QColor(244, 244, 245, alpha), 2))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(self.cursor_pos, 12, 12)

        p.setBrush(QBrush(QColor(251, 191, 36, alpha)))
        p.setPen(Qt.NoPen)
        p.drawEllipse(self.cursor_pos, 5, 5)

# ------------ Segmentation Demo Widget ------------

class SegmentationDemoWidget(QWidget):
    """
    Shows:
      - base microscopy image full-frame (letterboxed, keeps aspect)
      - segmentation mask overlay for a selected connected region (click/fist)
    """
    def __init__(self, img_path, mask_path, parent=None):
        super().__init__(parent)
        self.setMouseTracking(False)

        self.base_img = QImage(img_path)

        self.mask_np = None
        self.cc_labels = None
        self.hover_cc = None
        self.overlay_raw = {}
        self.overlay_scaled = {}
        self._scaled_size = None

        try:
            pil_mask = Image.open(mask_path).convert("L")
            self.mask_np = np.array(pil_mask)
        except Exception as e:
            print(f"Failed to load mask {mask_path}: {e}")

        if self.mask_np is not None:
            binary = (self.mask_np > 0).astype(np.uint8)
            if np.any(binary):
                num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
                self.cc_labels = labels
                self._build_overlays(num_labels, labels)
                print(f"Found {num_labels-1} cell components in mask.")
            else:
                print("Mask has no foreground pixels (all zero).")

        self.base_scaled = QImage()
        self.highlight_raw = None
        self.highlight_scaled = QImage()

    def _build_overlays(self, num_labels, labels):
        palette = [
            (56, 189, 248),
            (249, 115, 22),
            (52, 211, 153),
            (244, 114, 182),
            (129, 140, 248),
            (248, 180, 0),
            (248, 113, 113),
            (16, 185, 129),
        ]

        h, w = labels.shape
        for cc_id in range(1, num_labels):
            sel = labels == cc_id
            if not np.any(sel):
                continue

            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            color = palette[(cc_id - 1) % len(palette)]
            rgba[sel, 0] = color[0]
            rgba[sel, 1] = color[1]
            rgba[sel, 2] = color[2]
            rgba[sel, 3] = 200

            qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
            self.overlay_raw[cc_id] = qimg.copy()

    def _compute_display_rect(self):
        if self.base_img.isNull():
            return 1.0, 0, 0, self.width(), self.height()

        img_w = self.base_img.width()
        img_h = self.base_img.height()
        w = max(1, self.width())
        h = max(1, self.height())

        # Fill the widget; crop if necessary to avoid letterboxing
        scale = max(w / img_w, h / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        offset_x = (w - disp_w) // 2
        offset_y = (h - disp_h) // 2
        return scale, offset_x, offset_y, disp_w, disp_h

    def _scale_images(self):
        _, _, _, disp_w, disp_h = self._compute_display_rect()
        new_size = (disp_w, disp_h)
        if self._scaled_size == new_size:
            if self.hover_cc is not None:
                self.highlight_scaled = self.overlay_scaled.get(self.hover_cc, QImage())
            return

        self._scaled_size = new_size

        if not self.base_img.isNull():
            self.base_scaled = self.base_img.scaled(
                disp_w, disp_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )

        self.overlay_scaled = {}
        for cc_id, img in self.overlay_raw.items():
            self.overlay_scaled[cc_id] = img.scaled(
                disp_w, disp_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )

        if self.hover_cc is not None:
            self.highlight_scaled = self.overlay_scaled.get(self.hover_cc, QImage())
        else:
            self.highlight_scaled = QImage()

    def resizeEvent(self, e):
        self._scale_images()
        super().resizeEvent(e)

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position()
        self.select_region(pos.x(), pos.y())

    def select_region(self, x_f, y_f):
        if self.mask_np is None or self.base_img.isNull() or self.cc_labels is None:
            return

        scale, offset_x, offset_y, disp_w, disp_h = self._compute_display_rect()

        if (x_f < offset_x or x_f >= offset_x + disp_w or
            y_f < offset_y or y_f >= offset_y + disp_h):
            if self.hover_cc is not None:
                self.hover_cc = None
                self.highlight_raw = None
                self._scale_images()
                self.update()
            self.hover_cc = None
            self.highlight_raw = None
            self._scale_images()
            self.update()
            return

        ix = int((x_f - offset_x) / scale)
        iy = int((y_f - offset_y) / scale)

        h, w = self.cc_labels.shape
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return

        cc_id = int(self.cc_labels[iy, ix])

        if cc_id == 0:
            if self.hover_cc is not None:
                self.hover_cc = None
                self.highlight_raw = None
                self._scale_images()
                self.update()
            return

        if cc_id == self.hover_cc:
            return

        self.hover_cc = cc_id
        self.highlight_raw = self.overlay_raw.get(cc_id)
        self.highlight_scaled = self.overlay_scaled.get(cc_id, QImage())
        self._scale_images()
        self.update()

    def paintEvent(self, _):
        self._scale_images()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Fill background to avoid white letterbox
        p.fillRect(self.rect(), QColor(2, 6, 23))

        if self.base_img.isNull():
            p.setPen(QPen(QColor(220, 38, 38)))
            p.drawText(10, 20, "Failed to load base image.")
            return

        scale, offset_x, offset_y, disp_w, disp_h = self._compute_display_rect()

        if not self.base_scaled.isNull():
            p.drawImage(offset_x, offset_y, self.base_scaled)

        if not self.highlight_scaled.isNull() and self.hover_cc is not None:
            p.drawImage(offset_x, offset_y, self.highlight_scaled)

# ------------ Multi-mask Segmentation Widget (RNvU) ------------

class MultiMaskSegmentationWidget(QWidget):
    """
    Shows:
      - base microscopy image full-frame (letterboxed)
      - multiple binary masks; click/fist reveals whichever mask covers the pixel
    """
    def __init__(self, img_path, mask_paths, parent=None):
        super().__init__(parent)
        self.setMouseTracking(False)

        self.base_img = QImage(img_path)
        self.base_scaled = QImage()

        self.mask_overlays_raw = []   # QImages for each mask
        self.mask_overlays_scaled = []  # scaled caches
        self.mask_names = []      # stem names for display
        self.label_map = None     # int map to mask index+1
        self.hover_mask = None
        self.hover_label = ""
        self._scaled_size = None

        self.highlight_raw = None
        self.highlight_scaled = QImage()

        if self.base_img.isNull():
            print(f"Failed to load base image {img_path}")
            return

        self._build_masks(mask_paths)

    def _palette(self):
        return [
            (56, 189, 248),
            (249, 115, 22),
            (52, 211, 153),
            (244, 114, 182),
            (129, 140, 248),
            (248, 180, 0),
            (248, 113, 113),
            (16, 185, 129),
        ]

    def _build_masks(self, mask_paths):
        if not mask_paths:
            print("No RNvU mask files found.")
            return

        base_w = self.base_img.width()
        base_h = self.base_img.height()
        label_map = np.zeros((base_h, base_w), dtype=np.int32)
        palette = self._palette()

        for idx, mpath in enumerate(mask_paths):
            try:
                arr = np.array(Image.open(mpath).convert("L"))
            except Exception as e:
                print(f"Failed to load mask {mpath}: {e}")
                continue

            if arr.shape != (base_h, base_w):
                arr = cv2.resize(arr, (base_w, base_h), interpolation=cv2.INTER_NEAREST)

            mask_bool = arr > 0
            if not np.any(mask_bool):
                continue

            label_map[mask_bool] = idx + 1
            color = palette[idx % len(palette)]

            rgba = np.zeros((base_h, base_w, 4), dtype=np.uint8)
            rgba[mask_bool, 0] = color[0]
            rgba[mask_bool, 1] = color[1]
            rgba[mask_bool, 2] = color[2]
            rgba[mask_bool, 3] = 200

            qimg = QImage(rgba.data, base_w, base_h, 4 * base_w, QImage.Format_RGBA8888)
            self.mask_overlays_raw.append(qimg.copy())
            self.mask_names.append(Path(mpath).stem)

        if self.mask_overlays_raw:
            self.label_map = label_map
        else:
            print("No valid RNvU masks contained foreground.")

    def _compute_display_rect(self):
        if self.base_img.isNull():
            return 1.0, 0, 0, self.width(), self.height()

        img_w = self.base_img.width()
        img_h = self.base_img.height()
        w = max(1, self.width())
        h = max(1, self.height())

        # Fill the widget; crop if necessary to avoid letterboxing
        scale = max(w / img_w, h / img_h)
        disp_w = int(img_w * scale)
        disp_h = int(img_h * scale)
        offset_x = (w - disp_w) // 2
        offset_y = (h - disp_h) // 2
        return scale, offset_x, offset_y, disp_w, disp_h

    def _scale_images(self):
        _, _, _, disp_w, disp_h = self._compute_display_rect()
        new_size = (disp_w, disp_h)
        if self._scaled_size == new_size:
            if self.hover_mask is not None:
                self.highlight_scaled = (
                    self.mask_overlays_scaled[self.hover_mask]
                    if 0 <= self.hover_mask < len(self.mask_overlays_scaled)
                    else QImage()
                )
            return

        self._scaled_size = new_size

        if not self.base_img.isNull():
            self.base_scaled = self.base_img.scaled(
                disp_w, disp_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation
            )

        self.mask_overlays_scaled = []
        for img in self.mask_overlays_raw:
            self.mask_overlays_scaled.append(
                img.scaled(disp_w, disp_h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
            )

        if self.hover_mask is not None and 0 <= self.hover_mask < len(self.mask_overlays_scaled):
            self.highlight_scaled = self.mask_overlays_scaled[self.hover_mask]
        else:
            self.highlight_scaled = QImage()

    def resizeEvent(self, e):
        self._scale_images()
        super().resizeEvent(e)

    def mousePressEvent(self, event: QMouseEvent):
        pos = event.position()
        self.select_mask(pos.x(), pos.y())

    def select_mask(self, x_f, y_f):
        if self.label_map is None or self.base_img.isNull():
            return

        scale, offset_x, offset_y, disp_w, disp_h = self._compute_display_rect()

        if (x_f < offset_x or x_f >= offset_x + disp_w or
            y_f < offset_y or y_f >= offset_y + disp_h):
            self.hover_mask = None
            self.hover_label = ""
            self.highlight_raw = None
            self._scale_images()
            self.update()
            return

        ix = int((x_f - offset_x) / scale)
        iy = int((y_f - offset_y) / scale)

        h, w = self.label_map.shape
        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            return

        mask_id = int(self.label_map[iy, ix]) - 1

        if mask_id < 0 or mask_id >= len(self.mask_overlays_raw):
            self.hover_mask = None
            self.hover_label = ""
            self.highlight_raw = None
            self._scale_images()
            self.update()
            return

        if mask_id == self.hover_mask:
            return

        self.hover_mask = mask_id
        self.hover_label = self.mask_names[mask_id]
        self.highlight_raw = self.mask_overlays_raw[mask_id]
        if 0 <= mask_id < len(self.mask_overlays_scaled):
            self.highlight_scaled = self.mask_overlays_scaled[mask_id]
        self._scale_images()
        self.update()

    def paintEvent(self, _):
        self._scale_images()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Fill background to avoid white letterbox
        p.fillRect(self.rect(), QColor(2, 6, 23))

        if self.base_img.isNull():
            p.setPen(QPen(QColor(220, 38, 38)))
            p.drawText(10, 20, "Failed to load base image.")
            return

        scale, offset_x, offset_y, disp_w, disp_h = self._compute_display_rect()

        if not self.base_scaled.isNull():
            p.drawImage(offset_x, offset_y, self.base_scaled)

        if not self.highlight_scaled.isNull() and self.hover_mask is not None:
            p.drawImage(offset_x, offset_y, self.highlight_scaled)

        # Mask label intentionally hidden for a cleaner view

# ------------ Tone Mapping Spotlight Widget ------------

class ToneMappingDemoWidget(QWidget):
    """
    Draws:
      - 'bad' image full-frame (letterboxed, keeps aspect)
      - 'good' image only inside a spotlight circle
    """
    def __init__(self, bad_path, good_path, parent=None):
        super().__init__(parent)
        self.bad_img = QImage()
        self.good_img = QImage()

        self.bad_scaled = QImage()
        self.good_scaled = QImage()

        self.set_images(bad_path, good_path)

        self.spot_pos = QPoint(0, 0)
        self.spot_radius = 50.0

    def set_images(self, bad_path, good_path):
        self.bad_img = QImage(bad_path)
        self.good_img = QImage(good_path)
        self.bad_scaled = QImage()
        self.good_scaled = QImage()
        self._scale()
        self.update()

    def set_spot(self, pos, radius):
        self.spot_pos = QPoint(pos)
        self.spot_radius = float(radius)
        self.update()

    def _scale(self):
        size = self.size()
        if size.width() <= 0 or size.height() <= 0:
            return

        if not self.bad_img.isNull():
            self.bad_scaled = self.bad_img.scaled(
                size,
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation
            )
        if not self.good_img.isNull():
            self.good_scaled = self.good_img.scaled(
                size,
                Qt.KeepAspectRatioByExpanding,
                Qt.SmoothTransformation
            )

    def resizeEvent(self, e):
        self._scale()
        super().resizeEvent(e)

    def paintEvent(self, _):
        self._scale()
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        if not self.bad_scaled.isNull():
            bx = (self.width()  - self.bad_scaled.width())  // 2
            by = (self.height() - self.bad_scaled.height()) // 2
            p.drawImage(bx, by, self.bad_scaled)

        if not self.good_scaled.isNull():
            gx = (self.width()  - self.good_scaled.width())  // 2
            gy = (self.height() - self.good_scaled.height()) // 2

            path = QPainterPath()
            path.addEllipse(self.spot_pos, self.spot_radius, self.spot_radius)

            p.save()
            p.setClipPath(path)
            p.drawImage(gx, gy, self.good_scaled)
            p.restore()

# ------------ Main Window ------------

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Hand UI MVP (Portrait, Spotlight + Segmentation)")

        self.resize(TARGET_WIDTH, TARGET_HEIGHT)
        self.last_hand = time.time()
        self.stack = QStackedLayout(self)

        # ---- Start Page ----
        self.start_page = QWidget()
        self.start_label = QLabel("Wave your hand to begin.")
        self.start_label.setAlignment(Qt.AlignCenter)
        self.start_label.setObjectName("TitleLabel")

        self.go_button = QPushButton("Go")
        self.go_button.setVisible(False)
        self.go_button.setObjectName("PrimaryButton")

        s = QVBoxLayout()
        s.addStretch(1)
        s.addWidget(self.start_label, alignment=Qt.AlignCenter)
        s.setSpacing(20)
        s.addWidget(self.go_button, alignment=Qt.AlignCenter)
        s.addStretch(1)
        self.start_page.setLayout(s)

        # ---- Menu Page ----
        self.menu_page = QWidget()
        self.menu_label = QLabel("Main Menu")
        self.menu_label.setAlignment(Qt.AlignCenter)
        self.menu_label.setObjectName("TitleLabel")

        # University logo at the top of the menu
        self.logo_label = QLabel()
        self.logo_label.setAlignment(Qt.AlignCenter)
        logo_pix = QPixmap("data/Queen’s White Logo - Landscape.png")
        if not logo_pix.isNull():
            self.logo_label.setPixmap(
                logo_pix.scaledToWidth(720, Qt.SmoothTransformation)
            )

        self.btn_back = QPushButton("Back to Start")
        self.btn_seg = QPushButton("Segmentation Demo")
        self.btn_tone = QPushButton("Tone Mapping Demo")

        # More consistent button styling
        self.btn_back.setObjectName("SecondaryButton")
        self.btn_seg.setObjectName("PrimaryButton")
        self.btn_tone.setObjectName("PrimaryButton")

        for btn in (self.btn_back, self.btn_seg, self.btn_tone):
            btn.setMinimumWidth(380)
            btn.setFixedHeight(64)

        m = QVBoxLayout()
        m.setContentsMargins(96, 120, 96, 120)
        m.setSpacing(24)
        m.addWidget(self.logo_label, alignment=Qt.AlignCenter)
        m.addSpacing(100)
        m.addWidget(self.menu_label, alignment=Qt.AlignCenter)
        m.addSpacing(32)
        m.addWidget(self.btn_seg, alignment=Qt.AlignCenter)
        m.addWidget(self.btn_tone, alignment=Qt.AlignCenter)
        m.addSpacing(16)
        m.addWidget(self.btn_back, alignment=Qt.AlignCenter)
        m.addStretch(1)
        self.menu_page.setLayout(m)

        # ---- Segmentation Page ----
        self.seg_page = QWidget()

        # Title + nav buttons
        self.seg_label = QLabel()
        self.seg_label.setAlignment(Qt.AlignCenter)
        self.seg_label.setObjectName("TitleLabel")

        self.seg_menu_btn = QPushButton("Menu")
        self.seg_prev_btn = QPushButton("Previous")
        self.seg_next_btn = QPushButton("Next")
        for btn in (self.seg_menu_btn, self.seg_prev_btn, self.seg_next_btn):
            btn.setObjectName("SecondaryButton")
            btn.setFixedHeight(48)

        # Segmentation demos
        seg_img_path = "data/training_001.tif"
        seg_mask_path = "data/training_groundtruth_001.tif"
        base_seg_widget = SegmentationDemoWidget(seg_img_path, seg_mask_path)

        rnvu_img_path = "data/rnvu_data/Sarah_050.tif"
        rnvu_masks = self._load_rnvu_masks()
        rnvu_widget = MultiMaskSegmentationWidget(rnvu_img_path, rnvu_masks)

        # Description text (to the right of the headshot)
        self.seg_desc = QLabel()
        self.seg_desc.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.seg_desc.setWordWrap(True)
        self.seg_desc.setObjectName("SubtitleLabel")

        # Researcher name (under headshot)
        self.researcher_label = QLabel()
        self.researcher_label.setAlignment(Qt.AlignLeft)
        self.researcher_label.setObjectName("SubtitleLabel")

        # Headshot (left column)
        self.headshot_label = QLabel()
        self.headshot_label.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.seg_examples = [
            {
                "title": "Segment a Cell",
                "desc": (
                    "Information about demo: Segmentation models split an image into useful parts "
                    "such as finding different cells. When you click on the image, the point you "
                    "choose helps the model focus on the nearby shapes and edges! The model uses "
                    "these features to create predictions that show possible cells."
                ),
                "researcher": "Researcher: Victoria Porter",
                "headshot": "data/vp_headshot.jpg",
                "widget": base_seg_widget,
            },
            {
                "title": "RNvU Retina Segmentation",
                "desc": (
                    "Information about demo: Segmentation models split an image into useful parts "
                    "such as finding different cells. When you click on the image, the point you "
                    "choose helps the model focus on the nearby shapes and edges! The model uses "
                    "these features to create predictions that show possible cells."
                ),
                "researcher": "Researcher: Victoria Porter",
                "headshot": "data/vp_headshot.jpg",
                "widget": rnvu_widget,
            },
        ]

        self.seg_widget_stack = QStackedLayout()
        for ex in self.seg_examples:
            ex["widget"].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            self.seg_widget_stack.addWidget(ex["widget"])
        self.seg_index = 0

        # ----- Top container: title + back button (above image) -----
        top_container = QWidget()
        top_layout = QVBoxLayout(top_container)
        top_layout.setContentsMargins(0, 100, 0, 50)
        top_layout.setSpacing(20)
        top_layout.addWidget(self.seg_label, alignment=Qt.AlignCenter)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(12)
        nav_row.addStretch(1)
        nav_row.addWidget(self.seg_menu_btn)
        nav_row.addWidget(self.seg_prev_btn)
        nav_row.addWidget(self.seg_next_btn)
        nav_row.addStretch(1)
        top_layout.addLayout(nav_row)

        # ----- Middle container: image only -----
        middle_container = QWidget()
        middle_layout = QVBoxLayout(middle_container)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(0)
        middle_layout.addLayout(self.seg_widget_stack, stretch=1)

        # ----- Bottom container: headshot + name + logo on left, description on right -----
        bottom_container = QWidget()
        bottom_layout = QHBoxLayout(bottom_container)
        bottom_layout.setContentsMargins(100, 50, 100, 20)
        bottom_layout.setSpacing(20)

        # Left column: headshot, name, logo at bottom
        left_col = QVBoxLayout()
        left_col.setContentsMargins(0, 0, 0, 0)
        left_col.setSpacing(8)
        left_col.addWidget(self.headshot_label)
        left_col.addWidget(self.researcher_label)
        left_col.addStretch(1)

        # Right column: description
        right_col = QVBoxLayout()
        right_col.setContentsMargins(0, 0, 0, 0)
        right_col.setSpacing(8)
        right_col.addWidget(self.seg_desc)
        right_col.addStretch(1)

        bottom_layout.addLayout(left_col, stretch=0)
        bottom_layout.addLayout(right_col, stretch=1)

        # ----- Main segmentation page layout: top / image / bottom -----
        seg_layout = QVBoxLayout()
        seg_layout.setContentsMargins(24, 24, 24, 24)
        seg_layout.setSpacing(16)

        seg_layout.addWidget(top_container, stretch=0)   # title + back button
        seg_layout.addWidget(middle_container, stretch=3)  # main image
        seg_layout.addWidget(bottom_container, stretch=2)  # headshot + text + logo

        self.seg_page.setLayout(seg_layout)
        self._apply_seg_example()

        # ---- Tone Mapping Page ----
        self.tone_page = QWidget()
        self.tone_label = QLabel("Tone Mapping Demo")
        self.tone_label.setAlignment(Qt.AlignCenter)
        self.tone_label.setObjectName("TitleLabel")

        self.tone_desc = QLabel("Move your hand: spotlight reveals tone-mapped result.")
        self.tone_desc.setAlignment(Qt.AlignCenter)
        self.tone_desc.setWordWrap(True)
        self.tone_desc.setObjectName("SubtitleLabel")

        self.tone_pairs = [
            (
                "data/Convallaria_3C_3T_2x2grid_confocal_Channel 1_0_ref1.png",
                "data/Convallaria_3C_3T_2x2grid_confocal_Channel 1_0_Khan20.png"
            ),
            (
                "data/20x_Mouse_Kidney_WF_Channel 1_0_ref1.png",
                "data/20x_Mouse_Kidney_WF_Channel 1_0_Khan20.png"
            ),
            (
                "data/60x_BPAE_WF_Channel 1_0_ref1.png",
                "data/60x_BPAE_WF_Channel 1_0_Khan20.png"
            ),
            (
                "data/60x_BPAE_WF_Channel 2_0_ref.png",
                "data/60x_BPAE_WF_Channel 2_0_ref1.png"
            ),
            (
                "data/Convallaria_3C_3T_35Z_confocal_Channel 0_17_ref1.png",
                "data/Convallaria_3C_3T_35Z_confocal_Channel 0_17_Cao23.png"
            ),
        ]
        self.tone_index = 0

        # Tone demo assets + researcher info
        self.tone_headshot = QLabel()
        self.tone_headshot.setAlignment(Qt.AlignCenter)
        headshot_pix = QPixmap("data/Yelyzaveta Razumovska.png")
        if not headshot_pix.isNull():
            self.tone_headshot.setPixmap(
                headshot_pix.scaledToHeight(180, Qt.SmoothTransformation)
            )

        self.tone_logo = QLabel()
        self.tone_logo.setAlignment(Qt.AlignCenter)
        oi_pix = QPixmap("data/OI Unboxed (White) - 2024.png")
        if not oi_pix.isNull():
            self.tone_logo.setPixmap(
                oi_pix.scaledToWidth(320, Qt.SmoothTransformation)
            )

        self.tone_researcher = QLabel("Yelyzaveta Razumovska")
        self.tone_researcher.setAlignment(Qt.AlignCenter)
        self.tone_researcher.setObjectName("SubtitleLabel")

        self.tone_blurb = QLabel(
            "Scientific cameras and microscopes can capture tens of thousands of pieces of "
            "information at every pixel. However, most monitor only have capacity for 256 pieces "
            "of information. This project is creating new methods to enable all the relevant "
            "information to be displayed at one time without prior knowledge of the contents of the image."
        )
        self.tone_blurb.setAlignment(Qt.AlignCenter)
        self.tone_blurb.setWordWrap(True)
        self.tone_blurb.setObjectName("SubtitleLabel")
        self.tone_blurb.setMaximumWidth(900)

        self.tone_collab = QLabel("In collaboration with")
        self.tone_collab.setAlignment(Qt.AlignCenter)
        self.tone_collab.setObjectName("SubtitleLabel")

        bad_path, good_path = self.tone_pairs[self.tone_index]
        self.tone_widget = ToneMappingDemoWidget(bad_path, good_path)
        self.tone_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.tone_back = QPushButton("Menu")
        self.tone_prev = QPushButton("Previous")
        self.tone_next = QPushButton("Next")

        t = QVBoxLayout()
        t.setContentsMargins(0, 120, 0, 180)
        t.setSpacing(18)
        t.addWidget(self.tone_label, alignment=Qt.AlignCenter)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        btn_row.addStretch(1)
        btn_row.addWidget(self.tone_back)
        btn_row.addWidget(self.tone_prev)
        btn_row.addWidget(self.tone_next)
        btn_row.addStretch(1)
        t.addLayout(btn_row)
        t.addWidget(self.tone_widget, stretch=5)
        t.addSpacing(24)

        # Bottom band: images + descriptive copy
        bottom_info = QWidget()
        bottom_layout = QVBoxLayout(bottom_info)
        bottom_layout.setContentsMargins(40, 10, 40, 0)
        bottom_layout.setSpacing(14)
        bottom_info.setMaximumWidth(1100)

        image_row = QHBoxLayout()
        image_row.setContentsMargins(0, 0, 0, 0)
        image_row.setSpacing(32)
        image_row.addStretch(1)
        image_row.addWidget(self.tone_headshot)
        image_row.addWidget(self.tone_logo)
        image_row.addStretch(1)

        bottom_layout.addLayout(image_row)
        bottom_layout.addWidget(self.tone_researcher, alignment=Qt.AlignCenter)
        bottom_layout.addWidget(self.tone_blurb, alignment=Qt.AlignCenter)
        # bottom_layout.addWidget(self.tone_collab, alignment=Qt.AlignCenter)

        t.addWidget(bottom_info, stretch=3, alignment=Qt.AlignCenter)
        self.tone_page.setLayout(t)

        # Stack pages
        self.stack.addWidget(self.start_page)
        self.stack.addWidget(self.menu_page)
        self.stack.addWidget(self.seg_page)
        self.stack.addWidget(self.tone_page)
        self.stack.setCurrentWidget(self.start_page)

        # Overlay
        self.overlay = CursorOverlay(self)
        self.overlay.setObjectName("OverlayWidget")
        self.overlay.setGeometry(self.rect())
        self.overlay.raise_()

        # Vision thread
        self.worker = VisionThread()
        self.worker.camera_size.connect(self.on_camera_size)
        self.worker.pose.connect(self.on_pose)
        self.worker.start()

        # Overlay refresh timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.overlay.update)
        self.timer.start(33)

        # Buttons wiring
        self.go_button.clicked.connect(self.show_menu)
        self.btn_back.clicked.connect(self.show_start)
        self.btn_seg.clicked.connect(self.show_seg)
        self.btn_tone.clicked.connect(self.show_tone)
        self.seg_menu_btn.clicked.connect(self.show_menu)
        self.seg_prev_btn.clicked.connect(self.prev_seg_example)
        self.seg_next_btn.clicked.connect(self.next_seg_example)
        self.tone_back.clicked.connect(self.show_menu)
        self.tone_prev.clicked.connect(self.show_prev_tone)
        self.tone_next.clicked.connect(self.show_next_tone)

        self.cursor_px = QPoint(100, 100)

        self._init_style()

    # ---- Styling ----

    def _init_style(self):
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setStyleSheet("""
            QWidget {
                background-color: #020617;
                color: #e5e7eb;
                font-family: "Segoe UI", "Noto Sans", system-ui, -apple-system, sans-serif;
                font-size: 15px;
            }

            QWidget#OverlayWidget {
                background: transparent;
            }

            QLabel#TitleLabel {
                font-size: 26px;
                font-weight: 600;
                letter-spacing: 0.5px;
            }

            QLabel#SubtitleLabel {
                font-size: 18px;
                color: #9ca3af;
            }

            QPushButton {
                background-color: #111827;
                color: #e5e7eb;
                padding: 14px 28px;
                border-radius: 28px;
                border: 1px solid #374151;
                font-size: 17px;
            }

            QPushButton:hover {
                background-color: #1f2937;
            }

            QPushButton:pressed {
                background-color: #020617;
                border-color: #4b5563;
            }

            QPushButton#PrimaryButton {
                background-color: #2563eb;
                border-color: #2563eb;
                color: #f9fafb;
            }

            QPushButton#PrimaryButton:hover {
                background-color: #1d4ed8;
            }

            QPushButton#PrimaryButton:pressed {
                background-color: #1e40af;
            }

            QPushButton#SecondaryButton {
                background-color: #020617;
                border-color: #374151;
                color: #e5e7eb;
            }

            QPushButton#SecondaryButton:hover {
                background-color: #020617;
                border-color: #4b5563;
            }

            QPushButton#SecondaryButton:pressed {
                background-color: #111827;
            }
        """)

    # ---- Layout events ----

    def on_camera_size(self, w, h):
        self.overlay.set_res_label(f"{w} x {h}")
        self.overlay.setGeometry(self.rect())
        self.overlay.raise_()

    def resizeEvent(self, e):
        self.overlay.setGeometry(self.rect())
        self.overlay.raise_()
        super().resizeEvent(e)

    # ---- Navigation ----

    def show_start(self):
        self.stack.setCurrentWidget(self.start_page)
        self.go_button.setVisible(False)
        self.start_label.setText("Wave your hand to begin.")
        self.overlay.raise_()

    def show_menu(self):
        self.stack.setCurrentWidget(self.menu_page)
        self.overlay.raise_()

    def show_seg(self):
        self.stack.setCurrentWidget(self.seg_page)
        self.overlay.raise_()

    def show_tone(self):
        self.stack.setCurrentWidget(self.tone_page)
        self.overlay.raise_()

    def _set_tone_pair(self, idx):
        if not self.tone_pairs:
            return
        self.tone_index = idx % len(self.tone_pairs)
        bad_path, good_path = self.tone_pairs[self.tone_index]
        self.tone_widget.set_images(bad_path, good_path)

    def show_prev_tone(self):
        if not self.tone_pairs:
            return
        self._set_tone_pair(self.tone_index - 1)

    def show_next_tone(self):
        if not self.tone_pairs:
            return
        self._set_tone_pair(self.tone_index + 1)

    def _load_rnvu_masks(self):
        base_dir = Path("data/rnvu_data")
        if not base_dir.exists():
            print("RNvU directory not found at data/rnvu_data")
            return []
        return sorted(
            str(p) for p in base_dir.glob("*.tif")
            if p.name.lower() != "sarah_050.tif"
        )

    def _current_seg_widget(self):
        if not getattr(self, "seg_examples", None):
            return None
        return self.seg_examples[self.seg_index]["widget"]

    def _apply_seg_example(self):
        if not getattr(self, "seg_examples", None):
            return

        self.seg_index = self.seg_index % len(self.seg_examples)
        ex = self.seg_examples[self.seg_index]

        self.seg_label.setText(ex["title"])
        self.seg_desc.setText(ex["desc"])
        self.researcher_label.setText(ex["researcher"])
        self.seg_widget_stack.setCurrentIndex(self.seg_index)

        headshot = ex.get("headshot")
        if headshot:
            pix = QPixmap(headshot)
            if not pix.isNull():
                self.headshot_label.setPixmap(
                    pix.scaledToHeight(200, Qt.SmoothTransformation)
                )
                self.headshot_label.setVisible(True)
            else:
                self.headshot_label.clear()
                self.headshot_label.setVisible(False)
        else:
            self.headshot_label.clear()
            self.headshot_label.setVisible(False)

    def prev_seg_example(self):
        if not getattr(self, "seg_examples", None):
            return
        self.seg_index = (self.seg_index - 1) % len(self.seg_examples)
        self._apply_seg_example()

    def next_seg_example(self):
        if not getattr(self, "seg_examples", None):
            return
        self.seg_index = (self.seg_index + 1) % len(self.seg_examples)
        self._apply_seg_example()

    # ---- Mapping + Click ----

    def _map_uv(self, u, v):
        u = min(max(u, 0.0), 1.0)
        v = min(max(v, 0.0), 1.0)
        return QPoint(int(u * (self.width() - 1)), int(v * (self.height() - 1)))

    def _synth_click(self, p):
        global_pos = self.mapToGlobal(p)
        target = QApplication.widgetAt(global_pos)
        if not target:
            return
        local = target.mapFromGlobal(global_pos)
        press = QMouseEvent(
            QMouseEvent.MouseButtonPress, local,
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier
        )
        release = QMouseEvent(
            QMouseEvent.MouseButtonRelease, local,
            Qt.LeftButton, Qt.LeftButton, Qt.NoModifier
        )
        QApplication.sendEvent(target, press)
        QApplication.sendEvent(target, release)

    # ---- Pose updates ----

    def on_pose(self, u, v, hand_present, lm_uv, click):
        now = time.time()
        if hand_present:
            self.last_hand = now

        if MIRROR_X:
            u = 1.0 - u
            if lm_uv:
                lm_uv = [(1.0 - x, y) for (x, y) in lm_uv]

        if hand_present:
            target = self._map_uv(u, v)
            alpha = 0.35
            cur = self.cursor_px
            self.cursor_px = QPoint(
                int(alpha * target.x() + (1 - alpha) * cur.x()),
                int(alpha * target.y() + (1 - alpha) * cur.y())
            )

        lm_pts = None
        if hand_present and lm_uv:
            lm_pts = [self._map_uv(x, y) for (x, y) in lm_uv]

        if hand_present and self.stack.currentWidget() is self.start_page:
            if not self.go_button.isVisible():
                self.go_button.setVisible(True)
                self.start_label.setText("Hand detected. Make a fist to select.")

        if now - self.last_hand > HAND_TIMEOUT_S:
            if self.stack.currentWidget() is not self.start_page:
                self.show_start()

        # Tone mapping spotlight
        if (
            hand_present and lm_uv and
            self.stack.currentWidget() is self.tone_page and
            self.tone_widget.isVisible()
        ):
            pw = palm_width_uv(lm_uv)
            base = max(1, min(self.tone_widget.width(), self.tone_widget.height()))
            radius = max(10.0, pw * base * HAND_RADIUS_SCALE)

            global_cursor = self.mapToGlobal(self.cursor_px)
            local = self.tone_widget.mapFromGlobal(global_cursor)
            self.tone_widget.set_spot(local, radius)

        # Segmentation hover
        if (
            hand_present and
            self.stack.currentWidget() is self.seg_page
            and click
        ):
            seg_widget = self._current_seg_widget()
            if seg_widget and seg_widget.isVisible():
                global_cursor = self.mapToGlobal(self.cursor_px)
                local = seg_widget.mapFromGlobal(global_cursor)
                if hasattr(seg_widget, "select_region"):
                    seg_widget.select_region(local.x(), local.y())
                elif hasattr(seg_widget, "select_mask"):
                    seg_widget.select_mask(local.x(), local.y())

        if click:
            self._synth_click(self.cursor_px)

        self.overlay.set_cursor(self.cursor_px, hand_present)
        self.overlay.set_landmarks(lm_pts)

        if self.stack.currentWidget() is self.tone_page:
            self.tone_widget.update()

    def closeEvent(self, e):
        self.worker.stop()
        super().closeEvent(e)

# ------------ Main ------------

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    if FULLSCREEN:
        w.showFullScreen()
    else:
        w.show()
    sys.exit(app.exec())
