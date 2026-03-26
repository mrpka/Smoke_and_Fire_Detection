"""
Fire & Smoke Detection — Tkinter Desktop GUI
Run:  python gui.py
Deps: pip install opencv-python pillow numpy
Optional deep-learning layer: pip install ultralytics
"""

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import threading
import queue
import time
import sqlite3
import os
import numpy as np
from datetime import datetime
from PIL import Image, ImageTk

# ─── Optional YOLO layer ─────────────────────────────────────────
try:
    from ultralytics import YOLO
    _YOLO_PATH = "models/fire_smoke_yolov8n.pt"
    yolo_model = YOLO(_YOLO_PATH) if os.path.exists(_YOLO_PATH) else None
except ImportError:
    yolo_model = None

# ─── Theme ───────────────────────────────────────────────────────
DARK_BG      = "#0f1117"
PANEL_BG     = "#181c24"
CARD_BG      = "#1e2330"
BORDER       = "#2a3040"
ACCENT_RED   = "#e63946"
ACCENT_AMBER = "#f4a261"
ACCENT_GREEN = "#52b788"
TEXT_PRIMARY = "#e8eaf0"
TEXT_MUTED   = "#6b7280"
TEXT_DIM     = "#3d4455"
FONT_MONO    = ("Courier New", 10)
FONT_LABEL   = ("Segoe UI", 9)
FONT_BOLD    = ("Segoe UI", 9, "bold")
FONT_TITLE   = ("Segoe UI", 11, "bold")

# ─── Database ────────────────────────────────────────────────────
DB_PATH = "detections.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            ts         TEXT NOT NULL,
            camera     TEXT NOT NULL,
            label      TEXT NOT NULL,
            confidence REAL NOT NULL,
            snapshot   TEXT
        )
    """)
    conn.commit(); conn.close()

def log_event(camera_id, label, confidence, snapshot_path=""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO events (ts,camera,label,confidence,snapshot) VALUES (?,?,?,?,?)",
        (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), camera_id, label, confidence, snapshot_path)
    )
    conn.commit(); conn.close()

def fetch_events(limit=200):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT ts,camera,label,confidence,snapshot FROM events ORDER BY id DESC LIMIT ?",
        (limit,)
    ).fetchall()
    conn.close(); return rows

def count_events():
    conn = sqlite3.connect(DB_PATH)
    n = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    conn.close(); return n

def clear_all_data():
    """Delete all rows from the events table and remove saved snapshot files."""
    conn = sqlite3.connect(DB_PATH)
    snapshots = [r[0] for r in conn.execute(
        "SELECT snapshot FROM events WHERE snapshot != ''").fetchall()]
    conn.execute("DELETE FROM events")
    conn.commit(); conn.close()
    for path in snapshots:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


# ─────────────────────────────────────────────────────────────────
#  DETECTION ENGINE  (background thread)
# ─────────────────────────────────────────────────────────────────
class DetectionEngine:
    ALERT_COOLDOWN       = 30   # seconds between DB writes
    REQUIRED_CONSECUTIVE = 8    # FIX 5: raised from 4 → 8 frames before alert fires

    def __init__(self, camera_index=0):
        self.camera_index    = camera_index
        self.result_queue    = queue.Queue(maxsize=4)
        self._running        = False
        self._thread         = None
        self.bg_sub          = cv2.createBackgroundSubtractorMOG2(
                                   history=500, varThreshold=60,  # FIX 3: raised threshold
                                   detectShadows=False)
        self._consecutive    = 0
        self._last_alert_ts  = 0
        self._fire_history   = []   # FIX 4: for flicker/variance check
        self._warmed_up      = False
        os.makedirs("snapshots", exist_ok=True)

    def start(self):
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    # ─── main loop ───────────────────────────────────────────────
    def _loop(self):
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        if not cap.isOpened():
            self.result_queue.put({"error": f"Cannot open camera {self.camera_index}"})
            return

        # FIX 3: warm up background subtractor for 40 frames before enabling detection
        # This prevents static colored backgrounds from triggering on startup
        for _ in range(40):
            ret, frame = cap.read()
            if ret:
                self.bg_sub.apply(cv2.resize(frame, (640, 480)))
        self._warmed_up = True

        n = 0
        while self._running:
            ret, frame = cap.read()
            if not ret:
                self.result_queue.put({"error": "Frame read failed — check camera connection."})
                break
            n += 1
            if n % 2 != 0:
                continue
            try:
                self.result_queue.put_nowait(self._process(frame))
            except queue.Full:
                pass
        cap.release()

    # ─── per-frame processing ────────────────────────────────────
    def _process(self, frame):
        resized = cv2.resize(frame, (640, 480))
        kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg      = cv2.morphologyEx(self.bg_sub.apply(resized), cv2.MORPH_OPEN, kernel)
        fg      = cv2.dilate(fg, kernel, iterations=2)
        hsv     = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        # Return a blank result until background model is ready
        if not self._warmed_up:
            return dict(frame=resized, fire=False, smoke=False,
                        confirmed=False, confidence=0.0,
                        fire_pixels=0, smoke_pixels=0, snapshot="")

        f_mask, f_px = self._fire_mask(hsv, fg)
        s_mask, s_px = self._smoke_mask(hsv, fg)
        f_cnts       = self._contours(f_mask, 800)   # FIX 2: min_area raised 500 → 800
        s_cnts       = self._contours(s_mask, 1500)  # FIX 2: min_area raised 1000 → 1500
        yolo_dets    = self._yolo(resized)

        # FIX 4: flicker/variance check — real fire fluctuates, static bg does not
        self._fire_history.append(f_px)
        if len(self._fire_history) > 12:
            self._fire_history.pop(0)
        fire_variance = np.var(self._fire_history) if len(self._fire_history) > 5 else 0

        # FIX 2: pixel thresholds raised (5000/8000 instead of 2000/4000)
        # FIX 4: fire also requires variance > 500 (flickering), static red bg = ~0 variance
        fire_det  = (f_px > 5000 and bool(f_cnts) and fire_variance > 500) or \
                    any(d["class"] == "fire"  for d in yolo_dets)
        smoke_det = (s_px > 8000 and bool(s_cnts)) or \
                    any(d["class"] == "smoke" for d in yolo_dets)

        conf = max((d["confidence"] for d in yolo_dets), default=0.0)
        if fire_det  and conf == 0.0: conf = min(0.55 + f_px/40000, 0.85)
        if smoke_det and conf == 0.0: conf = min(0.45 + s_px/80000, 0.75)

        if fire_det or smoke_det:
            self._consecutive += 1
        else:
            self._consecutive = max(0, self._consecutive - 1)
        confirmed = self._consecutive >= self.REQUIRED_CONSECUTIVE

        annotated = self._annotate(resized.copy(), f_cnts, s_cnts, yolo_dets, conf, fire_det, smoke_det, confirmed)

        snapshot = ""
        if confirmed and (time.time() - self._last_alert_ts) > self.ALERT_COOLDOWN:
            snapshot = f"snapshots/{int(time.time())}.jpg"
            cv2.imwrite(snapshot, annotated)
            log_event(str(self.camera_index),
                      "FIRE" if fire_det else "SMOKE", conf, snapshot)
            self._last_alert_ts = time.time()
            self._consecutive   = 0

        return dict(frame=annotated, fire=fire_det, smoke=smoke_det,
                    confirmed=confirmed, confidence=conf,
                    fire_pixels=f_px, smoke_pixels=s_px, snapshot=snapshot)

    # ─── annotation ──────────────────────────────────────────────
    def _annotate(self, img, f_cnts, s_cnts, yolo_dets, conf, fire, smoke, confirmed):
        for x,y,w,h,_ in f_cnts:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,50,255),2)
            cv2.putText(img,f"FIRE {conf:.0%}",(x,max(y-6,12)),
                        cv2.FONT_HERSHEY_SIMPLEX,.55,(0,50,255),2)
        for x,y,w,h,_ in s_cnts:
            cv2.rectangle(img,(x,y),(x+w,y+h),(180,180,180),2)
            cv2.putText(img,"SMOKE",(x,max(y-6,12)),
                        cv2.FONT_HERSHEY_SIMPLEX,.55,(200,200,200),2)
        for d in yolo_dets:
            x,y,w,h = d["bbox"]
            c = (0,50,255) if d["class"]=="fire" else (180,180,180)
            cv2.rectangle(img,(x,y),(x+w,y+h),c,2)

        # top status bar
        cv2.rectangle(img,(0,0),(640,30),(0,0,0),-1)
        if confirmed:
            label  = "!!! FIRE DETECTED !!!" if fire else "!!! SMOKE DETECTED !!!"
            colour = (0,50,255)
        else:
            label  = "CLEAR"
            colour = (60,200,110)
        cv2.putText(img, label,(8,21),cv2.FONT_HERSHEY_SIMPLEX,.65,colour,2)
        cv2.putText(img, datetime.now().strftime("%H:%M:%S"),
                    (560,21),cv2.FONT_HERSHEY_SIMPLEX,.48,(100,100,100),1)
        return img

    # ─── helpers ─────────────────────────────────────────────────
    def _fire_mask(self, hsv, fg):
        # FIX 1: tighter HSV — high S (>180) and high V (>180) kills dark maroon and pale orange
        # Narrower hue range (0-25) excludes more orange-red false positives
        m = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,   180, 180]), np.array([25,  255, 255])),
            cv2.inRange(hsv, np.array([165, 180, 180]), np.array([180, 255, 255]))
        )
        c = cv2.bitwise_and(m, fg)
        # FIX 3: motion overlap check — reject if <30% of fire pixels are in moving regions
        total  = cv2.countNonZero(m)
        moving = cv2.countNonZero(c)
        if total > 0 and (moving / total) < 0.3:
            return np.zeros_like(fg), 0
        return c, moving

    def _smoke_mask(self, hsv, fg):
        # FIX 1: S cap lowered 60->40 (real smoke is very desaturated)
        # V capped at 220 (not 255) to exclude pure white walls/backgrounds
        m = cv2.inRange(hsv, np.array([0, 0, 160]), np.array([180, 40, 220]))
        c = cv2.bitwise_and(m, fg)
        return c, cv2.countNonZero(c)

    def _contours(self, mask, min_area):
        cnts,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid  = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area: continue
            x,y,w,h = cv2.boundingRect(c)
            ar      = w / float(h)
            ha      = cv2.contourArea(cv2.convexHull(c))
            sol     = area/ha if ha > 0 else 0
            if 0.2 < ar < 5.0 and sol > 0.3:
                valid.append((x,y,w,h,area))
        return valid

    def _yolo(self, frame):
        if yolo_model is None: return []
        res  = yolo_model(frame, conf=0.5, verbose=False)
        dets = []
        for r in res:
            for box in r.boxes:
                x1,y1,x2,y2 = map(int, box.xyxy[0])
                dets.append(dict(
                    **{"class": yolo_model.names[int(box.cls)]},
                    confidence=float(box.conf),
                    bbox=(x1,y1,x2-x1,y2-y1)
                ))
        return dets


# ─────────────────────────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        init_db()
        self.title("Fire & Smoke Detection")
        self.geometry("1200x760")
        self.minsize(900, 600)
        self.configure(bg=DARK_BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self._engine:  DetectionEngine | None = None
        self._running  = False
        self._canvas_img = None

        self._build_ui()
        self._scan_cameras()
        self._poll()

    # ─────────────────── BUILD UI ────────────────────────────────
    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────
        bar = tk.Frame(self, bg=PANEL_BG, height=52)
        bar.pack(fill=tk.X)
        bar.pack_propagate(False)

        tk.Label(bar, text="🔥", bg=PANEL_BG, fg=ACCENT_RED,
                 font=("Segoe UI Emoji", 18)).pack(side=tk.LEFT, padx=(14,4), pady=8)
        tk.Label(bar, text="Fire & Smoke Detection",
                 bg=PANEL_BG, fg=TEXT_PRIMARY,
                 font=("Segoe UI",13,"bold")).pack(side=tk.LEFT)

        self._badge = tk.Label(bar, text="  IDLE  ",
                               bg=CARD_BG, fg=TEXT_MUTED,
                               font=FONT_BOLD, padx=8, pady=3)
        self._badge.pack(side=tk.LEFT, padx=18, pady=12)

        yolo_lbl = "OpenCV + YOLO" if yolo_model else "OpenCV (color mode)"
        tk.Label(bar, text=f"v1.0  |  {yolo_lbl}",
                 bg=PANEL_BG, fg=TEXT_DIM, font=FONT_LABEL).pack(side=tk.RIGHT, padx=16)

        tk.Frame(self, bg=BORDER, height=1).pack(fill=tk.X)

        # ── Main split ───────────────────────────────────────────
        body = tk.Frame(self, bg=DARK_BG)
        body.pack(fill=tk.BOTH, expand=True)

        # LEFT: feed + controls
        left = tk.Frame(body, bg=DARK_BG)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(14,7), pady=14)

        feed_wrap = tk.Frame(left, bg=CARD_BG,
                             highlightbackground=BORDER, highlightthickness=1)
        feed_wrap.pack(fill=tk.BOTH, expand=True)

        self._canvas = tk.Canvas(feed_wrap, bg="#080808",
                                 bd=0, highlightthickness=0, cursor="crosshair")
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self._placeholder()

        # controls bar
        ctrl = tk.Frame(left, bg=DARK_BG)
        ctrl.pack(fill=tk.X, pady=(8,0))

        tk.Label(ctrl, text="Camera", bg=DARK_BG,
                 fg=TEXT_MUTED, font=FONT_LABEL).pack(side=tk.LEFT)

        self._cam_var   = tk.StringVar(value="0 – Default")
        self._cam_combo = ttk.Combobox(ctrl, textvariable=self._cam_var,
                                       width=22, state="readonly", font=FONT_LABEL)
        self._cam_combo.pack(side=tk.LEFT, padx=(6,14))

        self._btn_start = tk.Button(ctrl, text="▶  Start Detection",
                                    bg=ACCENT_GREEN, fg="#0d1a12",
                                    font=FONT_BOLD, relief="flat",
                                    cursor="hand2", padx=14, pady=5,
                                    command=self._start)
        self._btn_start.pack(side=tk.LEFT, padx=(0,6))

        self._btn_stop = tk.Button(ctrl, text="■  Stop",
                                   bg=CARD_BG, fg=TEXT_MUTED,
                                   font=FONT_BOLD, relief="flat",
                                   cursor="hand2", padx=14, pady=5,
                                   state=tk.DISABLED, command=self._stop)
        self._btn_stop.pack(side=tk.LEFT)

        # RIGHT panel
        right = tk.Frame(body, bg=DARK_BG, width=318)
        right.pack(side=tk.RIGHT, fill=tk.Y, padx=(7,14), pady=14)
        right.pack_propagate(False)

        # stat cards grid
        grid = tk.Frame(right, bg=DARK_BG)
        grid.pack(fill=tk.X)
        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

        self._s_fire  = self._stat(grid, "Fire pixels",  "—", ACCENT_RED,   0)
        self._s_smoke = self._stat(grid, "Smoke pixels", "—", ACCENT_AMBER, 1)
        self._s_conf  = self._stat(grid, "Confidence",   "—", ACCENT_GREEN, 2)
        self._s_total = self._stat(grid, "Total alerts", str(count_events()), TEXT_MUTED, 3)

        tk.Frame(right, bg=BORDER, height=1).pack(fill=tk.X, pady=(12,0))

        # log header
        lh = tk.Frame(right, bg=DARK_BG)
        lh.pack(fill=tk.X, pady=(8,6))
        tk.Label(lh, text="Alert History",
                 bg=DARK_BG, fg=TEXT_PRIMARY, font=FONT_TITLE).pack(side=tk.LEFT)
        tk.Button(lh, text="Refresh", bg=CARD_BG, fg=TEXT_MUTED,
                  font=FONT_LABEL, relief="flat", cursor="hand2",
                  padx=6, pady=2, command=self._refresh_log).pack(side=tk.RIGHT)
        tk.Button(lh, text="🗑  Clear All", bg=CARD_BG, fg=ACCENT_RED,
                  font=FONT_LABEL, relief="flat", cursor="hand2",
                  padx=6, pady=2, command=self._clear_data).pack(side=tk.RIGHT, padx=(0, 6))

        # treeview
        s = ttk.Style(self)
        s.theme_use("clam")
        s.configure("D.Treeview",
                    background=CARD_BG, foreground=TEXT_PRIMARY,
                    fieldbackground=CARD_BG, rowheight=24,
                    font=FONT_MONO, borderwidth=0)
        s.configure("D.Treeview.Heading",
                    background=PANEL_BG, foreground=TEXT_MUTED,
                    font=FONT_LABEL, borderwidth=0)
        s.map("D.Treeview", background=[("selected", BORDER)])

        tw = tk.Frame(right, bg=CARD_BG,
                      highlightbackground=BORDER, highlightthickness=1)
        tw.pack(fill=tk.BOTH, expand=True)

        cols = ("Time","Cam","Label","Conf")
        self._tree = ttk.Treeview(tw, columns=cols, show="headings",
                                  style="D.Treeview", selectmode="browse")
        for col, w in zip(cols, (90, 45, 65, 50)):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, anchor="center" if col!="Time" else "w")

        vsb = ttk.Scrollbar(tw, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        self._tree.pack(fill=tk.BOTH, expand=True)
        self._tree.tag_configure("FIRE",  foreground=ACCENT_RED)
        self._tree.tag_configure("SMOKE", foreground=ACCENT_AMBER)

        self._refresh_log()

    def _stat(self, parent, label, value, color, idx):
        r, c = divmod(idx, 2)
        card = tk.Frame(parent, bg=CARD_BG,
                        highlightbackground=BORDER, highlightthickness=1)
        card.grid(row=r, column=c, padx=(0,4) if c==0 else (4,0),
                  pady=(0,4), sticky="nsew", ipadx=8, ipady=6)
        tk.Label(card, text=label, bg=CARD_BG,
                 fg=TEXT_MUTED, font=FONT_LABEL).pack(anchor="w", padx=10, pady=(8,2))
        lbl = tk.Label(card, text=value, bg=CARD_BG,
                       fg=color, font=("Segoe UI",18,"bold"))
        lbl.pack(anchor="w", padx=10, pady=(0,8))
        return lbl

    def _placeholder(self):
        self._canvas.update_idletasks()
        w = self._canvas.winfo_width()  or 640
        h = self._canvas.winfo_height() or 420
        self._canvas.delete("all")
        self._canvas.create_rectangle(0, 0, w, h, fill="#080808", outline="")
        self._canvas.create_text(w//2, h//2,
                                  text="No camera feed  —  press Start Detection",
                                  fill=TEXT_DIM, font=("Segoe UI",12))

    # ─────────────────── CAMERA SCAN ─────────────────────────────
    def _scan_cameras(self):
        cams = []
        for i in range(6):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cams.append(f"{i} – {'Default' if i==0 else f'Camera {i}'}")
                cap.release()
        self._cam_combo["values"] = cams or ["0 – Default"]
        self._cam_combo.current(0)

    # ─────────────────── CONTROLS ────────────────────────────────
    def _start(self):
        idx = int(self._cam_var.get().split(" ")[0])
        self._engine  = DetectionEngine(camera_index=idx)
        self._engine.start()
        self._running = True
        self._btn_start.config(state=tk.DISABLED)
        self._btn_stop.config(state=tk.NORMAL, bg=ACCENT_RED, fg="white")
        self._cam_combo.config(state=tk.DISABLED)
        self._badge.config(text="  RUNNING  ", fg=ACCENT_GREEN)

    def _stop(self):
        if self._engine:
            self._engine.stop()
            self._engine = None
        self._running = False
        self._btn_start.config(state=tk.NORMAL)
        self._btn_stop.config(state=tk.DISABLED, bg=CARD_BG, fg=TEXT_MUTED)
        self._cam_combo.config(state="readonly")
        self._badge.config(text="  IDLE  ", fg=TEXT_MUTED)
        self._placeholder()
        self._refresh_log()

    # ─────────────────── POLLING ─────────────────────────────────
    def _poll(self):
        if self._running and self._engine:
            try:
                while True:
                    data = self._engine.result_queue.get_nowait()
                    if "error" in data:
                        messagebox.showerror("Camera Error", data["error"])
                        self._stop(); break
                    self._update(data)
            except queue.Empty:
                pass
        self.after(30, self._poll)

    def _update(self, d):
        # frame
        rgb   = cv2.cvtColor(d["frame"], cv2.COLOR_BGR2RGB)
        cw    = self._canvas.winfo_width()  or 640
        ch    = self._canvas.winfo_height() or 480
        img   = Image.fromarray(rgb)
        img.thumbnail((cw, ch), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)
        self._canvas.delete("all")
        self._canvas.create_image(cw//2, ch//2, anchor="center", image=photo)
        self._canvas_img = photo

        # alert overlay on canvas
        if d["confirmed"]:
            lbl = "🔥  FIRE DETECTED" if d["fire"] else "💨  SMOKE DETECTED"
            self._canvas.create_rectangle(0, 0, cw, 46, fill="#200000", outline="")
            self._canvas.create_text(cw//2, 23, text=lbl,
                                      fill=ACCENT_RED, font=("Segoe UI",13,"bold"))
            self._badge.config(text="  ⚠ ALERT  ", fg=ACCENT_RED)
            if d.get("snapshot"):
                self._refresh_log()
        else:
            self._badge.config(text="  RUNNING  ", fg=ACCENT_GREEN)

        # stats
        self._s_fire.config( text=f"{d['fire_pixels']:,}")
        self._s_smoke.config(text=f"{d['smoke_pixels']:,}")
        self._s_conf.config( text=f"{d['confidence']:.0%}")
        self._s_total.config(text=str(count_events()))

    # ─────────────────── LOG ─────────────────────────────────────
    def _refresh_log(self):
        for row in self._tree.get_children():
            self._tree.delete(row)
        for ts, cam, label, conf, _ in fetch_events(200):
            self._tree.insert("", 0,
                values=(ts[11:19], cam, label, f"{conf:.0%}"),
                tags=(label.upper(),)
            )

    def _clear_data(self):
        count = count_events()
        if count == 0:
            messagebox.showinfo("Clear Data", "There is no saved data to clear.")
            return
        confirmed = messagebox.askyesno(
            "Clear All Data",
            f"This will permanently delete {count} alert record{'s' if count != 1 else ''} "
            f"and all saved snapshots.\n\nThis cannot be undone. Continue?",
            icon="warning"
        )
        if confirmed:
            clear_all_data()
            self._refresh_log()
            self._s_total.config(text="0")

    # ─────────────────── CLOSE ───────────────────────────────────
    def _on_close(self):
        if self._engine: self._engine.stop()
        self.destroy()


if __name__ == "__main__":
    app = App()
    app.mainloop()
