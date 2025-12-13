# app.py — NEURODOKU (Final: 6x6 & 9x9, CNN-only OCR, Notebook background)
# Overwrite existing file with this. Requirements: streamlit, numpy, pillow,
# opencv-python-headless, tensorflow>=2.11, scikit-learn

import streamlit as st
from PIL import Image, ImageDraw
import numpy as np
import cv2, os, io, time, math, re, base64
import tensorflow as tf
from tensorflow.keras import layers, models

# ---------------- Page setup ----------------
st.set_page_config(page_title="NEURODOKU", layout="centered")
MODEL_PATH = "digit_model.h5"  # put trained model here (recommended)

# ---------------- Logo helper (make circular + embed as data URI) ----------------
def make_logo_datauri(path="logo.png", size_px=160):
    if not os.path.exists(path):
        return None
    im = Image.open(path).convert("RGBA")
    w,h = im.size
    s = min(w,h)
    im = im.crop(((w-s)//2, (h-s)//2, (w+s)//2, (h+s)//2)).resize((size_px,size_px), Image.LANCZOS)
    data = np.array(im)
    r,g,b,a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    white_mask = (r>240)&(g>240)&(b>240)
    data[white_mask,3] = 0
    im = Image.fromarray(data)
    mask = Image.new("L", (size_px,size_px), 0)
    draw = ImageDraw.Draw(mask); draw.ellipse((0,0,size_px,size_px), fill=255)
    im.putalpha(mask)
    buf = io.BytesIO(); im.save(buf, format="PNG"); buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

logo_data = make_logo_datauri("logo.png", 160)

# ---------------- Notebook CSS (Theme A) ----------------
st.markdown("""
<style>
html, body { background:#fbf8ef; color:#0b2540; font-family:'Segoe UI', Roboto, Arial, sans-serif; }
body { background-image: linear-gradient(rgba(0,0,0,0.03) 1px, transparent 1px); background-size: 100% 36px; }
.header { text-align:center; margin-top:12px; margin-bottom:6px; }
.card { background: rgba(255,255,255,0.92); padding:18px; border-radius:12px; width:92%; margin:auto;
       box-shadow:0 8px 30px rgba(10,30,50,0.06); border:1px solid rgba(10,30,50,0.03); }
.title { font-weight:900; font-size:36px; text-align:center; color:#083048; margin-bottom:6px; }
.small-note { color:#264653; text-align:center; margin-top:6px; font-size:13px; }
.stButton>button { font-weight:700; }
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("<div class='header'>", unsafe_allow_html=True)
if logo_data:
    st.markdown(f"<img src='{logo_data}' style='height:160px;width:160px;border-radius:50%;display:block;margin-left:auto;margin-right:auto;'/>", unsafe_allow_html=True)
st.markdown("<div class='title'>NEURODOKU</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

# ---------------- Image processing utilities ----------------
def find_largest_quad(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 40, 150)
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts[:12]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            return approx.reshape(4,2)
    return None

def order_pts(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1); rect[0]=pts[np.argmin(s)]; rect[2]=pts[np.argmax(s)]
    d = np.diff(pts, axis=1); rect[1]=pts[np.argmin(d)]; rect[3]=pts[np.argmax(d)]
    return rect

def warp_to_square(img, pts, min_side=480):
    rect = order_pts(pts)
    (tl,tr,br,bl) = rect
    widthA = np.linalg.norm(br-bl); widthB = np.linalg.norm(tr-tl)
    heightA = np.linalg.norm(tr-br); heightB = np.linalg.norm(tl-bl)
    side = int(max(widthA,widthB,heightA,heightB))
    if side < min_side: side = min_side
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (side, side))

# Normalize color to better separate digits
def normalize_color(bg_img):
    lab = cv2.cvtColor(bg_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    rgb = cv2.cvtColor(cv2.merge((l2,a,b)), cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    return gray

# Improved grid line removal and denoising
def remove_background_and_lines(gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(gray)
    th = cv2.adaptiveThreshold(l,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    h,w = th.shape
    vertical_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, h//30)))
    horizontal_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, w//30), 1))
    vertical = cv2.morphologyEx(th, cv2.MORPH_OPEN, vertical_k, iterations=1)
    horizontal = cv2.morphologyEx(th, cv2.MORPH_OPEN, horizontal_k, iterations=1)
    lines = cv2.bitwise_or(vertical, horizontal)
    cleaned = cv2.subtract(th, lines)
    cleaned = cv2.medianBlur(cleaned, 3)
    small_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, small_k, iterations=1)
    erode_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    cleaned = cv2.erode(cleaned, erode_k, iterations=1)
    return cleaned  # digits white on black

# connected components -> candidates mapping
def extract_candidates_from_binary(bin_img, n):
    h,w = bin_img.shape
    cell = h // n
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img, connectivity=8)
    candidates = {}
    for lab in range(1, num_labels):
        x,y,ww,hh,area = stats[lab]
        cx,cy = centroids[lab]
        if area < max(40, (cell*cell)//150): continue
        if area > (cell*cell*0.6): continue
        ar = ww / float(max(1, hh))
        if ar > 3.0 or ar < 0.25: continue
        i = int(cy // cell); j = int(cx // cell)
        if i<0 or i>=n or j<0 or j>=n: continue
        candidates.setdefault((i,j), []).append((x,y,ww,hh,area))
    return candidates

# ---------------- CNN model helpers ----------------
def build_small_cnn():  # (not used in inference path; kept for optional local training)
    model = models.Sequential([
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(32,(3,3),activation='relu'), layers.BatchNormalization(), layers.MaxPool2D(),
        layers.Conv2D(64,(3,3),activation='relu'), layers.BatchNormalization(), layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'), layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_model_infer(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            m = tf.keras.models.load_model(path, compile=False)  # compile=False avoids training-time metrics warning
            return m
        except Exception as e:
            st.warning(f"Failed to load model: {e}")
            return None
    return None

# ---------------- OCR with CNN (batch + confidence) ----------------
def ocr_with_cnn(warp_bgr, n, model, conf_thresh=0.65):
    bin_img = remove_background_and_lines(normalize_color(warp_bgr))
    side = bin_img.shape[0]; cell = side // n
    candidates = extract_candidates_from_binary(bin_img, n)
    grid = [[0]*n for _ in range(n)]
    detected = 0
    batch_inputs, batch_coords = [], []
    for i in range(n):
        for j in range(n):
            blobs = candidates.get((i,j), [])
            if not blobs: continue
            blob = sorted(blobs, key=lambda b: b[4], reverse=True)[0]
            x,y,ww,hh,area = blob
            pad = max(2, min(6, int(0.08 * max(ww, hh))))
            x1 = max(0, x-pad); y1 = max(0, y-pad)
            x2 = min(side, x+ww+pad); y2 = min(side, y+hh+pad)
            crop = bin_img[y1:y2, x1:x2]
            if crop.size == 0: continue
            inv = cv2.bitwise_not(crop)
            size=28; canvas = 255*np.ones((size,size), dtype=np.uint8)
            ah,aw = inv.shape
            scale = min((size-6)/max(1,aw), (size-6)/max(1,ah))
            nw = max(1,int(aw*scale)); nh = max(1,int(ah*scale))
            try: small = cv2.resize(inv, (nw,nh), interpolation=cv2.INTER_AREA)
            except: small = cv2.resize(inv, (nw,nh))
            xoff=(size-nw)//2; yoff=(size-nh)//2
            canvas[yoff:yoff+nh, xoff:xoff+nw] = small
            arr = canvas.astype('float32')/255.0
            batch_inputs.append(arr.reshape(28,28,1)); batch_coords.append((i,j))
    if not batch_inputs:
        return grid, 0.0, 0
    batch = np.stack(batch_inputs, axis=0)
    try:
        preds = model.predict(batch, verbose=0)
    except Exception as e:
        st.warning(f"Model predict error: {e}")
        return grid, 0.0, 0
    for idx, prob in enumerate(preds):
        i,j = batch_coords[idx]
        cls = int(np.argmax(prob)); conf = float(np.max(prob))
        if conf >= conf_thresh and 1 <= cls <= 9:
            grid[i][j] = cls
            detected += 1
    ratio = detected / (n*n)
    return grid, ratio, detected

# ---------------- Solver (MRV + timeout) ----------------
def box_shape(n):
    if n==6: return 2,3
    r=int(math.sqrt(n))
    if r*r==n: return r,r
    for a in range(r,0,-1):
        if n%a==0: return a, n//a
    return 1,n

def possible_values(board, r, c, n, br, bc):
    used=set()
    for x in range(n):
        if board[r][x]: used.add(board[r][x])
        if board[x][c]: used.add(board[x][c])
    rs=(r//br)*br; cs=(c//bc)*bc
    for i in range(rs, rs+br):
        for j in range(cs, cs+bc):
            if board[i][j]: used.add(board[i][j])
    return [v for v in range(1, n+1) if v not in used]

def solve_mrv(board, n, br, bc, deadline):
    pos=None; best_options=None
    for i in range(n):
        for j in range(n):
            if board[i][j]==0:
                opts = possible_values(board,i,j,n,br,bc)
                if not opts: return False
                if best_options is None or len(opts) < len(best_options):
                    best_options = opts; pos=(i,j)
    if pos is None: return True
    if time.time() > deadline:
        raise TimeoutError("Solver timed out")
    r,c = pos
    for num in best_options:
        board[r][c] = num
        if solve_mrv(board, n, br, bc, deadline): return True
        board[r][c] = 0
    return False

def solve_with_timeout(board, n, br, bc, timeout_s=5.0):
    deadline = time.time() + float(timeout_s)
    try:
        ok = solve_mrv(board, n, br, bc, deadline)
        return ok
    except TimeoutError:
        return None

# ---------------- Display grid ----------------
def display_grid(given, extracted, solution, n, title="Grid"):
    st.write(f"### {title}")
    html = "<div style='display:flex;justify-content:center;'><table style='border-collapse:collapse;'>"
    for i in range(n):
        html += "<tr>"
        for j in range(n):
            g = given[i][j] if given else 0
            e = extracted[i][j] if extracted else 0
            s = solution[i][j] if solution else 0
            if g and g!=0:
                text=str(g); color="#000"
            elif s and s!=0 and (e==0 or s!=e):
                text=str(s); color="#1b5e20"
            elif e and e!=0:
                text=str(e); color="#0b66b2"
            else:
                text=""; color="#000"
            if n==9:
                left="3px solid #4a4a4a" if j%3==0 else "1px solid #9aa0a6"
                top="3px solid #4a4a4a" if i%3==0 else "1px solid #9aa0a6"
            else:
                left="3px solid #4a4a4a" if j%3==0 else "1px solid #9aa0a6"
                top="2px solid #9aa0a6" if i%2==0 else "1px solid #9aa0a6"
            html += f"<td style='width:44px;height:44px;text-align:center;font-weight:800;color:{color};border-left:{left};border-top:{top};font-size:20px;padding:6px'>{text}</td>"
        html += "</tr>"
    html += "</table></div>"
    st.write(html, unsafe_allow_html=True)

# ---------------- App flow UI ----------------
st.markdown("<div style='text-align:center;margin-bottom:8px;'><b>Which puzzle are you solving?</b></div>", unsafe_allow_html=True)
c1,c2 = st.columns(2)
sel=None
with c1:
    if st.button("6 × 6 Sudoku", key="b6"): sel=6
with c2:
    if st.button("9 × 9 Sudoku", key="b9"): sel=9

if 'selected_n' not in st.session_state: st.session_state['selected_n']=None
if sel: st.session_state['selected_n']=sel

if st.session_state['selected_n'] is None:
    st.markdown("<div class='small-note'>Choose puzzle size to continue</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

n = st.session_state['selected_n']
st.markdown(f"<div style='text-align:center;margin-top:6px;margin-bottom:6px;'><b>Selected: {n} × {n}</b></div>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Sudoku image (tight crop or screenshot works best)", type=["png","jpg","jpeg"])
if uploaded is None:
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

start = time.time()
image = Image.open(uploaded).convert("RGB")
st.image(image, caption="Uploaded image", use_column_width=True)
arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
gray_full = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
pts = find_largest_quad(gray_full)
if pts is None:
    st.error("Could not find puzzle boundary automatically. Try cropping tightly to the puzzle and re-upload.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

warp = warp_to_square(arr, pts, min_side=480)

# load model (inference-only)
model = load_model_infer(MODEL_PATH)
if model is None:
    st.warning("CNN model (digit_model.h5) not found in repo root. Please upload a trained model to use CNN OCR.")
    up = st.file_uploader("Upload a trained digit_model.h5 (optional) — then reload the app", type=["h5"])
    if up:
        with open(MODEL_PATH, "wb") as f:
            f.write(up.read())
        st.success("Model uploaded. Please reload the app (top-right) to use it.")
    st.markdown("<div class='small-note'>Without digit_model.h5 the app cannot OCR. Train in Colab or ask me to prepare a tuned model.</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# perform CNN extraction
grid, ratio, detected = ocr_with_cnn(warp, n, model, conf_thresh=0.65)
given_grid = [[0]*n for _ in range(n)]
st.info(f"Extracted: {detected}/{n*n} cells ({ratio*100:.1f}%) using CNN")
display_grid(given_grid, grid if grid else [[0]*n for _ in range(n)], None, n, title="Extracted Grid (blue = OCR result)")

# Solve / Hint
colS, colH = st.columns(2)
if colS.button("Solve Puzzle"):
    board = [row[:] for row in grid] if grid else [[0]*n for _ in range(n)]
    br, bc = box_shape(n)
    for i in range(n):
        for j in range(n):
            v = board[i][j]
            if not (1 <= v <= n): board[i][j] = 0
    with st.spinner("Solving..."):
        result = solve_with_timeout(board, n, br, bc, timeout_s=5.0)
    if result is True:
        display_grid(given_grid, grid, board, n, title="Solved Grid (green = solution digits)")
        st.success("Solved ✅")
    elif result is None:
        st.error("Solving timed out after 5 seconds. OCR may be noisy — try cropping tighter or upload clearer image.")
    else:
        st.error("Could not solve — OCR likely has errors. Try clearer image.")

if colH.button("Get Hint"):
    board = [row[:] for row in grid] if grid else [[0]*n for _ in range(n)]
    br, bc = box_shape(n)
    for i in range(n):
        for j in range(n):
            if not (1 <= board[i][j] <= n): board[i][j] = 0
    with st.spinner("Computing hint..."):
        result = solve_with_timeout(board, n, br, bc, timeout_s=5.0)
    if result is True:
        found=False
        for i in range(n):
            for j in range(n):
                if grid[i][j]==0:
                    display_grid(given_grid, grid, board, n, title="Hint (green = solution digits)")
                    st.info(f"Hint: place **{board[i][j]}** at row {i+1}, column {j+1}")
                    found=True; break
            if found: break
        if not found:
            st.success("No empty cell found — puzzle appears complete.")
    elif result is None:
        st.error("Hint timed out — OCR may be noisy.")
    else:
        st.error("Cannot provide hint — puzzle likely unsolvable from detected digits.")

# Continue / Upload new puzzle
st.markdown("---")
cA, cB = st.columns(2)
if cA.button("Continue with Same Puzzle"):
    st.rerun()
if cB.button("Upload a New Puzzle"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

elapsed = time.time() - start
st.markdown(f"<div class='small-note'>Processed in {elapsed:.2f} s (CNN engine).</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
