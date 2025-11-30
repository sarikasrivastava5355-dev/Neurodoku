# app.py — NEURODOKU with small CNN digit classifier (6x6 & 9x9)
# Requirements: streamlit, numpy, pillow, opencv-python-headless, tensorflow, scikit-learn
# packages.txt: tesseract-ocr (optional), libgl1, libglib2.0-0

import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np
import cv2, os, io, time, math, re
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# set page
st.set_page_config(page_title="NEURODOKU", layout="centered")

# ------------------ CSS: Notebook full page ------------------
st.markdown("""
<style>
html, body { background: #fbf8ef; color:#0b2540; font-family:'Segoe UI', Roboto, Arial, sans-serif; }
body { background-image: linear-gradient(rgba(0,0,0,0.03) 1px, transparent 1px); background-size: 100% 36px; }
.header { text-align:center; margin-top:12px; margin-bottom:6px; }
.card { background: rgba(255,255,255,0.82); padding:18px; border-radius:12px; width:90%; margin:auto;
       box-shadow: 0 8px 30px rgba(10,30,50,0.06); border:1px solid rgba(10,30,50,0.03); }
.title { font-weight:900; font-size:36px; text-align:center; color:#083048; margin-bottom:6px; }
.btn { margin:8px; }
.big-button > button { background: linear-gradient(90deg,#2b6ea3,#1e73b7); color:white; padding:10px 18px; border-radius:10px; font-weight:700; }
.solve-btn > button { background: linear-gradient(90deg,#0b8f5c,#2e7d32); color:white; padding:10px 18px; border-radius:10px; font-weight:700; }
.hint-btn > button { background: linear-gradient(90deg,#d97706,#f59e0b); color:white; padding:10px 18px; border-radius:10px; font-weight:700; }
.small-note { color:#264653; text-align:center; margin-top:6px; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ------------------ Logo loader: make circular & transparent if white bg ------------------
def make_logo_circle_transparent(path, size_px=160):
    """Load image path (any ext), center-crop square, convert white background to transparent,
       mask to circle, return BytesIO PNG."""
    try:
        im = Image.open(path).convert("RGBA")
    except Exception:
        return None
    # center crop
    w,h = im.size
    s = min(w,h)
    left = (w-s)//2; top = (h-s)//2
    im = im.crop((left, top, left+s, top+s)).resize((size_px, size_px), Image.LANCZOS)
    # replace near-white with transparent
    data = np.array(im)
    r,g,b,a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    # detect white-ish pixels
    white_mask = (r>240) & (g>240) & (b>240)
    data[white_mask,3] = 0
    im = Image.fromarray(data)
    # apply circular mask to ensure round shape
    mask = Image.new("L", (size_px, size_px), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0,0,size_px,size_px), fill=255)
    im.putalpha(mask)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf

logo_buf = None
if os.path.exists("logo.png"):
    logo_buf = make_logo_circle_transparent("logo.png", size_px=160)

st.markdown("<div class='header'>", unsafe_allow_html=True)
if logo_buf:
    st.image(logo_buf, width=160)
st.markdown("<div class='title'>NEURODOKU</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)

# ------------------ Preprocessing utilities ------------------
def find_largest_quad(gray):
    if gray is None: return None
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 40, 150)
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts[:8]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            return approx.reshape(4,2)
    return None

def order_pts(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]; rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]; rect[3] = pts[np.argmax(diff)]
    return rect

def warp_to_square(img, pts, min_side=360):
    rect = order_pts(pts)
    (tl,tr,br,bl) = rect
    widthA = np.linalg.norm(br-bl); widthB = np.linalg.norm(tr-tl)
    heightA = np.linalg.norm(tr-br); heightB = np.linalg.norm(tl-bl)
    side = int(max(widthA, widthB, heightA, heightB))
    if side < min_side: side = min_side
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (side, side))

def remove_grid_lines(binary):
    h,w = binary.shape
    vertical_k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h//40)))
    horizontal_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w//40), 1))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_k, iterations=1)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_k, iterations=1)
    lines = cv2.bitwise_or(vertical, horizontal)
    cleaned = cv2.subtract(binary, lines)
    cleaned = cv2.medianBlur(cleaned, 3)
    return cleaned

def preprocess_for_digit_classification(warp_bgr):
    # returns binary image with digits as white on black background (for easy ROI extraction)
    gray = cv2.cvtColor(warp_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    # invert-aware adaptive threshold: produce digits white
    th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 12)
    cleaned = remove_grid_lines(th)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thick = cv2.dilate(cleaned, kernel, iterations=1)
    final = cv2.morphologyEx(thick, cv2.MORPH_OPEN, kernel, iterations=1)
    return final

# ------------------ CNN model helpers ------------------
MODEL_PATH = "digit_model.h5"

def build_small_cnn(input_shape=(28,28,1), num_classes=10):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32,(3,3),activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_model_if_exists():
    if os.path.exists(MODEL_PATH):
        try:
            model = tf.keras.models.load_model(MODEL_PATH)
            return model
        except Exception as e:
            st.warning("Could not load digit_model.h5 (corrupt?). Will attempt to retrain.")
            return None
    return None

# ------------------ Extract digits using CNN ------------------
def ocr_with_cnn(warp_bgr, n, model):
    bin_img = preprocess_for_digit_classification(warp_bgr)  # digits white on black
    side = bin_img.shape[0]
    cell = side // n
    grid = [[0]*n for _ in range(n)]
    detected = 0
    for i in range(n):
        for j in range(n):
            y = i*cell; x = j*cell
            m = max(4, cell//12)
            y1 = max(0, y+m); y2 = min(side, (i+1)*cell - m)
            x1 = max(0, x+m); x2 = min(side, (j+1)*cell - m)
            crop = bin_img[y1:y2, x1:x2] if (y2>y1 and x2>x1) else bin_img[y:y+cell, x:x+cell]
            if crop.size == 0: continue
            white_frac = np.sum(crop==255) / (crop.size+1e-9)
            if white_frac < 0.01:
                continue
            # prepare for CNN: center on 28x28 white background black digit inverted for training style
            inv = cv2.bitwise_not(crop)
            size = 28
            canvas = 255 * np.ones((size,size), dtype=np.uint8)
            ah,aw = inv.shape
            scale = min((size-6)/max(1,aw), (size-6)/max(1,ah))
            nw = max(1,int(aw*scale)); nh = max(1,int(ah*scale))
            try:
                small = cv2.resize(inv, (nw,nh), interpolation=cv2.INTER_AREA)
            except:
                small = cv2.resize(inv, (nw,nh))
            xoff = (size-nw)//2; yoff = (size-nh)//2
            canvas[yoff:yoff+nh, xoff:xoff+nw] = small
            # normalize and feed to model
            arr = canvas.astype('float32')/255.0
            arr = arr.reshape(1,28,28,1)
            pred = model.predict(arr, verbose=0)
            cls = np.argmax(pred, axis=1)[0]
            grid[i][j] = int(cls)
            detected += 1
    ratio = detected / (n*n)
    return grid, ratio, detected

# ------------------ Solver code (same as before) ------------------
def box_shape(n):
    if n==6: return 2,3
    r=int(math.sqrt(n))
    if r*r==n: return r,r
    for a in range(r,0,-1):
        if n%a==0: return a, n//a
    return 1,n

def find_empty(board, n):
    for i in range(n):
        for j in range(n):
            if board[i][j]==0: return (i,j)
    return None

def valid(board, r, c, num, n, br, bc):
    for x in range(n):
        if board[r][x]==num or board[x][c]==num: return False
    rs=(r//br)*br; cs=(c//bc)*bc
    for i in range(rs, rs+br):
        for j in range(cs, cs+bc):
            if board[i][j]==num: return False
    return True

def solve(board, n, br, bc):
    pos = find_empty(board, n)
    if not pos: return True
    r,c = pos
    for num in range(1, n+1):
        if valid(board, r, c, num, n, br, bc):
            board[r][c] = num
            if solve(board, n, br, bc): return True
            board[r][c] = 0
    return False

# ------------------ Display grid function ------------------
def display_grid_colored(grid_given, grid_extracted, solution, n, title="Grid"):
    st.write(f"### {title}")
    html = "<div style='display:flex;justify-content:center;'><table style='border-collapse:collapse;'>"
    for i in range(n):
        html += "<tr>"
        for j in range(n):
            given = grid_given[i][j] if grid_given else 0
            ext = grid_extracted[i][j] if grid_extracted else 0
            sol = solution[i][j] if solution else 0
            if given and given!=0:
                text = str(given); color="#000000"
            elif sol and sol!=0 and (ext==0 or sol!=ext):
                text = str(sol); color="#1b5e20"
            elif ext and ext!=0:
                text = str(ext); color="#0b66b2"
            else:
                text = ""; color="#000000"
            if n==9:
                left = "3px solid #4a4a4a" if j%3==0 else "1px solid #9aa0a6"
                top = "3px solid #4a4a4a" if i%3==0 else "1px solid #9aa0a6"
            else:
                left = "3px solid #4a4a4a" if j%3==0 else "1px solid #9aa0a6"
                top = "2px solid #9aa0a6" if i%2==0 else "1px solid #9aa0a6"
            html += f"<td style='width:44px;height:44px;text-align:center;font-weight:800;color:{color};border-left:{left};border-top:{top};font-size:20px;padding:6px'>{text}</td>"
        html += "</tr>"
    html += "</table></div>"
    st.write(html, unsafe_allow_html=True)

# ------------------ App flow ------------------
st.markdown("<div style='text-align:center;margin-bottom:8px;'><b>Which puzzle are you solving?</b></div>", unsafe_allow_html=True)
c1,c2 = st.columns(2)
choice=None
with c1:
    if st.button("6 × 6 Sudoku", key="btn6"): choice=6
with c2:
    if st.button("9 × 9 Sudoku", key="btn9"): choice=9

if 'selected_n' not in st.session_state: st.session_state['selected_n']=None
if choice: st.session_state['selected_n']=choice

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

start=time.time()
image = Image.open(uploaded).convert("RGB")
st.image(image, caption="Uploaded image", use_column_width=True)
arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
gray_full = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
pts = find_largest_quad(gray_full)
if pts is None:
    st.error("Could not locate puzzle boundary. Try cropping tighter around the puzzle and re-upload.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

warp = warp_to_square(arr, pts, min_side=480)

# load / build model
model = load_model_if_exists()
if model is None:
    st.warning("Digit model not found in repository (digit_model.h5). Prediction will be unavailable until model exists.")
    st.info("Options: (1) Upload a prepared 'digit_model.h5' (preferred), or (2) Train here (not recommended on Streamlit Cloud).")
    col_up, col_train = st.columns([1,1])
    with col_up:
        uploaded_model = st.file_uploader("Upload digit_model.h5 (optional, faster)", type=["h5"])
        if uploaded_model:
            with open(MODEL_PATH, "wb") as f:
                f.write(uploaded_model.read())
            st.success("Model uploaded — reload app to use it.")
            st.stop()
    with col_train:
        if st.button("Train digit model now (Colab recommended)"):
            st.info("Training will start — this can take many minutes and is recommended to run in Colab, not Streamlit Cloud.")
            # simple quick training on MNIST as fallback
            (x_train,y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()
            x = np.concatenate([x_train, x_test], axis=0)
            y = np.concatenate([y_train, y_test], axis=0)
            x = x.astype('float32')/255.0
            x = x.reshape((-1,28,28,1))
            x_train,x_val,y_train,y_val = train_test_split(x,y,test_size=0.12, random_state=42)
            model = build_small_cnn()
            model.fit(x_train, y_train, validation_data=(x_val,y_val), epochs=6, batch_size=256)
            model.save(MODEL_PATH)
            st.success("Training complete and model saved.")
            st.experimental_rerun()
        else:
            st.markdown("<div class='small-note'>To get best results: run the training snippet in Google Colab, download digit_model.h5 and upload it above.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            st.stop()

# If model loaded, perform extraction using CNN
grid, ratio, detected = ocr_with_cnn(warp, n, model)
given_grid = [[0]*n for _ in range(n)]
st.info(f"Extracted: {detected}/{n*n} cells ({ratio*100:.1f}%)")
display_grid_colored(given_grid, grid, None, n, title="Extracted Grid (blue = OCR result)")

# Solve / Hint
colS, colH = st.columns(2)
if colS.button("Solve Puzzle"):
    board = [row[:] for row in grid]
    br, bc = box_shape(n)
    for i in range(n):
        for j in range(n):
            v = board[i][j]
            if not (1<=v<=n): board[i][j]=0
    with st.spinner("Solving..."):
        ok = solve(board, n, br, bc)
    if ok:
        display_grid_colored(given_grid, grid, board, n, title="Solved Grid (green = solution digits)")
        st.success("Solved ✅")
    else:
        st.error("Could not solve — OCR may have errors. Try cropping tighter or provide another puzzle image.")

if colH.button("Get Hint"):
    board = [row[:] for row in grid]
    br, bc = box_shape(n)
    for i in range(n):
        for j in range(n):
            if not (1 <= board[i][j] <= n): board[i][j] = 0
    with st.spinner("Computing hint..."):
        ok = solve(board, n, br, bc)
    if not ok:
        st.error("Cannot provide hint — puzzle likely unsolvable from OCR.")
    else:
        found=False
        for i in range(n):
            for j in range(n):
                if grid[i][j]==0:
                    display_grid_colored(given_grid, grid, board, n, title="Hint (green = solution digits)")
                    st.info(f"Hint: place **{board[i][j]}** at row {i+1}, column {j+1}")
                    found=True; break
            if found: break
        if not found:
            st.success("No empty cell — puzzle appears complete.")

# Continue / Upload new (use st.rerun)
st.markdown("---")
cc1, cc2 = st.columns(2)
if cc1.button("Continue with Same Puzzle"):
    st.rerun()
if cc2.button("Upload a New Puzzle"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

elapsed = time.time() - start
st.markdown(f"<div class='small-note'>Processed in {elapsed:.2f} s.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
