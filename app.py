# app.py — NEURODOKU (robust mixed-input version)
# Paste into repo root. Put logo.png in repo root. Put digit_model.h5 in repo root (optional).
# Requirements: streamlit, numpy, pillow, opencv-python-headless, tensorflow>=2.11, pytesseract, scikit-learn

import streamlit as st
from PIL import Image, ImageDraw, ImageFilter, ImageOps
import numpy as np
import cv2, os, io, time, math, re, base64
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# --- Config & page ---
st.set_page_config(page_title="NEURODOKU", layout="centered")
MODEL_PATH = "digit_model.h5"

# --- Helper: embed circular logo as data URI so Streamlit won't block it ---
def make_logo_datauri(path="logo.png", size_px=160):
    if not os.path.exists(path):
        return None
    im = Image.open(path).convert("RGBA")
    # center-crop to square
    w,h = im.size
    s = min(w,h)
    left=(w-s)//2; top=(h-s)//2
    im = im.crop((left,top,left+s,top+s)).resize((size_px,size_px), Image.LANCZOS)
    # convert near-white to transparent
    data = np.array(im)
    r,g,b,a = data[:,:,0], data[:,:,1], data[:,:,2], data[:,:,3]
    white_mask = (r>240)&(g>240)&(b>240)
    data[white_mask,3]=0
    im = Image.fromarray(data)
    # circular mask
    mask = Image.new("L", (size_px,size_px), 0)
    draw = ImageDraw.Draw(mask); draw.ellipse((0,0,size_px,size_px), fill=255)
    im.putalpha(mask)
    buf = io.BytesIO(); im.save(buf, format="PNG"); buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    return f"data:image/png;base64,{b64}"

logo_data = make_logo_datauri("logo.png", 160)

# --- CSS / Notebook style header ---
st.markdown("""
<style>
html, body { background:#fbf8ef; color:#0b2540; font-family:'Segoe UI', Roboto, Arial, sans-serif; }
body { background-image: linear-gradient(rgba(0,0,0,0.03) 1px, transparent 1px); background-size: 100% 36px; }
.header { text-align:center; margin-top:12px; margin-bottom:6px; }
.card { background: rgba(255,255,255,0.88); padding:18px; border-radius:12px; width:92%; margin:auto; box-shadow:0 8px 30px rgba(10,30,50,0.06); border:1px solid rgba(10,30,50,0.03); }
.title { font-weight:900; font-size:36px; text-align:center; color:#083048; margin-bottom:6px; }
.small-note { color:#264653; text-align:center; margin-top:6px; font-size:13px; }
button.stButton>div { font-weight:700; }
</style>
""", unsafe_allow_html=True)

# render logo + title
st.markdown("<div class='header'>", unsafe_allow_html=True)
if logo_data:
    st.markdown(f"<img src='{logo_data}' style='height:160px;width:160px;border-radius:50%;display:block;margin-left:auto;margin-right:auto;'/>", unsafe_allow_html=True)
st.markdown("<div class='title'>NEURODOKU</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

# ---------------- Image processing utilities ----------------
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
        if len(approx)==4: return approx.reshape(4,2)
    return None

def order_pts(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1); rect[0]=pts[np.argmin(s)]; rect[2]=pts[np.argmax(s)]
    d = np.diff(pts, axis=1); rect[1]=pts[np.argmin(d)]; rect[3]=pts[np.argmax(d)]
    return rect

def warp_to_square(img, pts, min_side=480):
    rect = order_pts(pts); (tl,tr,br,bl)=rect
    widthA=np.linalg.norm(br-bl); widthB=np.linalg.norm(tr-tl)
    heightA=np.linalg.norm(tr-br); heightB=np.linalg.norm(tl-bl)
    side=int(max(widthA,widthB,heightA,heightB))
    if side < min_side: side = min_side
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (side,side))

def normalize_color(bg_img):
    """Convert mixed-color puzzle to cleaner grayscale where digits contrast."""
    lab = cv2.cvtColor(bg_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    # enhance L channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l2 = clahe.apply(l)
    norm = cv2.merge((l2,a,b))
    rgb = cv2.cvtColor(norm, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    return gray

def remove_background_and_lines(gray):
    # adaptive threshold (invert so digits white)
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 10)
    # morphological open to detect grid lines
    h,w = th.shape
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h//40)))
    hor_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w//40), 1))
    vertical = cv2.morphologyEx(th, cv2.MORPH_OPEN, vert_kernel, iterations=1)
    horizontal = cv2.morphologyEx(th, cv2.MORPH_OPEN, hor_kernel, iterations=1)
    lines = cv2.bitwise_or(vertical, horizontal)
    cleaned = cv2.subtract(th, lines)
    # remove small noise and thicken digits a bit
    cleaned = cv2.medianBlur(cleaned, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    thick = cv2.dilate(cleaned, kernel, iterations=1)
    final = cv2.morphologyEx(thick, cv2.MORPH_OPEN, kernel, iterations=1)
    return final  # white digits on black background

# ---------------- CNN model helpers ----------------
def build_small_cnn():
    model = models.Sequential([
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(32,(3,3),activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(),
        layers.Conv2D(64,(3,3),activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def load_model_quiet(path=MODEL_PATH):
    if os.path.exists(path):
        try:
            m = tf.keras.models.load_model(path)
            # compile to suppress the absl warning (harmless) and build metrics
            m.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            return m
        except Exception as e:
            st.warning(f"Failed to load model: {e}")
            return None
    return None

# ---------------- Extract using CNN with strong preprocessing ----------------
def ocr_with_cnn(warp_bgr, n, model):
    bin_img = remove_background_and_lines(normalize_color(warp_bgr))
    side = bin_img.shape[0]; cell = side // n
    grid = [[0]*n for _ in range(n)]
    detected = 0
    for i in range(n):
        for j in range(n):
            y=i*cell; x=j*cell
            m = max(4, cell//12)
            y1=max(0,y+m); y2=min(side,(i+1)*cell-m)
            x1=max(0,x+m); x2=min(side,(j+1)*cell-m)
            crop = bin_img[y1:y2, x1:x2] if (y2>y1 and x2>x1) else bin_img[y:y+cell, x:x+cell]
            if crop.size==0: continue
            white_frac = np.sum(crop==255)/(crop.size+1e-9)
            if white_frac < 0.01: continue
            # prepare 28x28 input for model
            inv = cv2.bitwise_not(crop)
            size=28
            canvas = 255*np.ones((size,size), dtype=np.uint8)
            ah,aw = inv.shape
            scale = min((size-6)/max(1,aw), (size-6)/max(1,ah))
            nw = max(1,int(aw*scale)); nh = max(1,int(ah*scale))
            try:
                small = cv2.resize(inv, (nw,nh), interpolation=cv2.INTER_AREA)
            except:
                small = cv2.resize(inv, (nw,nh))
            xoff=(size-nw)//2; yoff=(size-nh)//2
            canvas[yoff:yoff+nh, xoff:xoff+nw] = small
            arr = canvas.astype('float32')/255.0; arr = arr.reshape(1,28,28,1)
            pred = model.predict(arr, verbose=0)
            cls = int(np.argmax(pred, axis=1)[0])
            # if model is unsure (very low prob) we skip
            if np.max(pred) < 0.5:
                # low confidence — skip (will be 0)
                continue
            grid[i][j] = cls
            detected += 1
    ratio = detected/(n*n)
    return grid, ratio, detected

# ---------------- Fallback OCR (tesseract) ----------------
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def ocr_cell_tesseract(cell_img):
    try:
        pil = Image.fromarray(cell_img)
        cfg = '--psm 10 -c tessedit_char_whitelist=0123456789'
        txt = pytesseract.image_to_string(pil, config=cfg)
        txt = re.sub(r'[^0-9]','',txt).strip()
        return txt
    except Exception:
        return ""

def ocr_with_tesseract(warp_bgr, n):
    bin_img = remove_background_and_lines(normalize_color(warp_bgr))
    side = bin_img.shape[0]; cell = side // n
    grid = [[0]*n for _ in range(n)]; detected=0
    for i in range(n):
        for j in range(n):
            y=i*cell; x=j*cell
            m=max(4,cell//12)
            y1=max(0,y+m); y2=min(side,(i+1)*cell-m)
            x1=max(0,x+m); x2=min(side,(j+1)*cell-m)
            crop = bin_img[y1:y2, x1:x2] if (y2>y1 and x2>x1) else bin_img[y:y+cell, x:x+cell]
            if crop.size==0: continue
            white_frac = np.sum(crop==255)/(crop.size+1e-9)
            if white_frac < 0.01: continue
            inv = cv2.bitwise_not(crop)
            # center and resize for tesseract
            size=28; canvas = 255*np.ones((size,size),dtype=np.uint8)
            ah,aw = inv.shape
            scale = min((size-6)/max(1,aw),(size-6)/max(1,ah))
            nw = max(1,int(aw*scale)); nh = max(1,int(ah*scale))
            try: small=cv2.resize(inv,(nw,nh),interpolation=cv2.INTER_AREA)
            except: small=cv2.resize(inv,(nw,nh))
            xoff=(size-nw)//2; yoff=(size-nh)//2
            canvas[yoff:yoff+nh, xoff:xoff+nw]=small
            txt = ocr_cell_tesseract(canvas)
            if txt:
                v=int(txt[0]); grid[i][j]=v; detected+=1
    ratio = detected/(n*n)
    return grid, ratio, detected

# ---------------- Solver functions ----------------
def box_shape(n):
    if n==6: return 2,3
    r=int(math.sqrt(n))
    if r*r==n: return r,r
    for a in range(r,0,-1):
        if n%a==0: return a, n//a
    return 1,n

def find_empty(board,n):
    for i in range(n):
        for j in range(n):
            if board[i][j]==0: return (i,j)
    return None

def valid(board,r,c,num,n,br,bc):
    for x in range(n):
        if board[r][x]==num or board[x][c]==num: return False
    rs=(r//br)*br; cs=(c//bc)*bc
    for i in range(rs, rs+br):
        for j in range(cs, cs+bc):
            if board[i][j]==num: return False
    return True

def solve(board,n,br,bc):
    pos=find_empty(board,n)
    if not pos: return True
    r,c=pos
    for num in range(1,n+1):
        if valid(board,r,c,num,n,br,bc):
            board[r][c]=num
            if solve(board,n,br,bc): return True
            board[r][c]=0
    return False

# ---------------- Display grid ----------------
def display_grid(grid_given, grid_extracted, solution, n, title="Grid"):
    st.write(f"### {title}")
    html = "<div style='display:flex;justify-content:center;'><table style='border-collapse:collapse;'>"
    for i in range(n):
        html += "<tr>"
        for j in range(n):
            given = grid_given[i][j] if grid_given else 0
            ext = grid_extracted[i][j] if grid_extracted else 0
            sol = solution[i][j] if solution else 0
            if given and given!=0:
                text=str(given); color="#000"
            elif sol and sol!=0 and (ext==0 or sol!=ext):
                text=str(sol); color="#1b5e20"
            elif ext and ext!=0:
                text=str(ext); color="#0b66b2"
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

# ---------------- App flow ----------------
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

uploaded = st.file_uploader("Upload Sudoku image (tight crop/screenshot works best)", type=["png","jpg","jpeg"])
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
    st.error("Could not find grid boundary automatically. Try cropping the image to puzzle only and re-upload.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

warp = warp_to_square(arr, pts, min_side=480)

# try to load CNN model
model = load_model_quiet(MODEL_PATH)
used_engine = None
grid = None; ratio=0; detected=0

if model is not None:
    try:
        grid, ratio, detected = ocr_with_cnn(warp, n, model)
        used_engine = "cnn"
    except Exception as e:
        st.warning(f"CNN extraction failed: {e}")
        model = None

# if CNN not loaded or low detection, try tesseract fallback
if model is None or detected < (0.25 * n * n):
    # try tesseract as fallback
    tgrid, tratio, tdet = ocr_with_tesseract(warp, n)
    # choose whichever extracted more
    if tdet > detected:
        grid, ratio, detected = tgrid, tratio, tdet
        used_engine = "tesseract"
    else:
        # keep cnn result even if low
        used_engine = "cnn" if model else "tesseract"

given_grid = [[0]*n for _ in range(n)]
st.info(f"Extracted: {detected}/{n*n} cells ({ratio*100:.1f}%) using {used_engine}")
display_grid(given_grid, grid if grid else [[0]*n for _ in range(n)], None, n, title="Extracted Grid (blue = OCR result)")

# Solve/Hint
colS, colH = st.columns(2)
if colS.button("Solve Puzzle"):
    board = [row[:] for row in grid] if grid else [[0]*n for _ in range(n)]
    br, bc = box_shape(n)
    for i in range(n):
        for j in range(n):
            v = board[i][j]
            if not (1 <= v <= n): board[i][j] = 0
    with st.spinner("Solving..."):
        ok = solve(board, n, br, bc)
    if ok:
        display_grid(given_grid, grid, board, n, title="Solved Grid (green = solution digits)")
        st.success("Solved ✅")
    else:
        st.error("Could not solve — OCR likely has errors. Try a clearer crop or use another image.")

if colH.button("Get Hint"):
    board = [row[:] for row in grid] if grid else [[0]*n for _ in range(n)]
    br, bc = box_shape(n)
    for i in range(n):
        for j in range(n):
            if not (1 <= board[i][j] <= n): board[i][j] = 0
    with st.spinner("Computing hint..."):
        ok = solve(board, n, br, bc)
    if not ok:
        st.error("Cannot provide hint — puzzle likely unsolvable from detected digits.")
    else:
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

# Continue / Upload new
st.markdown("---")
cA, cB = st.columns(2)
if cA.button("Continue with Same Puzzle"):
    st.rerun()
if cB.button("Upload a New Puzzle"):
    for k in list(st.session_state.keys()): del st.session_state[k]
    st.rerun()

elapsed=time.time()-start
st.markdown(f"<div class='small-note'>Processed in {elapsed:.2f} s (engine: {used_engine}).</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
