# app_A.py — NEURODOKU (Full-page notebook background)
# Place your logo at repo root as 'logo.png' (medium size ~160px)
# Requirements: streamlit, numpy, pillow, pytesseract, opencv-python-headless

import streamlit as st
from PIL import Image
import numpy as np
import cv2, pytesseract, re, math, time
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="NEURODOKU", layout="centered", initial_sidebar_state="collapsed")

# ---- STYLE: full-page notebook background ----
st.markdown("""
<style>
/* full-page notebook: lined paper */
body {
  background: #fbfbf7;
  background-image:
    linear-gradient(rgba(0,0,0,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,0,0,0.01) 1px, transparent 1px);
  background-size: 100% 36px, 100% 100%;
  font-family: "Segoe UI", Roboto, sans-serif;
  color: #0b2540;
}
.app-header {text-align:center; margin-top:12px; margin-bottom:6px;}
.logo { height:160px; width:auto; display:block; margin-left:auto; margin-right:auto; }
.title { font-size:36px; font-weight:800; margin-top:6px; color:#0b2540; text-align:center; letter-spacing:1px;}
.subtitle { text-align:center; color:#16426b; margin-bottom:12px; font-size:14px; }
.card { background: rgba(255,255,255,0.6); padding:18px; border-radius:12px; width:88%; margin-left:auto; margin-right:auto; box-shadow: 0 6px 20px rgba(10,30,50,0.06); border:1px solid rgba(10,30,50,0.04); }
.btn { margin:6px; }
.small-note { color:#2b5876; font-size:13px; margin-top:6px; text-align:center; }
.solver-btn > button { background: linear-gradient(90deg,#2b6ea3,#1e73b7); color:white; font-weight:700; border-radius:8px; padding:8px 14px; }
</style>
""", unsafe_allow_html=True)

# ---- header with logo ----
logo_path = "logo.png"  # put your logo file here in repo root
try:
    logo_img = Image.open(logo_path).convert("RGBA")
    st.markdown(f"<div class='app-header'><img src='data:image/png;base64,{st.image(logo_img, output_format='PNG')}' class='logo'/></div>", unsafe_allow_html=True)
except Exception:
    # fallback: try direct display
    try:
        st.image(logo_path, width=160)
    except:
        pass

st.markdown("<div class='title'>NEURODOKU</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Select puzzle size, upload image — automatic extraction & solve</div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

# -------------------- Helper functions --------------------
def find_largest_quad(gray):
    if gray is None: return None
    b = cv2.GaussianBlur(gray, (5,5), 0)
    e = cv2.Canny(b, 50, 150)
    cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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
    d = np.diff(pts, axis=1); rect[1] = pts[np.argmin(d)]; rect[3] = pts[np.argmax(d)]
    return rect

def warp_to_square(gray, pts, min_side=240):
    rect = order_pts(pts)
    (tl,tr,br,bl) = rect
    widthA = np.linalg.norm(br-bl); widthB = np.linalg.norm(tr-tl)
    heightA = np.linalg.norm(tr-br); heightB = np.linalg.norm(tl-bl)
    side = int(max(widthA, widthB, heightA, heightB))
    if side < min_side: side = min_side
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(gray, M, (side, side))

def ocr_cell(img_cell):
    # safe small OCR tuned for single digits
    try:
        if img_cell is None or img_cell.size==0: return ""
        if len(img_cell.shape)==3: img_cell = cv2.cvtColor(img_cell, cv2.COLOR_BGR2GRAY)
        h,w = img_cell.shape
        if h<6 or w<6: return ""
        # increase contrast, normalize
        norm = cv2.equalizeHist(img_cell)
        _, th = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # pad/resize to 28x28 for speed & clarity
        size = 28
        canvas = 255 * np.ones((size,size), dtype=np.uint8)
        ah,aw = th.shape
        scale = min((size-6)/max(1,aw), (size-6)/max(1,ah))
        nw = max(1, int(aw*scale)); nh = max(1, int(ah*scale))
        small = cv2.resize(th, (nw,nh))
        xoff = (size-nw)//2; yoff = (size-nh)//2
        canvas[yoff:yoff+nh, xoff:xoff+nw] = small
        pil = Image.fromarray(canvas)
        cfg = '--psm 10 -c tessedit_char_whitelist=0123456789'
        txt = pytesseract.image_to_string(pil, config=cfg)
        txt = re.sub(r'[^0-9]', '', txt).strip()
        return txt
    except:
        return ""

def ocr_full_for_size(warp, n):
    side = warp.shape[0]; cell = side // n
    grid = [[0]*n for _ in range(n)]
    detected = 0
    for i in range(n):
        for j in range(n):
            y = i*cell; x = j*cell
            m = max(3, cell//12)
            y1 = max(0, y+m); y2 = min(side, y+cell-m)
            x1 = max(0, x+m); x2 = min(side, x+cell-m)
            crop = warp[y1:y2, x1:x2] if (y2>y1 and x2>x1) else warp[y:y+cell, x:x+cell]
            txt = ocr_cell(crop)
            if txt:
                v = int(txt[0])
                if 1 <= v <= n:
                    grid[i][j] = v; detected += 1
    ratio = detected / (n*n)
    return grid, ratio, detected

def box_shape(n):
    if n==6: return 2,3
    r=int(math.sqrt(n))
    if r*r==n: return r,r
    for a in range(r,0,-1):
        if n%a==0: return a, n//a
    return 1, n

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
            board[r][c]=num
            if solve(board, n, br, bc): return True
            board[r][c]=0
    return False

# grid display with colors: original (black), extracted (blue), solution (green)
def display_grid(grid_given, grid_extracted, solution, n, title="Grid"):
    st.write(f"### {title}")
    html = "<table style='margin:auto;'>"
    for i in range(n):
        html += "<tr>"
        for j in range(n):
            given = grid_given[i][j]
            ext = grid_extracted[i][j]
            sol = solution[i][j] if solution else None
            if given and given!=0:
                text = str(given)
                color = "#000"  # black for given
            elif sol and sol!=0 and (ext==0 or sol!=ext):
                text = str(sol)
                color = "#2e7d32"  # green solution
            elif ext and ext!=0:
                text = str(ext)
                color = "#1e88e5"  # blue extracted
            else:
                text = ""
                color = "#000"
            # thick separators for 9x9 box lines
            if n==9:
                left = "3px solid #999" if j%3==0 else "1px solid #bbb"
                top  = "3px solid #999" if i%3==0 else "1px solid #bbb"
            else:
                left = "3px solid #999" if j%3==0 else "1px solid #bbb"
                top  = "2px solid #bbb" if i%2==0 else "1px solid #bbb"
            cell = f"<td style='width:42px;height:42px;text-align:center;font-weight:700;color:{color};border-left:{left};border-top:{top};font-size:18px'>{text}</td>"
            html += cell
        html += "</tr>"
    html += "</table>"
    st.write(html, unsafe_allow_html=True)

# --------- App flow (no manual edit) ----------
st.markdown("<div style='text-align:center;margin-bottom:8px;'><b>Which puzzle are you solving?</b></div>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
selected_size = None
with col1:
    if st.button("6 × 6 Sudoku", key="btn6"):
        selected_size = 6
with col2:
    if st.button("9 × 9 Sudoku", key="btn9"):
        selected_size = 9

# store selected size in session
if 'selected_size' not in st.session_state:
    st.session_state['selected_size'] = None
if selected_size:
    st.session_state['selected_size'] = selected_size

if st.session_state.get('selected_size') is None:
    st.markdown("<div class='small-note'>Choose puzzle size to continue</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

n = st.session_state['selected_size']
st.markdown(f"<div style='text-align:center;margin-top:6px;margin-bottom:6px;'><b>Selected: {n} × {n}</b></div>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Sudoku image (tight crop/screenshot works best)", type=["png","jpg","jpeg"])
if uploaded is None:
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

# process
start = time.time()
image = Image.open(uploaded).convert("RGB")
st.image(image, caption="Uploaded image", use_column_width=True)
arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
pts = find_largest_quad(gray)
if pts is None:
    st.error("Could not find grid. Try tighter crop or clearer image.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()
warp = warp_to_square(gray, pts)
# OCR extraction
extracted_grid, ratio, detected = ocr_full_for_size(warp, n)
# We consider any non-zero cell in extracted_grid as extracted; set given_grid to zeros (we don't have labeled given cells from image)
given_grid = [[0]*n for _ in range(n)]  # we don't infer 'given' vs blank; treat OCRed numbers as extracted
st.info(f"Extracted: {detected}/{n*n} cells ({ratio*100:.1f}%)")
display_grid(given_grid, extracted_grid, None, n, title="Extracted Grid (blue = OCR result)")

# Solve / Hint
colA, colB = st.columns(2)
if colA.button("Solve Puzzle"):
    board = [row[:] for row in extracted_grid]
    br, bc = box_shape(n)
    # sanitize
    for i in range(n):
        for j in range(n):
            v = board[i][j]
            if not (1 <= v <= n): board[i][j] = 0
    with st.spinner("Solving..."):
        ok = solve(board, n, br, bc)
    if ok:
        display_grid(given_grid, extracted_grid, board, n, title="Solved Grid (green = solution digits)")
        st.success("Solved ✅")
    else:
        st.error("Could not solve — OCR may have errors. Try a clearer image.")

if colB.button("Get Hint"):
    board = [row[:] for row in extracted_grid]
    br, bc = box_shape(n)
    for i in range(n):
        for j in range(n):
            v = board[i][j]
            if not (1 <= v <= n): board[i][j] = 0
    with st.spinner("Computing hint..."):
        ok = solve(board, n, br, bc)
    if not ok:
        st.error("Cannot produce a hint — puzzle likely unsolvable with current OCR results.")
    else:
        # highlight first empty cell
        for i in range(n):
            for j in range(n):
                if extracted_grid[i][j] == 0:
                    display_grid(given_grid, extracted_grid, board, n, title="Hint (green = solution, highlighted cell)")
                    st.info(f"Hint: place **{board[i][j]}** at row {i+1}, column {j+1}")
                    break
            else:
                continue
            break

# Continue / Upload new
st.markdown("---")
c1, c2 = st.columns(2)
if c1.button("Continue with Same Puzzle"):
    st.experimental_rerun()
if c2.button("Upload a New Puzzle"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.experimental_rerun()

st.markdown("</div>", unsafe_allow_html=True)
