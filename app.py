# app.py — NEURODOKU (Updated: improved OCR preprocessing, centered round logo, full notebook background)
# Put 'logo.png' in repo root (medium size recommended). Requirements:
# streamlit, numpy, pillow, pytesseract, opencv-python-headless

import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageDraw
import numpy as np
import cv2, pytesseract, math, time, io, os, re

# Path to tesseract on Streamlit Cloud
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="NEURODOKU", layout="centered")

# ------------------ Notebook-style full-page CSS ------------------
st.markdown("""
<style>
/* Full-page notebook look */
html, body {
  background: #fbf8ef;
  color: #0b2540;
  font-family: "Segoe UI", Roboto, Arial, sans-serif;
}
body {
  background-image:
    linear-gradient(rgba(0,0,0,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,0.0) 1px, transparent 1px);
  background-size: 100% 36px, 100% 100%;
}
.header { text-align:center; margin-top:14px; margin-bottom:8px; }
.logo-wrap { display:flex; justify-content:center; align-items:center; margin-bottom:6px; }
.title { font-weight:900; font-size:36px; color:#083048; letter-spacing:1px; text-align:center; margin-bottom:4px; }
.card { background: rgba(255,255,255,0.8); padding:18px; border-radius:12px; width:92%; margin-left:auto; margin-right:auto; box-shadow: 0 8px 30px rgba(10,30,50,0.06); border:1px solid rgba(10,30,50,0.03); }
.btn { margin:8px; }
.grid-btn > button { background: linear-gradient(90deg,#2b6ea3,#1e73b7); color:white; font-weight:700; border-radius:10px; padding:10px 18px; }
.action-btn > button { background: linear-gradient(90deg,#0b8f5c,#2e7d32); color:white; font-weight:700; border-radius:10px; padding:10px 18px; }
.hint-btn > button { background: linear-gradient(90deg,#d97706,#f59e0b); color:white; font-weight:700; border-radius:10px; padding:10px 18px; }
.small-note { color:#264653; text-align:center; margin-top:6px; font-size:13px; }
</style>
""", unsafe_allow_html=True)

# ------------------ Logo helper: circular crop + display center ------------------
def load_and_circle_logo(path, size_px=160):
    try:
        im = Image.open(path).convert("RGBA")
    except Exception:
        return None
    # create circle mask and paste
    w,h = im.size
    s = min(w,h)
    # center crop square
    left = (w-s)//2; top = (h-s)//2
    im = im.crop((left, top, left+s, top+s)).resize((size_px, size_px), Image.LANCZOS)
    # circle mask
    mask = Image.new("L", (size_px, size_px), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0,0,size_px, size_px), fill=255)
    im.putalpha(mask)
    # return bytes for st.image
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    return buf

logo_buf = load_and_circle_logo("logo.png", size_px=160)
st.markdown("<div class='header'>", unsafe_allow_html=True)
if logo_buf:
    st.image(logo_buf, width=160)
st.markdown("<div class='title'>NEURODOKU</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)

# ------------------ Image processing and OCR helpers ------------------

def find_largest_quad(gray):
    if gray is None: return None
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 30, 150)
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
    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    side = int(max(widthA, widthB, heightA, heightB))
    if side < min_side: side = min_side
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (side, side))

def remove_grid_lines(gray):
    # remove both vertical and horizontal lines using morphological operations
    # gray should be binary (0/255)
    h, w = gray.shape
    # detect vertical lines
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(10, h//40)))
    vertical = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_v, iterations=1)
    # detect horizontal lines
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (max(10, w//40), 1))
    horizontal = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel_h, iterations=1)
    # combine
    lines = cv2.bitwise_or(vertical, horizontal)
    # subtract lines from image: keep digits
    cleaned = cv2.subtract(gray, lines)
    # further remove small noise
    cleaned = cv2.medianBlur(cleaned, 3)
    return cleaned

def preprocess_for_ocr(warp_rgb):
    # Input warp_rgb: BGR or RGB image (numpy)
    if warp_rgb is None: return None
    gray = cv2.cvtColor(warp_rgb, cv2.COLOR_BGR2GRAY)
    # enhance contrast
    gray = cv2.equalizeHist(gray)
    # adaptive threshold to get binary
    th = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 12)
    # remove grid lines
    cleaned = remove_grid_lines(th)
    # thicken digits a bit (dilation)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    thick = cv2.dilate(cleaned, kernel, iterations=1)
    # small opening to remove specks
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2,2))
    final = cv2.morphologyEx(thick, cv2.MORPH_OPEN, kernel2, iterations=1)
    return final  # binary image with digits white on black background

def ocr_cell_from_binary(bin_img_cell, n):
    # bin_img_cell: binary image with digits white (255) on black (0)
    try:
        # invert to black-on-white for PIL -> tesseract better for single char
        inv = cv2.bitwise_not(bin_img_cell)
        # center on white canvas, upscale x2 for tesseract clarity
        size = 56
        canvas = 255 * np.ones((size,size), dtype=np.uint8)
        ah,aw = inv.shape
        scale = min((size-8)/max(1,aw), (size-8)/max(1,ah))
        nw = max(1, int(aw*scale)); nh = max(1, int(ah*scale))
        try:
            small = cv2.resize(inv, (nw, nh), interpolation=cv2.INTER_AREA)
        except:
            small = cv2.resize(inv, (nw, nh))
        x = (size-nw)//2; y = (size-nh)//2
        canvas[y:y+nh, x:x+nw] = small
        # convert to PIL
        pil = Image.fromarray(canvas)
        # sharpen a little
        pil = pil.filter(ImageFilter.SHARPEN)
        cfg = f'--psm 10 -c tessedit_char_whitelist=0123456789'
        txt = pytesseract.image_to_string(pil, config=cfg)
        txt = re.sub(r'[^0-9]', '', txt).strip()
        return txt
    except Exception:
        return ""

def ocr_full_with_preprocessing(warp_bgr, n):
    # returns grid (n x n), ratio, detected_count
    processed_bin = preprocess_for_ocr(warp_bgr)  # digits white on black
    side = processed_bin.shape[0]
    cell = side // n
    grid = [[0]*n for _ in range(n)]
    detected = 0
    for i in range(n):
        for j in range(n):
            y = i*cell; x = j*cell
            m = max(4, cell//12)
            y1 = max(0, y + m); y2 = min(side, (i+1)*cell - m)
            x1 = max(0, x + m); x2 = min(side, (j+1)*cell - m)
            crop = processed_bin[y1:y2, x1:x2] if (y2>y1 and x2>x1) else processed_bin[y:y+cell, x:x+cell]
            # skip too small
            if crop.size == 0:
                continue
            # compute fraction of white pixels to decide presence
            white_frac = np.sum(crop==255) / (crop.size+1e-9)
            if white_frac < 0.01:
                continue
            txt = ocr_cell_from_binary(crop, n)
            if txt:
                v = int(txt[0])
                if 1 <= v <= n:
                    grid[i][j] = v
                    detected += 1
    ratio = detected / (n*n)
    return grid, ratio, detected

# ------------------ Solver (unchanged) ------------------
def box_shape(n):
    if n==6: return 2,3
    r=int(math.sqrt(n))
    if r*r==n: return r,r
    for a in range(r,0,-1):
        if n%a==0: return a,n//a
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

# ------------------ Grid display (colored) ------------------
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
                text = str(sol); color="#1b5e20"  # green
            elif ext and ext!=0:
                text = str(ext); color="#0b66b2"  # blue
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
c1, c2 = st.columns(2)
sel = None
with c1:
    if st.button("6 × 6 Sudoku", key="b6"): sel = 6
with c2:
    if st.button("9 × 9 Sudoku", key="b9"): sel = 9

if 'selected_n' not in st.session_state: st.session_state['selected_n'] = None
if sel: st.session_state['selected_n'] = sel

if st.session_state['selected_n'] is None:
    st.markdown("<div class='small-note'>Choose puzzle size to continue</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

n = st.session_state['selected_n']
st.markdown(f"<div style='text-align:center;margin-top:6px;margin-bottom:6px;'><b>Selected: {n} × {n}</b></div>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Sudoku image (tight crop / screenshot works best)", type=["png","jpg","jpeg"])
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
    st.error("Could not find grid boundary automatically. Try cropping the image to the puzzle only and re-upload.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

warp = warp_to_square(arr, pts, min_side=360)

# Run OCR extraction with improved preprocessing
extracted_grid, ratio, detected = ocr_full_with_preprocessing(warp, n)
given_grid = [[0]*n for _ in range(n)]  # we don't try to detect "given" vs blank
st.info(f"Extracted: {detected}/{n*n} cells ({ratio*100:.1f}%)")
display_grid_colored(given_grid, extracted_grid, None, n, title="Extracted Grid (blue = OCR result)")

# Buttons Solve / Hint
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
        display_grid_colored(given_grid, extracted_grid, board, n, title="Solved Grid (green = solution digits)")
        st.success("Solved ✅")
    else:
        st.error("Could not solve — OCR may still have errors for this image. Try a clearer crop or upload another sample.")

if colB.button("Get Hint"):
    board = [row[:] for row in extracted_grid]
    br, bc = box_shape(n)
    for i in range(n):
        for j in range(n):
            if not (1 <= board[i][j] <= n): board[i][j] = 0
    with st.spinner("Computing hint..."):
        ok = solve(board, n, br, bc)
    if not ok:
        st.error("Cannot provide hint — puzzle likely unsolvable from current OCR results.")
    else:
        # highlight first empty cell
        found = False
        for i in range(n):
            for j in range(n):
                if extracted_grid[i][j] == 0:
                    display_grid_colored(given_grid, extracted_grid, board, n, title="Hint (green = solution)")
                    st.info(f"Hint: place **{board[i][j]}** at row {i+1}, column {j+1}")
                    found = True
                    break
            if found: break
        if not found:
            st.success("No empty cell found — puzzle appears complete.")

# Continue / Upload new
st.markdown("---")
cc1, cc2 = st.columns(2)
if cc1.button("Continue with Same Puzzle"):
    st.experimental_rerun()
if cc2.button("Upload a New Puzzle"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.experimental_rerun()

elapsed = time.time() - start
st.markdown(f"<div class='small-note'>Processed in {elapsed:.2f} s — preprocessing tuned for your uploaded image.</div>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)
