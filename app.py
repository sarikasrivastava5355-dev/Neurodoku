# app.py — NEURODOKU (FINAL: 6x6 & 9x9, Auto-detect + Manual override, Editable grid)
# Requirements: streamlit, numpy, pillow, opencv-python-headless==4.10.0.84, pytesseract, pandas, scikit-image
# packages.txt: tesseract-ocr, libgl1, libglib2.0-0
# runtime.txt: python-3.11

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pytesseract
import re
import math
import time

# Tesseract path for Streamlit Cloud
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="NEURODOKU (6x6 & 9x9)", layout="centered")

# --------- UI Styling (Blue Tech) ----------
st.markdown("""
    <style>
    body { background: linear-gradient(135deg,#071428,#0b2540); color:#e6f7ff; font-family: 'Segoe UI', sans-serif; }
    .title { font-size:34px; font-weight:800; text-align:center;
        background:linear-gradient(90deg,#61dafb,#1e88e5); -webkit-background-clip:text; color:transparent; margin-bottom:6px; }
    .subtitle { text-align:center; color:#9fd6ff; margin-bottom:12px; }
    .card { background: rgba(255,255,255,0.03); padding:16px; border-radius:12px; border:1px solid rgba(30,136,229,0.06); }
    .stButton>button { background: linear-gradient(90deg,#0288d1,#0277bd); color:white; border-radius:8px; padding:8px 14px; font-weight:700;}
    table { border-collapse:collapse; margin-top:8px;}
    td { width:36px; height:36px; text-align:center; font-weight:700; font-size:16px; color:#e6f7ff; border:1px solid rgba(97,218,251,0.14);}
    .highlight { background:#ffd54f; color:#001; }
    .note { color:#bfe9ff; font-size:13px; margin-top:6px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🧠 NEURODOKU</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Auto-detect (6×6/9×9) with manual override — edit OCR before solving</div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

# --------- Helpers: geometry, OCR (fast), solver, display ---------

def safe_gray(img):
    if img is None:
        return None
    try:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    except:
        return None

def find_largest_quad(gray):
    if gray is None:
        return None
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 150)
    cnts, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
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
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_to_square(gray, pts, min_side=240):
    rect = order_pts(pts)
    (tl,tr,br,bl) = rect
    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    side = int(max(widthA, widthB, heightA, heightB))
    if side < min_side:
        side = min_side
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(gray, M, (side, side))

# Lightweight OCR for speed: single Otsu; small canvas (32x32)
def ocr_cell_fast(img_cell, whitelist):
    try:
        if img_cell is None or img_cell.size == 0:
            return ""
        if len(img_cell.shape) == 3:
            img_cell = cv2.cvtColor(img_cell, cv2.COLOR_BGR2GRAY)
        h,w = img_cell.shape
        if h < 6 or w < 6:
            return ""
        _, th = cv2.threshold(img_cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        size = 32
        canvas = 255 * np.ones((size,size), dtype=np.uint8)
        ah, aw = th.shape
        scale = min((size-6)/max(1,aw), (size-6)/max(1,ah))
        nw = max(1, int(aw * scale)); nh = max(1, int(ah * scale))
        small = cv2.resize(th, (nw, nh))
        xoff = (size - nw)//2; yoff = (size - nh)//2
        canvas[yoff:yoff+nh, xoff:xoff+nw] = small
        pil = Image.fromarray(canvas)
        cfg = f'--psm 10 -c tessedit_char_whitelist={whitelist}'
        txt = pytesseract.image_to_string(pil, config=cfg)
        txt = re.sub(r'[^0-9A-Za-z]', '', txt)
        return txt.strip()
    except:
        return ""

# Sample a few cells to estimate which size is likelier (fast)
def sample_scores(warp, samples_per_dim=3):
    side = warp.shape[0]
    candidates = [6,9]
    scores = {}
    for n in candidates:
        s = 0
        step = max(1, n // samples_per_dim)
        for i in range(0, n, step):
            for j in range(0, n, step):
                cell = side // n
                m = max(3, cell//12)
                y1 = max(0, i*cell + m); y2 = min(side, (i+1)*cell - m)
                x1 = max(0, j*cell + m); x2 = min(side, (j+1)*cell - m)
                crop = warp[y1:y2, x1:x2] if (y2>y1 and x2>x1) else warp[i*cell:(i+1)*cell, j*cell:(j+1)*cell]
                txt = ocr_cell_fast(crop, "0123456789")
                if txt and txt[0].isdigit():
                    s += 1
        scores[n] = s
    return scores

# Full OCR pass for chosen size
def ocr_full_for_size(warp, n):
    side = warp.shape[0]
    cell = side // n
    grid = [[0]*n for _ in range(n)]
    detected = 0
    for i in range(n):
        for j in range(n):
            y = i*cell; x = j*cell
            m = max(3, cell//12)
            y1 = max(0, y+m); y2 = min(side, y+cell-m)
            x1 = max(0, x+m); x2 = min(side, x+cell-m)
            crop = warp[y1:y2, x1:x2] if (y2>y1 and x2>x1) else warp[y:y+cell, x:x+cell]
            txt = ocr_cell_fast(crop, "0123456789")
            if txt:
                c = txt[0]
                if c.isdigit():
                    v = int(c)
                    if 1 <= v <= n:
                        grid[i][j] = v
                        detected += 1
    ratio = detected / (n*n)
    return grid, ratio, detected

# Solver (generic for 6 and 9)
def box_shape(n):
    if n == 6: return 2,3
    root = int(math.sqrt(n))
    if root*root == n: return root, root
    for r in range(root, 0, -1):
        if n % r == 0: return r, n//r
    return 1, n

def find_empty(board, n):
    for i in range(n):
        for j in range(n):
            if board[i][j] == 0:
                return (i,j)
    return None

def valid(board, r, c, num, n, br, bc):
    for x in range(n):
        if board[r][x] == num or board[x][c] == num:
            return False
    rs = (r//br)*br; cs = (c//bc)*bc
    for i in range(rs, rs+br):
        for j in range(cs, cs+bc):
            if board[i][j] == num:
                return False
    return True

def solve(board, n, br, bc):
    pos = find_empty(board, n)
    if not pos:
        return True
    r,c = pos
    for num in range(1, n+1):
        if valid(board, r, c, num, n, br, bc):
            board[r][c] = num
            if solve(board, n, br, bc):
                return True
            board[r][c] = 0
    return False

# Display small table HTML (used for preview)
def display_grid_strings(strings, title="Grid", highlight=None):
    st.write(f"### {title}")
    size = len(strings)
    html = "<table>"
    for i in range(size):
        html += "<tr>"
        for j in range(size):
            val = strings[i][j]
            is_h = highlight and (i,j)==highlight
            style = "background:#ffd54f;color:#001;" if is_h else ""
            if size == 9:
                left = "2px solid #61dafb" if j%3==0 else "1px solid rgba(97,218,251,0.14)"
                top  = "2px solid #61dafb" if i%3==0 else "1px solid rgba(97,218,251,0.14)"
            else:
                left = "2px solid #61dafb" if j%3==0 else "1px solid rgba(97,218,251,0.14)"
                top  = "2px solid #61dafb" if i%2==0 else "1px solid rgba(97,218,251,0.14)"
            cell_style = f"border-left:{left}; border-top:{top}; padding:4px; width:34px; height:34px; text-align:center; font-weight:700; {style}"
            html += f"<td style='{cell_style}'>{val if val!='' else ''}</td>"
        html += "</tr>"
    html += "</table>"
    st.write(html, unsafe_allow_html=True)

# -------- Session initialization ----------
if 'last_extracted' not in st.session_state: st.session_state['last_extracted'] = None
if 'last_n' not in st.session_state: st.session_state['last_n'] = None
if 'last_solution' not in st.session_state: st.session_state['last_solution'] = None
if 'last_action' not in st.session_state: st.session_state['last_action'] = None

# -------- UI: manual override control ----------
mode_label = st.selectbox("Grid detection mode:", ("Auto-detect (recommended)", "6 × 6 (force)", "9 × 9 (force)"))
uploaded = st.file_uploader("Upload Sudoku image (tight crop or screenshot works best)", type=["png","jpg","jpeg"])

if uploaded:
    start = time.time()
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    pts = find_largest_quad(gray)
    if pts is None:
        st.error("Could not find grid boundary. Try cropping closer or improving contrast.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        warp = warp_to_square(gray, pts, min_side=220)

        # Auto-detect or manual
        if mode_label.startswith("Auto"):
            # sample scoring
            scores = sample_scores(warp, samples_per_dim=3)
            # pick higher score; if tie prefer 9
            s6 = scores.get(6, 0); s9 = scores.get(9, 0)
            est = 9 if s9 >= s6 else 6
            # then run full OCR for est
            grid, ratio, detected = ocr_full_for_size(warp, est)
            # if ratio very low, try the other size too and compare
            if ratio < 0.18:
                other = 6 if est == 9 else 9
                grid_o, ratio_o, det_o = ocr_full_for_size(warp, other)
                # pick the better ratio (with slight bias to 9)
                if ratio_o > ratio + 0.05:
                    n = other; grid = grid_o; ratio = ratio_o; detected = det_o
                else:
                    n = est
            else:
                n = est
        elif mode_label.startswith("6"):
            n = 6
            grid, ratio, detected = ocr_full_for_size(warp, 6)
        else:
            n = 9
            grid, ratio, detected = ocr_full_for_size(warp, 9)

        st.info(f"Detected: {n}×{n} • OCR filled: {ratio*100:.1f}% ({detected}/{n*n})")

        # store session extracted grid
        st.session_state['last_extracted'] = grid
        st.session_state['last_n'] = n
        st.session_state['last_solution'] = None
        st.session_state['last_action'] = None

        # show extracted grid preview
        display_grid_strings([[str(v) if v!=0 else "" for v in row] for row in grid], title="Extracted Grid (OCR preview)")
        st.markdown("<div class='note'>If numbers are wrong or missing, edit the grid below before solving.</div>", unsafe_allow_html=True)

        # ---- Manual-edit UI: create numeric inputs in a grid ----
        st.write("### Edit / Confirm Grid (click any cell to correct)")
        edit_cols = []
        edited = [[0]*n for _ in range(n)]
        # Use a form to group inputs and an Apply button
        with st.form(key="grid_edit_form"):
            for i in range(n):
                cols = st.columns(n)
                for j in range(n):
                    default = grid[i][j] if grid[i][j] != 0 else 0
                    key = f"cell_{i}_{j}"
                    # number_input with empty displayed as 0; user can set 0 to mean empty
                    val = cols[j].number_input(label="", min_value=0, max_value=n, value=default, key=key, step=1, format="%d")
                    edited[i][j] = int(val)
            apply = st.form_submit_button("Apply Edits")

        if apply:
            # convert zeros back to 0 and values > n to 0
            for i in range(n):
                for j in range(n):
                    v = edited[i][j]
                    if v < 1 or v > n:
                        edited[i][j] = 0
            st.session_state['last_extracted'] = edited
            st.success("Edits applied. You can now Solve or request a Hint.")

        # show the (possibly edited) grid
        cur_grid = st.session_state['last_extracted']
        display_grid_strings([[str(v) if v!=0 else "" for v in row] for row in cur_grid], title="Grid to Solve")

        # Solve / Hint / Reset buttons
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            if st.button("Solve Puzzle"):
                board = [row[:] for row in st.session_state['last_extracted']]
                br, bc = box_shape(n)
                # sanitize
                for i in range(n):
                    for j in range(n):
                        v = board[i][j]
                        if not (1 <= v <= n):
                            board[i][j] = 0
                with st.spinner("Solving..."):
                    ok = solve(board, n, br, bc)
                if ok:
                    # display solution
                    display_grid_strings([[format(v,"X") if (n==16 and v>9) else (str(v) if v!=0 else "") for v in row] for row in board], title="Solved Grid")
                    st.session_state['last_solution'] = board
                    st.session_state['last_action'] = 'solved'
                    st.success("Solved ✅")
                else:
                    st.error("Could not solve — check your edits or upload a clearer image.")

        with c2:
            if st.button("Hint"):
                # ensure solution exists
                if st.session_state.get('last_solution') is None:
                    board = [row[:] for row in st.session_state['last_extracted']]
                    br, bc = box_shape(n)
                    for i in range(n):
                        for j in range(n):
                            v = board[i][j]
                            if not (1 <= v <= n):
                                board[i][j] = 0
                    with st.spinner("Computing solution for hint..."):
                        ok = solve(board, n, br, bc)
                    if not ok:
                        st.error("Cannot provide hint — puzzle unsolvable (likely OCR/edit errors).")
                    else:
                        st.session_state['last_solution'] = board
                if st.session_state.get('last_solution') is not None:
                    sol = st.session_state['last_solution']
                    found = False
                    for i in range(n):
                        for j in range(n):
                            if st.session_state['last_extracted'][i][j] == 0:
                                val = sol[i][j]
                                display_grid_strings([[format(v,"X") if (n==16 and v>9) else (str(v) if v!=0 else "") for v in row] for row in sol], title="Solved Grid (Hint Highlighted)", highlight=(i,j))
                                st.info(f"Hint → place **{val}** at row {i+1}, column {j+1}")
                                st.session_state['last_action'] = 'hint'
                                found = True
                                break
                        if found: break
                    if not found:
                        st.success("No empty cells — puzzle already complete.")

        with c3:
            if st.button("Reset"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.rerun()

        elapsed = time.time() - start
        st.markdown(f"<div class='note'>Processed in {elapsed:.2f} s.</div>", unsafe_allow_html=True)

        # After an action, show Continue / Upload new as buttons
        if st.session_state.get('last_action') in ('solved', 'hint'):
            st.markdown("---")
            st.subheader("What would you like to do next?")
            left, right = st.columns(2)
            with left:
                if st.button("Continue with current puzzle", key="continue_btn"):
                    st.success("Continuing with current puzzle...")
                    st.rerun()
            with right:
                if st.button("Upload a new puzzle", key="upload_btn"):
                    for k in list(st.session_state.keys()):
                        del st.session_state[k]
                    st.rerun()

else:
    st.write("Upload a 6×6 or 9×9 Sudoku image to start. Use tight crop or a screenshot for best OCR results.")

st.markdown("</div>", unsafe_allow_html=True)
