# app.py — NEURODOKU final (stable multi-grid: 6x6, 9x9, 16x16)
# Uses Tesseract-only OCR, robust preprocessing, improved size-detection heuristics,
# and button-style "Continue" / "Upload new puzzle" flow.

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pytesseract
import re
import math

# Tesseract path (Streamlit Cloud)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="NEURODOKU", layout="centered")

# ------------------ Blue Tech CSS ------------------
st.markdown("""
    <style>
    body { background: linear-gradient(135deg, #0e1a2b, #112b45); color: #e6eef8; font-family: 'Segoe UI',sans-serif;}
    .main-title{font-size:38px; font-weight:800; text-align:center;
        background:linear-gradient(90deg,#61dafb,#1e88e5); -webkit-background-clip:text; color:transparent;}
    .subtitle {text-align:center; color:#9fd6ff; margin-bottom:18px;}
    .card { background: rgba(255,255,255,0.03); padding:18px; border-radius:14px; box-shadow:0 8px 24px rgba(3, 169, 244, 0.06); border:1px solid rgba(30,136,229,0.08); }
    .big-btn > button { background: linear-gradient(90deg,#00b0ff,#0288d1); color:white; border-radius:10px; padding:12px 18px; font-weight:700; font-size:16px; }
    .small-btn > button { background: linear-gradient(90deg,#0288d1,#0277bd); color:white; border-radius:8px; padding:8px 12px; font-weight:700; }
    table { border-collapse:collapse; margin-top:12px; }
    td { width:40px; height:40px; text-align:center; font-weight:700; font-size:18px; color:#e8f6ff; border:1px solid rgba(30,136,229,0.25); }
    .highlight { background: #ffd54f; color:#001; }
    .note { color:#bfe9ff; font-size:14px; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🧠 NEURODOKU</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Auto-detects 6×6 / 9×9 / 16×16 — Robust OCR for screenshots & phone photos</div>", unsafe_allow_html=True)
st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Sudoku image (tight crop preferred)", type=["png","jpg","jpeg"])

# ------------------ Helper funcs ------------------

def safe_gray(img):
    if img is None: return None
    try:
        if len(img.shape) == 3:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    except:
        return None

def find_largest_contour(gray):
    if gray is None: return None
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 150)
    cnts,_ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts: return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
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

def warp_to_square(gray, pts, min_side=200):
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
    warp = cv2.warpPerspective(gray, M, (side, side))
    return warp

# Robust OCR attempt: multiple preprocess attempts, safe canvas, returns cleaned string
def ocr_cell(img_cell, whitelist):
    if img_cell is None or not hasattr(img_cell, "shape") or img_cell.size == 0:
        return ""
    if len(img_cell.shape) == 3:
        try:
            img_cell = cv2.cvtColor(img_cell, cv2.COLOR_BGR2GRAY)
        except:
            return ""
    h,w = img_cell.shape
    if h < 6 or w < 6:
        return ""
    attempts = []
    try:
        attempts.append(cv2.adaptiveThreshold(img_cell, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2))
    except:
        pass
    try:
        _, o = cv2.threshold(img_cell, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        attempts.append(o)
    except:
        pass
    # invert attempts
    for a in attempts[:]:
        try: attempts.append(cv2.bitwise_not(a))
        except: pass
    try:
        b = cv2.GaussianBlur(img_cell, (3,3), 0)
        _, bth = cv2.threshold(b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        attempts.append(bth); attempts.append(cv2.bitwise_not(bth))
    except:
        pass
    try:
        _, raw = cv2.threshold(img_cell, 128, 255, cv2.THRESH_BINARY)
        attempts.append(raw); attempts.append(cv2.bitwise_not(raw))
    except:
        pass

    size = 64
    for att in attempts:
        try:
            ah, aw = att.shape
            canvas = 255 * np.ones((size,size), dtype=np.uint8)
            scale = min((size-8)/max(1,aw), (size-8)/max(1,ah))
            nw = max(1, int(aw*scale)); nh = max(1, int(ah*scale))
            resized = cv2.resize(att, (nw, nh))
            xoff = (size-nw)//2; yoff = (size-nh)//2
            canvas[yoff:yoff+nh, xoff:xoff+nw] = resized
            pil = Image.fromarray(canvas)
        except:
            continue
        try:
            cfg = f'--psm 10 -c tessedit_char_whitelist={whitelist}'
            txt = pytesseract.image_to_string(pil, config=cfg)
            txt = re.sub(r'[^0-9A-Za-z]', '', txt).strip()
            if txt != "":
                return txt
        except:
            continue
    return ""

# Improved size detection: returns (n, grid, ratio, digit_count, hex_count)
def test_sizes_with_stats(warp, sizes=(6,9,16)):
    best = None
    side = warp.shape[0]
    for n in sizes:
        cell = side // n
        grid = [[0]*n for _ in range(n)]
        count = 0
        hex_count = 0
        digit_count = 0
        for i in range(n):
            for j in range(n):
                y = i*cell; x = j*cell
                m = max(4, cell//12)
                y1 = max(0, y+m); y2 = min(side, y+cell-m)
                x1 = max(0, x+m); x2 = min(side, x+cell-m)
                if y2<=y1 or x2<=x1:
                    crop = warp[y:y+cell, x:x+cell]
                else:
                    crop = warp[y1:y2, x1:x2]
                whitelist = "0123456789ABCDEFabcdef" if n==16 else "0123456789"
                txt = ocr_cell(crop, whitelist)
                if txt:
                    if n==16:
                        ch = txt[0].upper()
                        try:
                            val = int(ch, 16)
                        except:
                            val = 0
                        # count hex letters specifically
                        if re.match(r'[A-Fa-f]', txt[0]):
                            hex_count += 1
                        elif txt[0].isdigit():
                            digit_count += 1
                    else:
                        try:
                            val = int(txt[0])
                            digit_count += 1
                        except:
                            val = 0
                    if 1 <= val <= n:
                        grid[i][j] = val
                        count += 1
        ratio = count / (n*n)
        # store extra stats
        stats = (n, grid, ratio, digit_count, hex_count)
        if best is None or ratio > best[2]:
            best = stats
    # apply heuristic: if 16x16 chosen but hex_count is tiny and digit_count suggests 9x9 better, prefer 9x9
    if best and best[0] == 16:
        # find 9x9 stats
        nine = None
        for s in [test for test in [best] if True]:
            pass
        # compute second best manually by re-evaluating nine and six
        # simpler: recompute stats for 9 and 6 quickly
    # compute lightweight stats for 9 and 6 to compare
    stats6 = None; stats9 = None; stats16 = None
    for n in (6,9,16):
        side = warp.shape[0]; cell = side//n
        grid = [[0]*n for _ in range(n)]
        count=0; digit_count=0; hex_count=0
        for i in range(n):
            for j in range(n):
                y = i*cell; x = j*cell
                m = max(4, cell//12)
                y1 = max(0, y+m); y2 = min(side, y+cell-m)
                x1 = max(0, x+m); x2 = min(side, x+cell-m)
                if y2<=y1 or x2<=x1:
                    crop = warp[y:y+cell, x:x+cell]
                else:
                    crop = warp[y1:y2, x1:x2]
                wl = "0123456789ABCDEFabcdef" if n==16 else "0123456789"
                txt = ocr_cell(crop, wl)
                if txt:
                    if n==16:
                        ch = txt[0].upper()
                        try: v=int(ch,16)
                        except: v=0
                        if re.match(r'[A-Fa-f]', txt[0]): hex_count+=1
                        elif txt[0].isdigit(): digit_count+=1
                    else:
                        try: v=int(txt[0]); digit_count+=1
                        except: v=0
                    if 1<=v<=n:
                        grid[i][j]=v; count+=1
        r = count/(n*n)
        if n==6: stats6=(n,grid,r,digit_count,hex_count)
        if n==9: stats9=(n,grid,r,digit_count,hex_count)
        if n==16: stats16=(n,grid,r,digit_count,hex_count)
    # Now decide:
    # Prefer 9x9 unless 16x16 has significant hex_count or much higher ratio
    if stats16:
        n16, g16, r16, d16, h16 = stats16
    else:
        r16 = -1; h16 = 0
    if stats9:
        n9, g9, r9, d9, h9 = stats9
    else:
        r9 = -1
    if stats6:
        n6, g6, r6, d6, h6 = stats6
    else:
        r6 = -1
    # Heuristics:
    # If hex letters present (>2) then 16 is likely real
    if h16 >= 2 and r16 > 0.05:
        return (16, g16, r16)
    # else prefer 9 if its ratio is close or better
    # require 16 ratio to be significantly higher to pick it
    if r9 >= r16 - 0.12:
        # pick the best among 9 and 6 by ratio
        if r9 >= r6:
            return (9, g9, r9)
        else:
            return (6, g6, r6)
    else:
        # 16 was clearly better
        return (16, g16, r16)

def convert_to_strings(grid, n):
    out=[]
    for row in grid:
        r=[]
        for v in row:
            if v==0: r.append("")
            else:
                if n==16 and v>9: r.append(format(v,"X"))
                else: r.append(str(v))
        out.append(r)
    return out

def box_shape(n):
    if n==6: return 2,3
    root=int(math.sqrt(n))
    if root*root==n: return root,root
    for r in range(root,0,-1):
        if n%r==0: return r, n//r
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
    for i in range(rs,rs+br):
        for j in range(cs,cs+bc):
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

def display_grid_strings(strings, title="Grid", highlight=None):
    st.write(f"### {title}")
    size=len(strings)
    html="<table>"
    for i in range(size):
        html+="<tr>"
        for j in range(size):
            val=str(strings[i][j]) if strings[i][j] is not None else ""
            is_h = highlight and (i,j)==highlight
            style = "background:#ffd54f;color:#001;" if is_h else ""
            if size==9:
                left = "2px solid #61dafb" if j%3==0 else "1px solid rgba(97,218,251,0.18)"
                top  = "2px solid #61dafb" if i%3==0 else "1px solid rgba(97,218,251,0.18)"
            elif size==6:
                left = "2px solid #61dafb" if j%3==0 else "1px solid rgba(97,218,251,0.18)"
                top  = "2px solid #61dafb" if i%2==0 else "1px solid rgba(97,218,251,0.18)"
            else:
                left = "2px solid #61dafb" if j%4==0 else "1px solid rgba(97,218,251,0.18)"
                top  = "2px solid #61dafb" if i%4==0 else "1px solid rgba(97,218,251,0.18)"
            cell_style = f"border-left:{left}; border-top:{top}; padding:6px; width:36px; height:36px; text-align:center; font-weight:700; {style}"
            html += f"<td style='{cell_style}'>{val}</td>"
        html += "</tr>"
    html += "</table>"
    st.write(html, unsafe_allow_html=True)

# Session keys
if 'last_extracted' not in st.session_state: st.session_state['last_extracted'] = None
if 'last_n' not in st.session_state: st.session_state['last_n'] = None
if 'last_solution' not in st.session_state: st.session_state['last_solution'] = None
if 'last_action' not in st.session_state: st.session_state['last_action'] = None

# ------------------ Main flow ------------------
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    pts = find_largest_contour(gray)
    if pts is None:
        st.error("Could not detect Sudoku grid. Try cropping closer or increasing contrast.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        warp = warp_to_square(gray, pts)
        n, grid_guess, ratio = test_sizes_with_stats(warp, sizes=(6,9,16))
        st.info(f"Candidate detection: {n}×{n} — {ratio*100:.1f}% filled")

        # store
        st.session_state['last_extracted'] = grid_guess
        st.session_state['last_n'] = n
        st.session_state['last_solution'] = None
        st.session_state['last_action'] = None

        grid_display = convert_to_strings(grid_guess, n)
        display_grid_strings(grid_display, title="Extracted Grid (OCR result)")
        st.markdown("<div class='note'>If many cells are empty, OCR likely failed. Try a tighter crop or higher contrast.</div>", unsafe_allow_html=True)

        if ratio < 0.12:
            st.warning("Very few digits detected → solver disabled until clearer image.")
        else:
            col1, col2, col3 = st.columns([1,1,1])
            with col1:
                if st.button("Solve Puzzle"):
                    board = [row[:] for row in grid_guess]
                    br, bc = box_shape(n)
                    for i in range(n):
                        for j in range(n):
                            v = board[i][j]
                            if not (1 <= v <= n):
                                board[i][j] = 0
                    with st.spinner("Solving..."):
                        ok = solve(board,n,br,bc)
                    if ok:
                        disp = convert_to_strings(board,n)
                        display_grid_strings(disp, title="Solved Grid")
                        st.session_state['last_solution'] = board
                        st.session_state['last_action'] = 'solved'
                        st.success("Solved ✅")
                    else:
                        st.error("Could not solve — OCR errors likely.")

            with col2:
                if st.button("Hint"):
                    if st.session_state.get('last_solution') is None:
                        board = [row[:] for row in grid_guess]
                        br, bc = box_shape(n)
                        for i in range(n):
                            for j in range(n):
                                v = board[i][j]
                                if not (1 <= v <= n):
                                    board[i][j] = 0
                        with st.spinner("Computing solution for hint..."):
                            ok = solve(board, n, br, bc)
                        if not ok:
                            st.error("Cannot provide hint — puzzle unsolvable (likely OCR mistakes).")
                        else:
                            st.session_state['last_solution'] = board

                    if st.session_state.get('last_solution') is not None:
                        sol = st.session_state['last_solution']
                        found=False
                        for i in range(n):
                            for j in range(n):
                                if grid_guess[i][j] == 0:
                                    val = sol[i][j]
                                    display_grid_strings(convert_to_strings(sol,n), title="Solved Grid (Hint Highlighted)", highlight=(i,j))
                                    vdisp = format(val,'X') if n==16 and val>9 else val
                                    st.info(f"Hint → place **{vdisp}** at row {i+1}, column {j+1}")
                                    st.session_state['last_action'] = 'hint'
                                    found=True
                                    break
                            if found: break
                        if not found:
                            st.success("Puzzle already complete.")

            with col3:
                if st.button("Reset"):
                    for k in list(st.session_state.keys()):
                        del st.session_state[k]
                    st.rerun()

        # Button-style next step (after solve/hint)
        if st.session_state.get('last_action') in ('solved','hint'):
            st.markdown("---")
            st.subheader("What would you like to do next?")
            left, right = st.columns([1,1])
            with left:
                if st.button("Continue with current puzzle", key="cont_btn", help="Keep current puzzle and go back to extracted grid"):
                    st.success("Continuing with current puzzle...")
                    st.rerun()
            with right:
                if st.button("Upload a new puzzle", key="new_btn", help="Clear current puzzle and upload new image"):
                    for k in list(st.session_state.keys()):
                        del st.session_state[k]
                    st.rerun()
else:
    st.write("Upload a Sudoku image to start (tight crop of the grid gives best OCR results).")

st.markdown("</div>", unsafe_allow_html=True)
