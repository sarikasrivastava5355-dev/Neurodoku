# app.py — FINAL NEURODOKU
# Supports 6x6, 9x9, 16x16 using Tesseract OCR only
# Blue Tech UI, Solve / Hint, and Continue / Upload-new flow

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pytesseract
import pandas as pd
import re
import math

# Tesseract path for Streamlit Cloud
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="NEURODOKU", layout="centered")

# ---------- Blue Tech CSS ----------
st.markdown("""
    <style>
    body { background: linear-gradient(135deg, #0e1a2b, #112b45); color: #e6eef8; font-family: 'Segoe UI',sans-serif;}
    .main-title{font-size:38px; font-weight:800; text-align:center;
        background:linear-gradient(90deg,#61dafb,#1e88e5); -webkit-background-clip:text; color:transparent;}
    .subtitle {text-align:center; color:#9fd6ff; margin-bottom:18px;}
    .card { background: rgba(255,255,255,0.03); padding:18px; border-radius:14px; box-shadow:0 8px 24px rgba(3, 169, 244, 0.06); border:1px solid rgba(30,136,229,0.08); }
    .stButton>button { background: linear-gradient(90deg,#0288d1,#0277bd); color:white; border-radius:10px; padding:8px 16px; font-weight:700; }
    table { border-collapse:collapse; margin-top:12px; }
    td { width:40px; height:40px; text-align:center; font-weight:700; font-size:18px; color:#e8f6ff; border:1px solid rgba(30,136,229,0.25); }
    .highlight { background: #ffd54f; color:#001; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🧠 NEURODOKU</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload Sudoku image — Supports 6×6, 9×9, 16×16 • OCR Powered</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)

# file uploader appears at top; we'll show it when needed
uploaded = st.file_uploader("Upload Sudoku image (photo / screenshot)", type=["png","jpg","jpeg"])

# ---------- Utility / OCR / Solver functions ----------

def find_largest_contour(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edge = cv2.Canny(blur, 50, 150)
    cnts,_ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4:
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

def warp_to_square(gray, pts):
    rect = order_pts(pts)
    (tl,tr,br,bl) = rect
    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    side = int(max(widthA,widthB,heightA,heightB))
    if side < 100:
        side = max(100, side)
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]],dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(gray, M, (side,side))
    return warp

def ocr_cell(img, whitelist):
    # Accepts grayscale cell image
    if len(img.shape)==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Try adaptive threshold and also inverted variant to maximize detection
    try:
        _,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    except:
        th = img
    # Prepare canvas for tesseract
    h,w = th.shape
    size = 64
    canvas = 255 * np.ones((size,size), dtype=np.uint8)
    # Resize preserve aspect
    scale = min((size-8)/w, (size-8)/h)
    nw = max(1, int(w*scale))
    nh = max(1, int(h*scale))
    resized = cv2.resize(th, (nw,nh))
    xoff = (size-nw)//2
    yoff = (size-nh)//2
    canvas[yoff:yoff+nh, xoff:xoff+nw] = resized
    pil = Image.fromarray(canvas)
    config = f'--psm 10 -c tessedit_char_whitelist={whitelist}'
    txt = pytesseract.image_to_string(pil, config=config)
    txt = re.sub(r'[^0-9A-Za-z]', '', txt)
    return txt.strip()

def test_sizes(warp, sizes=(6,9,16)):
    # For each candidate size try to OCR each cell; return best size/grid/ratio
    best = None
    side = warp.shape[0]
    for n in sizes:
        cell = side // n
        grid = [[0]*n for _ in range(n)]
        count = 0
        for i in range(n):
            for j in range(n):
                y = i*cell; x = j*cell
                m = max(4, cell//12)
                y1 = y+m; y2 = min(y+cell-m, side)
                x1 = x+m; x2 = min(x+cell-m, side)
                if y2<=y1 or x2<=x1:
                    crop = warp[y:y+cell, x:x+cell]
                else:
                    crop = warp[y1:y2, x1:x2]
                whitelist = "0123456789ABCDEFabcdef" if n==16 else "0123456789"
                txt = ocr_cell(crop, whitelist)
                if txt != "":
                    if n==16:
                        ch = txt[0].upper()
                        try:
                            val = int(ch, 16)
                        except:
                            val = 0
                    else:
                        try:
                            val = int(txt[0])
                        except:
                            val = 0
                    if 1 <= val <= n:
                        grid[i][j] = val
                        count += 1
        ratio = count / (n*n)
        if best is None or ratio > best[2]:
            best = (n, grid, ratio)
    return best

def convert_to_strings(grid, n):
    out = []
    for r in grid:
        row = []
        for v in r:
            if v == 0:
                row.append("")
            else:
                if n==16 and v>9:
                    row.append(format(v, "X"))
                else:
                    row.append(str(v))
        out.append(row)
    return out

def box_shape(n):
    if n == 6: return 2,3
    root = int(math.sqrt(n))
    if root*root == n:
        return root,root
    for r in range(root,0,-1):
        if n%r==0:
            return r, n//r
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
    rs = (r//br)*br
    cs = (c//bc)*bc
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

def display_grid_strings(strings, title="Grid", highlight=None):
    size = len(strings)
    st.write(f"### {title}")
    html = "<table>"
    for i in range(size):
        html += "<tr>"
        for j in range(size):
            val = strings[i][j]
            is_h = highlight and (i,j) == highlight
            style = ""
            if is_h:
                style = "background:#ffd54f;color:#001;"
            # box border thickness rules for nicer look
            # for 9x9 we mark every 3, for 6x6 every 2/3, for 16x16 every 4
            if size == 9:
                left = "2px solid #61dafb" if j % 3 == 0 else "1px solid rgba(97,218,251,0.18)"
                top  = "2px solid #61dafb" if i % 3 == 0 else "1px solid rgba(97,218,251,0.18)"
            elif size == 6:
                left = "2px solid #61dafb" if j % 3 == 0 else "1px solid rgba(97,218,251,0.18)"
                top  = "2px solid #61dafb" if i % 2 == 0 else "1px solid rgba(97,218,251,0.18)"
            else:
                left = "2px solid #61dafb" if j % 4 == 0 else "1px solid rgba(97,218,251,0.18)"
                top  = "2px solid #61dafb" if i % 4 == 0 else "1px solid rgba(97,218,251,0.18)"
            cell_style = f"border-left:{left}; border-top:{top}; padding:6px; width:40px; height:40px; text-align:center; font-weight:700; {style}"
            html += f"<td style='{cell_style}'>{val}</td>"
        html += "</tr>"
    html += "</table>"
    st.write(html, unsafe_allow_html=True)


# ---------- Main App Flow ----------

# Ensure session state keys exist
if 'last_action' not in st.session_state:
    st.session_state['last_action'] = None
if 'last_solution' not in st.session_state:
    st.session_state['last_solution'] = None
if 'last_extracted' not in st.session_state:
    st.session_state['last_extracted'] = None
if 'last_n' not in st.session_state:
    st.session_state['last_n'] = None

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and detect grid
    arr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

    pts = find_largest_contour(gray)
    if pts is None:
        st.error("Could not detect Sudoku grid. Try cropping closer to the grid or increasing contrast.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        warp = warp_to_square(gray, pts)

        # Determine best grid size and extracted grid
        n, grid_guess, ratio = test_sizes(warp, sizes=(6,9,16))
        st.info(f"Detected candidate: {n}×{n} • detected {ratio*100:.1f}% of cells")

        # Save into session so we can continue with same puzzle
        st.session_state['last_extracted'] = grid_guess
        st.session_state['last_n'] = n
        # Reset last_solution when new upload happens
        st.session_state['last_solution'] = None
        st.session_state['last_action'] = None

        # Show extracted grid (all detected digits)
        grid_display = convert_to_strings(grid_guess, n)
        display_grid_strings(grid_display, title="Extracted Grid (OCR result)")

        if ratio < 0.12:
            st.warning("OCR detected very few digits. Try a clearer or more tightly-cropped image. Solver disabled until enough digits are detected.")
        else:
            # Buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Solve Puzzle"):
                    board = [row[:] for row in grid_guess]
                    br, bc = box_shape(n)
                    # sanitize board values
                    for i in range(n):
                        for j in range(n):
                            v = board[i][j]
                            if not (1 <= v <= n):
                                board[i][j] = 0
                    with st.spinner("Solving..."):
                        ok = solve(board, n, br, bc)
                    if ok:
                        sol_display = convert_to_strings(board, n)
                        display_grid_strings(sol_display, title="Solved Grid")
                        st.session_state['last_solution'] = board
                        st.session_state['last_action'] = 'solved'
                        st.success("Solved ✅")
                    else:
                        st.error("Could not solve the puzzle — maybe OCR mistakes. Try a clearer image or correct OCR manually later.")

            with col2:
                if st.button("Hint"):
                    # ensure solution exists
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
                            st.error("Cannot compute hint — puzzle unsolvable (likely OCR errors).")
                        else:
                            st.session_state['last_solution'] = board

                    if st.session_state.get('last_solution') is not None:
                        sol = st.session_state['last_solution']
                        # find first empty cell from extracted
                        found = False
                        for i in range(n):
                            for j in range(n):
                                if grid_guess[i][j] == 0:
                                    val = sol[i][j]
                                    # show solved with highlighted hint
                                    sol_disp = convert_to_strings(sol, n)
                                    display_grid_strings(sol_disp, title="Solved Grid (Hint Highlighted)", highlight=(i,j))
                                    val_disp = format(val, 'X') if n==16 and val>9 else val
                                    st.info(f"Hint → place **{val_disp}** at row {i+1}, column {j+1}")
                                    st.session_state['last_action'] = 'hint'
                                    found = True
                                    break
                            if found:
                                break
                        if not found:
                            st.success("No empty cells — puzzle appears complete.")

            with col3:
                if st.button("Reset"):
                    for k in list(st.session_state.keys()):
                        del st.session_state[k]
                    st.rerun()

        # After action (hint or solve) ask user what to do next
        if st.session_state.get('last_action') in ('solved', 'hint'):
            st.markdown("---")
            st.subheader("What would you like to do next?")
            choice = st.radio("Choose:", ["Continue with current puzzle", "Upload a new puzzle"], key="next_choice")
            if st.button("Proceed"):
                if choice == "Continue with current puzzle":
                    # Keep session_state values and rerun so user can re-use buttons
                    st.success("Continuing with the current puzzle...")
                    st.rerun()
                else:
                    # Clear everything and show uploader
                    for k in list(st.session_state.keys()):
                        del st.session_state[k]
                    st.rerun()
else:
    st.write("Upload a Sudoku image to start (tight crop of the grid gives best OCR results).")

st.markdown("</div>", unsafe_allow_html=True)
