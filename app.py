# app.py — NEURODOKU (Upgraded OCR + optional CNN digit recognizer)
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import cv2
import pytesseract
import pandas as pd
import re
import os

# Optional: use a CNN model if available for digit recognition
USE_CNN_IF_AVAILABLE = True
CNN_MODEL_PATH = "mnist_cnn.h5"  # Put trained model here (see Colab training code below)

# Tesseract path (Streamlit Cloud)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="NEURODOKU - Upgraded", layout="centered")

st.markdown("""
    <style>
    body { background-color: #f5f5dc; }
    .title { font-size:36px; color:#4B3832; text-align:center; font-weight:700; }
    .subtitle { font-size:14px; color:#6F4E37; text-align:center; margin-bottom:10px; }
    .grid-cell { font-size:18px; text-align:center; font-weight:bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🧠 NEURODOKU</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload a Sudoku image. Uses OCR + optional CNN for digit recognition.</div>", unsafe_allow_html=True)
st.write("---")

# ---------------------------
# Load optional CNN model if present
# ---------------------------
cnn_model = None
if USE_CNN_IF_AVAILABLE and os.path.exists(CNN_MODEL_PATH):
    try:
        from tensorflow.keras.models import load_model
        cnn_model = load_model(CNN_MODEL_PATH)
        st.success("Loaded CNN model for digit recognition.")
    except Exception as e:
        st.warning(f"Could not load CNN model ({CNN_MODEL_PATH}): {e}")
        cnn_model = None

# ---------------------------
# Sudoku solver (9x9)
# ---------------------------
def find_empty(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                return i, j
    return None

def is_valid(board, row, col, num):
    for x in range(9):
        if board[row][x] == num or board[x][col] == num:
            return False
    br = (row // 3) * 3
    bc = (col // 3) * 3
    for r in range(br, br+3):
        for c in range(bc, bc+3):
            if board[r][c] == num:
                return False
    return True

def solve_board(board):
    empty = find_empty(board)
    if not empty:
        return True
    r, c = empty
    for num in range(1, 10):
        if is_valid(board, r, c, num):
            board[r][c] = num
            if solve_board(board):
                return True
            board[r][c] = 0
    return False

# ---------------------------
# Utility: show grid nicely
# ---------------------------
def show_grid(grid, title="Grid", highlight=None):
    df = pd.DataFrame(grid)
    st.write(f"### {title}")
    # Create styled HTML table manually to allow highlighting
    cell_style = "border:1px solid #999; width:36px; height:36px; text-align:center; font-size:18px; font-weight:bold;"
    highlight_style = "background-color:#ffd54f;"  # yellowish
    html = "<table style='border-collapse:collapse'>"
    for i in range(9):
        html += "<tr>"
        for j in range(9):
            val = "" if grid[i][j] == 0 else str(grid[i][j])
            style = cell_style
            # heavier border every 3 cells
            left = "2px solid #333" if j % 3 == 0 else "1px solid #999"
            top = "2px solid #333" if i % 3 == 0 else "1px solid #999"
            style = style.replace("border:1px solid #999;", f"border-left:{left}; border-top:{top};")
            if (i,j) == highlight:
                style += highlight_style
            html += f"<td style='{style}'>{val}</td>"
        html += "</tr>"
    html += "</table>"
    st.write(html, unsafe_allow_html=True)

# ---------------------------
# OCR & grid extraction functions
# ---------------------------
def order_points(pts):
    # order rect points TL, TR, BR, BL
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def warp_grid(img, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warp = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
    return warp

def detect_largest_square(contours):
    # return contour approximated to 4 points of largest square-like contour
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            return approx.reshape(4,2)
    return None

def extract_grid_from_image(image_pil):
    img = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # increase contrast
    gray = cv2.equalizeHist(gray)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        st.warning("No contours found in image. Try a clearer photo or crop tightly.")
        return None
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    grid_pts = detect_largest_square(contours)
    if grid_pts is None:
        st.warning("Could not detect Sudoku grid boundary. Try a brighter, more contrasty image.")
        return None
    warp = warp_grid(gray, grid_pts)
    # ensure square by padding/cropping
    h, w = warp.shape
    side = min(h,w)
    warp = cv2.resize(warp, (side, side))
    # enhance
    warp = cv2.adaptiveThreshold(warp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY_INV, 11, 2)
    # Now split into 9x9
    cell_h = side // 9
    cell_w = side // 9
    grid = [[0]*9 for _ in range(9)]
    for i in range(9):
        for j in range(9):
            y1 = i*cell_h
            x1 = j*cell_w
            cell = warp[y1:y1+cell_h, x1:x1+cell_w]
            # clean cell: remove borders/lines by clearing outer pixels
            margin = int(cell_h*0.12)
            cropped = cell[margin:cell_h-margin, margin:cell_w-margin]
            # if mostly empty skip
            if cv2.countNonZero(cropped) < (cropped.size * 0.02):
                grid[i][j] = 0
                continue
            # prep for recognition
            digit = recognize_digit(cropped)
            grid[i][j] = digit
    return grid

# ---------------------------
# Digit recognition: either CNN (if loaded) or Tesseract
# ---------------------------
def recognize_digit(cell_img):
    """Given binary image of a cell (white foreground digits on black background), return int digit or 0."""
    # Resize to 28x28 for CNN
    try:
        # Ensure cell has correct dtype
        cell = cell_img.copy()
        cell = cv2.resize(cell, (28,28))
        # Invert for keras (white background) if needed
        # Normalize
        if cnn_model is not None:
            inp = cell.astype("float32") / 255.0
            inp = np.expand_dims(inp, -1)
            inp = np.expand_dims(inp, 0)  # batch
            pred = cnn_model.predict(inp, verbose=0)
            val = int(np.argmax(pred, axis=1)[0])
            # filter low-confidence: check max prob
            if pred[0].max() < 0.6:
                # fallback to tesseract
                return tesseract_digit(cell)
            return val
        else:
            return tesseract_digit(cell)
    except Exception:
        return 0

def tesseract_digit(img28):
    # Tesseract expects black-on-white for digits; invert and scale
    img = cv2.resize(img28, (50,50))
    inv = cv2.bitwise_not(img)
    pil = Image.fromarray(inv)
    # run tesseract single char
    text = pytesseract.image_to_string(pil, config='--psm 10 -c tessedit_char_whitelist=0123456789')
    text = re.sub('[^0-9]', '', text)
    if text.isdigit():
        d = int(text)
        if d == 0:
            return 0
        return d
    return 0

# ---------------------------
# UI: Upload, extract, solve, hint
# ---------------------------
uploaded = st.file_uploader("Upload Sudoku (photo/screenshot)", type=["png","jpg","jpeg"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    # Extract grid
    with st.spinner("Detecting grid and extracting digits..."):
        grid = extract_grid_from_image(image)
    if grid is None:
        st.error("Failed to extract grid. Try a clearer image or crop closer to the puzzle.")
    else:
        show_grid(grid, "Extracted Grid")
        st.session_state["extracted"] = grid

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Solve"):
                puzzle = [row[:] for row in grid]
                if solve_board(puzzle):
                    st.success("Solved!")
                    show_grid(puzzle, "Solved Grid")
                    st.session_state["solution"] = puzzle
                else:
                    st.error("Could not solve the puzzle (maybe OCR errors). Try re-uploading a clearer image.")
        with col2:
            if st.button("Get Hint"):
                if "solution" not in st.session_state:
                    # try to solve to get hints
                    puzzle = [row[:] for row in grid]
                    if not solve_board(puzzle):
                        st.error("Cannot provide hint — puzzle unsolvable (maybe OCR errors).")
                    else:
                        st.session_state["solution"] = puzzle
                if "solution" in st.session_state:
                    sol = st.session_state["solution"]
                    # find first empty in extracted grid
                    found = False
                    for i in range(9):
                        for j in range(9):
                            if grid[i][j] == 0:
                                hint_val = sol[i][j]
                                st.info(f"Hint: place **{hint_val}** at row {i+1}, column {j+1}")
                                # show solved grid with highlighted cell
                                show_grid(sol, "Solved Grid (hint highlighted)", highlight=(i,j))
                                found = True
                                break
                        if found:
                            break
                    if not found:
                        st.info("No empty cells — puzzle already complete.")
        with col3:
            if st.button("Reset"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.experimental_rerun()
else:
    st.write("Upload a Sudoku image to start.")

st.write("---")
st.markdown("**Notes:** For best OCR accuracy: upload a clear, top-down crop of the grid with high contrast. To improve accuracy further, run the optional CNN training in Colab (code shown below) and upload `mnist_cnn.h5` to the app folder.")              
                
