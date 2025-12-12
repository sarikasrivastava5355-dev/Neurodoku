import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
import matplotlib.pyplot as plt
import random

# ---------------- PAGE SETTINGS ----------------
st.set_page_config(page_title="NEURODOKU", layout="wide")

st.markdown("""
<style>
body {background-color: #f4f5f7;}
.stButton>button {
    background-color: #1f77b4;
    color: white;
    font-size: 16px;
    border-radius: 8px;
    padding: 8px 20px;
}
.title {font-size: 40px; text-align: center; font-weight: 700; color: #003366;}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGO & TITLE ----------------
colL, colM, colR = st.columns([1, 2, 1])
with colM:
    st.image("logo.png", width=180)  # Ensure logo is circular
    st.markdown("<div class='title'>NEURODOKU</div>", unsafe_allow_html=True)

st.write("")

# ---------------- SESSION STATE INIT ----------------
if "puzzle_selected" not in st.session_state:
    st.session_state.puzzle_selected = False
if "uploaded_image" not in st.session_state:
    st.session_state.uploaded_image = None
if "grid" not in st.session_state:
    st.session_state.grid = None
if "solved_grid" not in st.session_state:
    st.session_state.solved_grid = None

# ---------------- SELECT PUZZLE BUTTON ----------------
if not st.session_state.puzzle_selected:
    if st.button("9 × 9 Sudoku"):
        st.session_state.puzzle_selected = True
        st.experimental_rerun()

# ---------------- FILE UPLOAD ----------------
if st.session_state.puzzle_selected and st.session_state.uploaded_image is None:
    uploaded = st.file_uploader("Upload Sudoku image", type=["png","jpg","jpeg"])
    if uploaded:
        st.session_state.uploaded_image = uploaded
        st.experimental_rerun()

# ---------------- IMAGE DISPLAY & OCR ----------------
if st.session_state.uploaded_image:
    img = Image.open(st.session_state.uploaded_image).convert("RGB")
    img_np = np.array(img)
    st.image(img_np, caption="Uploaded Image", width=350)

    if st.session_state.grid is None:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5),0)
        thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV,11,3)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        grid_contour = None
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02*peri, True)
            if len(approx) == 4:
                grid_contour = approx
                break
        if grid_contour is None:
            st.error("❌ Could not detect puzzle grid.")
            st.stop()

        pts = grid_contour.reshape(4,2)
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        ordered = np.array([pts[np.argmin(s)], pts[np.argmin(diff)],
                            pts[np.argmax(s)], pts[np.argmax(diff)]], dtype="float32")

        side = 450
        dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype="float32")
        M = cv2.getPerspectiveTransform(ordered, dst)
        warp = cv2.warpPerspective(gray, M, (side, side))

        # OCR
        grid_size = 9
        cell = side // grid_size
        grid = np.zeros((grid_size,grid_size),dtype=int)
        for r in range(grid_size):
            for c in range(grid_size):
                crop = warp[r*cell:(r+1)*cell, c*cell:(c+1)*cell]
                crop = cv2.resize(crop, (60,60))
                crop = cv2.adaptiveThreshold(crop,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY,11,2)
                text = pytesseract.image_to_string(crop, 
                        config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789').strip()
                grid[r,c] = int(text) if text.isdigit() else 0
        st.session_state.grid = grid

# ---------------- DISPLAY GRID ----------------
def show_grid(grid, title="Extracted Grid", colors=None):
    fig, ax = plt.subplots(figsize=(5,5))
    ax.imshow(np.ones(grid.shape), cmap="gray", vmin=0, vmax=1)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            if grid[r,c]!=0:
                color = colors[r][c] if colors is not None else "blue"
                ax.text(c,r,str(grid[r,c]),ha="center",va="center",fontsize=16,color=color)
    ax.set_xticks(np.arange(-0.5, grid.shape[0],1))
    ax.set_yticks(np.arange(-0.5, grid.shape[1],1))
    ax.grid(color="black")
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    st.pyplot(fig)

if st.session_state.grid is not None:
    colors = [["blue" for _ in range(9)] for _ in range(9)]
    show_grid(st.session_state.grid, colors=colors)

# ---------------- SOLVER ----------------
def valid(board,r,c,x):
    for i in range(9):
        if board[r][i]==x or board[i][c]==x:
            return False
    br = (r//3)*3
    bc = (c//3)*3
    for i in range(3):
        for j in range(3):
            if board[br+i][bc+j]==x:
                return False
    return True

def solve(board):
    for i in range(9):
        for j in range(9):
            if board[i][j]==0:
                for x in range(1,10):
                    if valid(board,i,j,x):
                        board[i][j]=x
                        if solve(board):
                            return True
                        board[i][j]=0
                return False
    return True

# ---------------- HINT FUNCTION ----------------
def give_hint(grid, solved):
    empty_cells = [(r,c) for r in range(9) for c in range(9) if grid[r][c]==0]
    if not empty_cells:
        st.info("No empty cells left for hint!")
        return grid
    r,c = random.choice(empty_cells)
    grid[r][c] = solved[r][c]
    st.success(f"Hint: Filled cell at row {r+1}, column {c+1}")
    return grid

# ---------------- BUTTONS: Hint / Solve / Continue ----------------
if st.session_state.grid is not None:
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Hint"):
            if st.session_state.solved_grid is None:
                st.session_state.solved_grid = st.session_state.grid.copy()
                solve(st.session_state.solved_grid)
            st.session_state.grid = give_hint(st.session_state.grid.copy(), st.session_state.solved_grid)
            show_grid(st.session_state.grid, colors=[["blue" if st.session_state.grid[r][c]==st.session_state.solved_grid[r][c] else "green" for c in range(9)] for r in range(9)])

    with col2:
        if st.button("Full Solution"):
            if st.session_state.solved_grid is None:
                st.session_state.solved_grid = st.session_state.grid.copy()
                solve(st.session_state.solved_grid)
            show_grid(st.session_state.solved_grid, colors=[["green" for _ in range(9)] for _ in range(9)])

# ---------------- CONTINUE OR NEW PUZZLE ----------------
col1, col2 = st.columns(2)
with col1:
    if st.button("Continue with this puzzle"):
        st.experimental_rerun()

with col2:
    if st.button("Upload New Puzzle"):
        st.session_state.clear()
        st.experimental_rerun()
