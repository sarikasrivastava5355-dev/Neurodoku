import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
import tempfile

st.set_page_config(page_title="NEURODOKU", layout="wide")

# ----------------------------- STYLE -----------------------------
st.markdown("""
<style>
    body {background-color: #f4f5f7;}
    .title {text-align: center; font-size: 36px; font-weight: 700; color: #003366;}
    .stButton>button {background-color: #1f77b4; color: white; font-size: 16px; border-radius: 8px; padding: 8px 20px;}
</style>
""", unsafe_allow_html=True)

# ----------------------------- LOGO AND TITLE -----------------------------
col1, col2, col3 = st.columns([1,2,1])
with col2:
    st.image("logo.png", width=180)  # circle part only
    st.markdown('<div class="title">NEURODOKU</div>', unsafe_allow_html=True)

st.write("")

# ----------------------------- STATE MANAGEMENT -----------------------------
if 'stage' not in st.session_state:
    st.session_state.stage = "select_puzzle"  # stages: select_puzzle, upload, extracted, solved
if 'board' not in st.session_state:
    st.session_state.board = None
if 'grid_img' not in st.session_state:
    st.session_state.grid_img = None

# ----------------------------- PUZZLE SIZE SELECTION -----------------------------
if st.session_state.stage == "select_puzzle":
    st.subheader("Which puzzle are you solving?")
    if st.button("9 × 9 Sudoku"):
        st.session_state.stage = "upload"

# ----------------------------- FILE UPLOADER -----------------------------
if st.session_state.stage == "upload":
    uploaded = st.file_uploader("Upload your 9×9 Sudoku image", type=["png","jpg","jpeg"])
    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        img_np = np.array(img)
        st.session_state.grid_img = img_np
        st.session_state.stage = "extracted"
        st.experimental_rerun()

# ----------------------------- OCR / GRID EXTRACTION -----------------------------
def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,11,3)
    return thresh

def extract_grid(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh = preprocess(img)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    grid_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            grid_contour = approx
            break
    if grid_contour is None: return None
    pts = grid_contour.reshape(4,2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    ordered = np.array([pts[np.argmin(s)], pts[np.argmin(diff)], pts[np.argmax(s)], pts[np.argmax(diff)]], dtype="float32")
    side = 450
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered,dst)
    warp = cv2.warpPerspective(gray,M,(side,side))
    return warp

custom_config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=123456789'

def read_digit(cell):
    h, w = cell.shape
    cell = cell[5:h-5, 5:w-5]
    _, cell = cv2.threshold(cell,0,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    txt = pytesseract.image_to_string(cell, config=custom_config).strip()
    return int(txt) if txt.isdigit() else 0

def extract_digits(grid_img):
    size = 9
    side = grid_img.shape[0]
    cell = side // size
    board = np.zeros((size,size), dtype=int)
    for r in range(size):
        for c in range(size):
            crop = grid_img[r*cell:(r+1)*cell, c*cell:(c+1)*cell]
            board[r,c] = read_digit(crop)
    return board

# ----------------------------- SOLVER -----------------------------
def valid(b, r, c, x):
    if x in b[r]: return False
    if x in b[:,c]: return False
    br, bc = (r//3)*3, (c//3)*3
    if x in b[br:br+3, bc:bc+3]: return False
    return True

def solve(b):
    for i in range(9):
        for j in range(9):
            if b[i][j]==0:
                for x in range(1,10):
                    if valid(b,i,j,x):
                        b[i][j]=x
                        if solve(b): return True
                        b[i][j]=0
                return False
    return True

# ----------------------------- EXTRACTED GRID DISPLAY -----------------------------
if st.session_state.stage == "extracted":
    img_np = st.session_state.grid_img
    grid_img = extract_grid(img_np)
    if grid_img is None:
        st.error("❌ Could not detect grid.")
    else:
        board = extract_digits(grid_img)
        st.session_state.board = board

        # Display extracted grid
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5,5))
        ax.imshow(np.ones((9,9)), cmap="gray", vmin=0,vmax=1)
        for r in range(9):
            for c in range(9):
                if board[r,c] != 0:
                    ax.text(c,r,str(board[r,c]), ha="center",va="center",fontsize=16,color="blue")
        ax.set_xticks(np.arange(-0.5,9,1))
        ax.set_yticks(np.arange(-0.5,9,1))
        ax.grid(color="black")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        st.pyplot(fig)

        # Buttons for Hint / Full Solution
        st.write("### Choose an option:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Hint"):
                # Fill first empty cell as hint
                b_copy = board.copy()
                for r in range(9):
                    for c in range(9):
                        if b_copy[r,c]==0:
                            for x in range(1,10):
                                if valid(b_copy,r,c,x):
                                    b_copy[r,c]=x
                                    break
                            break
                    else:
                        continue
                    break
                # Display grid with hint in green
                fig2, ax2 = plt.subplots(figsize=(5,5))
                ax2.imshow(np.ones((9,9)), cmap="gray", vmin=0,vmax=1)
                for r in range(9):
                    for c in range(9):
                        if board[r,c] != 0:
                            ax2.text(c,r,str(board[r,c]),ha="center",va="center",fontsize=16,color="blue")
                        elif b_copy[r,c]!=0:
                            ax2.text(c,r,str(b_copy[r,c]),ha="center",va="center",fontsize=16,color="green")
                ax2.set_xticks(np.arange(-0.5,9,1))
                ax2.set_yticks(np.arange(-0.5,9,1))
                ax2.grid(color="black")
                ax2.set_xticklabels([]); ax2.set_yticklabels([])
                st.pyplot(fig2)

        with col2:
            if st.button("Full Solution"):
                b_copy = board.copy()
                solve(b_copy)
                fig3, ax3 = plt.subplots(figsize=(5,5))
                ax3.imshow(np.ones((9,9)), cmap="gray", vmin=0,vmax=1)
                for r in range(9):
                    for c in range(9):
                        if board[r,c] !=0:
                            ax3.text(c,r,str(board[r,c]),ha="center",va="center",fontsize=16,color="blue")
                        else:
                            ax3.text(c,r,str(b_copy[r,c]),ha="center",va="center",fontsize=16,color="black")
                ax3.set_xticks(np.arange(-0.5,9,1))
                ax3.set_yticks(np.arange(-0.5,9,1))
                ax3.grid(color="black")
                ax3.set_xticklabels([]); ax3.set_yticklabels([])
                st.pyplot(fig3)

        # Continue / New Puzzle
        st.write("### Next step:")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Continue with this puzzle"):
                st.session_state.stage="extracted"
                st.experimental_rerun()
        with col2:
            if st.button("Upload new puzzle"):
                st.session_state.stage="upload"
                st.experimental_rerun()
