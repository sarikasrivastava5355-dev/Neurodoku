import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract

# ---------------------------------------------------------
#                    PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(page_title="NEURODOKU", layout="wide")

# Light notebook-style background
st.markdown("""
    <style>
        body {
            background-color: #f4f5f7;
        }
        .main {
            background-color: #f4f5f7;
        }
        .stButton>button {
            background-color: #1f77b4;
            color: white;
            font-size: 16px;
            border-radius: 8px;
            padding: 8px 20px;
        }
        .title {
            font-size: 40px;
            text-align: center;
            font-weight: 700;
            color: #003366;
        }
        .section-title {
            font-size: 28px;
            font-weight: 600;
            color: #003366;
        }
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
#              DISPLAY LOGO + TITLE
# ---------------------------------------------------------

colL, colM, colR = st.columns([1, 2, 1])
with colM:
    st.image("logo.png", width=180)
    st.markdown("<div class='title'>NEURODOKU</div>", unsafe_allow_html=True)

st.write("")

# ---------------------------------------------------------
#                USER SELECTS PUZZLE SIZE
# ---------------------------------------------------------

st.subheader("Which puzzle are you solving?")
puzzle = st.radio("", ["6 × 6 Sudoku", "9 × 9 Sudoku"], horizontal=True)

grid_size = 6 if puzzle == "6 × 6 Sudoku" else 9

# ---------------------------------------------------------
#                FILE UPLOAD SECTION
# ---------------------------------------------------------

uploaded = st.file_uploader("Upload Sudoku image", type=["png", "jpg", "jpeg"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    st.image(img_np, caption="Uploaded Image", width=350)

    # ---------------------------------------------------------
    #      GRID EXTRACTION (robust for your images)
    # ---------------------------------------------------------

    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 3
    )

    # Find largest contour → sudoku grid
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    grid_contour = None
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            grid_contour = approx
            break

    if grid_contour is None:
        st.error("❌ Could not detect puzzle grid.")
        st.stop()

    # Order grid corners
    pts = grid_contour.reshape(4, 2)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    ordered = np.array([
        pts[np.argmin(s)],  # top-left
        pts[np.argmin(diff)],  # top-right
        pts[np.argmax(s)],  # bottom-right
        pts[np.argmax(diff)]   # bottom-left
    ], dtype="float32")

    # Warp to perfect square
    side = 450
    dst = np.array([[0, 0], [side-1, 0], [side-1, side-1], [0, side-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(ordered, dst)
    warp = cv2.warpPerspective(gray, M, (side, side))

    # ---------------------------------------------------------
    #               OCR EXTRACT DIGITS (TESSERACT)
    # ---------------------------------------------------------

    cell = side // grid_size
    grid = np.zeros((grid_size, grid_size), dtype=int)

    for r in range(grid_size):
        for c in range(grid_size):
            crop = warp[r*cell:(r+1)*cell, c*cell:(c+1)*cell]

            crop = cv2.resize(crop, (60, 60))
            crop = cv2.adaptiveThreshold(
                crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 2
            )

            text = pytesseract.image_to_string(
                crop,
                config='--psm 10 --oem 3 -c tessedit_char_whitelist=123456789'
            ).strip()

            grid[r, c] = int(text) if text.isdigit() else 0

    # ---------------------------------------------------------
    #              SHOW EXTRACTED GRID
    # ---------------------------------------------------------

    st.subheader("Extracted Grid (blue = OCR result)")

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(np.ones((grid_size, grid_size)), cmap="gray", vmin=0, vmax=1)

    for r in range(grid_size):
        for c in range(grid_size):
            if grid[r, c] != 0:
                ax.text(c, r, str(grid[r, c]), color="blue", ha="center", va="center", fontsize=16)

    ax.set_xticks(np.arange(-0.5, grid_size, 1))
    ax.set_yticks(np.arange(-0.5, grid_size, 1))
    ax.grid(color="black")
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    st.pyplot(fig)

    # ---------------------------------------------------------
    #               SUDOKU SOLVER (FAST)
    # ---------------------------------------------------------

    def valid(board, r, c, x):
        for i in range(grid_size):
            if board[r][i] == x or board[i][c] == x:
                return False
        br = (r // int(np.sqrt(grid_size))) * int(np.sqrt(grid_size))
        bc = (c // int(np.sqrt(grid_size))) * int(np.sqrt(grid_size))
        for i in range(int(np.sqrt(grid_size))):
            for j in range(int(np.sqrt(grid_size))):
                if board[br+i][bc+j] == x:
                    return False
        return True

    def solve(board):
        for i in range(grid_size):
            for j in range(grid_size):
                if board[i][j] == 0:
                    for x in range(1, grid_size+1):
                        if valid(board, i, j, x):
                            board[i][j] = x
                            if solve(board):
                                return True
                            board[i][j] = 0
                    return False
        return True

    # ---------------------------------------------------------
    #                   BUTTONS (Solve / Continue)
    # ---------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Solve Puzzle"):
            grid_copy = grid.copy()
            if solve(grid_copy):
                st.success("Solved Puzzle")

                fig2, ax2 = plt.subplots(figsize=(5, 5))
                for r in range(grid_size):
                    for c in range(grid_size):
                        color = "black" if grid[r, c] == 0 else "blue"
                        ax2.text(c, r, str(grid_copy[r, c]), ha="center", va="center", fontsize=16, color=color)

                ax2.imshow(np.ones((grid_size, grid_size)), cmap="gray", vmin=0, vmax=1)
                ax2.set_xticks(np.arange(-0.5, grid_size, 1))
                ax2.set_yticks(np.arange(-0.5, grid_size, 1))
                ax2.grid(color="black")
                ax2.set_xticklabels([])
                ax2.set_yticklabels([])
                st.pyplot(fig2)
            else:
                st.error("This puzzle cannot be solved.")

    with col2:
        if st.button("Upload New Puzzle"):
            st.session_state.clear()
            st.rerun()
