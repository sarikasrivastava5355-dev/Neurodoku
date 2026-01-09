import streamlit as st
import numpy as np
import random
from PIL import Image

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="NEURODOKU",
    page_icon="🧠",
    layout="centered"
)

# --------------------------------------------------
# DARK MODE TOGGLE
# --------------------------------------------------
dark_mode = st.toggle("🌙 Dark Mode")

bg = "#1e1e1e" if dark_mode else "#f4f6fa"
text = "#ffffff" if dark_mode else "#2b2d42"
cell_bg = "#2a2a2a" if dark_mode else "#edf2f4"

st.markdown(f"""
<style>
body {{
    background-color: {bg};
    color: {text};
}}
input {{
    text-align: center;
    font-size: 20px !important;
    font-weight: 600;
    background-color: {cell_bg} !important;
    border-radius: 6px !important;
}}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOGO (CENTERED ABOVE TITLE)
# --------------------------------------------------
logo = Image.open("logo.png")

c1, c2, c3 = st.columns([1, 2, 1])
with c2:
    st.image(logo, width=150)

# --------------------------------------------------
# TITLE
# --------------------------------------------------
st.markdown(
    f"<h1 style='text-align:center; color:{text}; margin-top:10px;'>NEURODOKU</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Smart Sudoku Solver</p>",
    unsafe_allow_html=True
)
st.write("---")

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------
if "grid" not in st.session_state:
    st.session_state.grid = np.zeros((9, 9), dtype=int)

if "solution" not in st.session_state:
    st.session_state.solution = None

if "hinted_cells" not in st.session_state:
    st.session_state.hinted_cells = set()

if "locked_cells" not in st.session_state:
    st.session_state.locked_cells = set()

if "erase_mode" not in st.session_state:
    st.session_state.erase_mode = False

# --------------------------------------------------
# CONTROL BUTTONS
# --------------------------------------------------
ctrl1, ctrl2, ctrl3 = st.columns(3)

with ctrl1:
    if st.button("🧹 Erase"):
        st.session_state.erase_mode = True

with ctrl2:
    if st.button("🔄 Reset"):
        st.session_state.grid = np.zeros((9, 9), dtype=int)
        st.session_state.solution = None
        st.session_state.hinted_cells.clear()
        st.session_state.locked_cells.clear()
        st.session_state.erase_mode = False

with ctrl3:
    if st.button("🔒 Lock Input"):
        st.session_state.locked_cells = {
            (i, j) for i in range(9) for j in range(9)
            if st.session_state.grid[i][j] != 0
        }

st.write("---")

# --------------------------------------------------
# INPUT GRID
# --------------------------------------------------
for i in range(9):
    cols = st.columns(9)
    for j in range(9):
        with cols[j]:
            disabled = (i, j) in st.session_state.locked_cells
            value = "" if st.session_state.grid[i][j] == 0 else st.session_state.grid[i][j]

            user_input = st.text_input(
                "",
                value=value,
                key=f"cell_{i}_{j}",
                max_chars=1,
                disabled=disabled
            )

            if st.session_state.erase_mode and not disabled:
                st.session_state.grid[i][j] = 0
            elif user_input.isdigit() and user_input != "0" and not disabled:
                st.session_state.grid[i][j] = int(user_input)

# --------------------------------------------------
# SUDOKU SOLVER LOGIC
# --------------------------------------------------
def is_valid(board, row, col, num):
    if num in board[row]:
        return False
    if num in board[:, col]:
        return False
    bx, by = row // 3, col // 3
    if num in board[bx*3:bx*3+3, by*3:by*3+3]:
        return False
    return True

def solve(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                for num in range(1, 10):
                    if is_valid(board, i, j, num):
                        board[i][j] = num
                        if solve(board):
                            return True
                        board[i][j] = 0
                return False
    return True

# --------------------------------------------------
# ACTION BUTTONS
# --------------------------------------------------
act1, act2 = st.columns(2)

with act1:
    if st.button("🧠 Solve"):
        board = st.session_state.grid.copy()
        if solve(board):
            st.session_state.solution = board
            st.success("Sudoku solved successfully!")
        else:
            st.error("Invalid or unsolvable Sudoku")

with act2:
    if st.button("💡 Hint"):
        board = st.session_state.grid.copy()
        solved = board.copy()

        if solve(solved):
            available = [
                (i, j) for i in range(9) for j in range(9)
                if board[i][j] == 0 and (i, j) not in st.session_state.hinted_cells
            ]
            if available:
                i, j = random.choice(available)
                st.session_state.grid[i][j] = solved[i][j]
                st.session_state.hinted_cells.add((i, j))
            else:
                st.warning("No more hints available")

# --------------------------------------------------
# DISPLAY SOLUTION
# --------------------------------------------------
if st.session_state.solution is not None:
    st.write("---")
    st.subheader("✅ Solution")

    for i in range(9):
        cols = st.columns(9)
        for j in range(9):
            cols[j].markdown(
                f"<div style='text-align:center; font-size:22px; font-weight:700; color:{text};'>"
                f"{st.session_state.solution[i][j]}</div>",
                unsafe_allow_html=True
)
