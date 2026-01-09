import streamlit as st
import numpy as np
import copy
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="NEURODOKU", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
button {
    height: 60px !important;
    font-size: 22px !important;
    border-radius: 12px !important;
}
.success-box {
    background-color: #d1fae5;
    padding: 12px;
    border-radius: 8px;
    color: #065f46;
    font-weight: 600;
}
.cell-input input {
    text-align: center;
    font-size: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- CENTER LOGO ----------------
def center_logo(path, width=120):
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <div style="display:flex; justify-content:center;">
            <img src="data:image/png;base64,{encoded}" width="{width}">
        </div>
        """,
        unsafe_allow_html=True
    )

center_logo("logo.png")
st.markdown("<h1 style='text-align:center;'>NEURODOKU</h1>", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "grid" not in st.session_state:
    st.session_state.grid = np.zeros((9, 9), dtype=int)
    st.session_state.original = np.zeros((9, 9), dtype=int)
    st.session_state.history = []
    st.session_state.locked = False
    st.session_state.hints_used = set()
    st.session_state.selected = None
    st.session_state.solved = False

# ---------------- SUDOKU LOGIC ----------------
def valid(grid, r, c, n):
    if n in grid[r]: return False
    if n in grid[:, c]: return False
    br, bc = r//3*3, c//3*3
    return n not in grid[br:br+3, bc:bc+3]

def solve(grid):
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                for n in range(1, 10):
                    if valid(grid, r, c, n):
                        grid[r][c] = n
                        if solve(grid):
                            return True
                        grid[r][c] = 0
                return False
    return True

# ---------------- GRID UI ----------------
grid_col, control_col = st.columns([1.3, 1])

with grid_col:
    for r in range(9):
        cols = st.columns(9)
        for c in range(9):
            key = f"{r}-{c}"
            disabled = st.session_state.locked and st.session_state.original[r][c] != 0
            val = st.session_state.grid[r][c]
            val = "" if val == 0 else val

            if cols[c].button(str(val) if val else " ", key=key):
                st.session_state.selected = (r, c)

# ---------------- CONTROLS ----------------
with control_col:

    # TOP BUTTONS
    b1, b2, b3 = st.columns(3)
    if b1.button("Undo"):
        if st.session_state.history:
            st.session_state.grid = st.session_state.history.pop()
    if b2.button("Erase"):
        if st.session_state.selected:
            r, c = st.session_state.selected
            if not st.session_state.locked:
                st.session_state.grid[r][c] = 0
    if b3.button("Clear All"):
        st.session_state.grid[:] = 0
        st.session_state.original[:] = 0
        st.session_state.hints_used.clear()
        st.session_state.locked = False
        st.session_state.solved = False

    st.write("")

    # NUMBER PAD
    nums = [1,2,3,4,5,6,7,8,9]
    for i in range(0, 9, 3):
        cols = st.columns(3)
        for j in range(3):
            n = nums[i+j]
            if cols[j].button(str(n)):
                if st.session_state.selected:
                    r, c = st.session_state.selected
                    if not (st.session_state.locked and st.session_state.original[r][c] != 0):
                        st.session_state.history.append(copy.deepcopy(st.session_state.grid))
                        st.session_state.grid[r][c] = n

    st.write("")

    # LOCK BUTTON
    if st.button("🔒 Lock Cells"):
        st.session_state.original = copy.deepcopy(st.session_state.grid)
        st.session_state.locked = True

    st.write("")

    # SOLVE & HINT
    s1, s2 = st.columns(2)

    if s1.button("Solve"):
        temp = copy.deepcopy(st.session_state.grid)
        if solve(temp):
            st.session_state.grid = temp
            st.session_state.solved = True

    if s2.button("Hint"):
        solution = copy.deepcopy(st.session_state.grid)
        if solve(solution):
            for r in range(9):
                for c in range(9):
                    if st.session_state.grid[r][c] == 0 and (r,c) not in st.session_state.hints_used:
                        st.session_state.history.append(copy.deepcopy(st.session_state.grid))
                        st.session_state.grid[r][c] = solution[r][c]
                        st.session_state.hints_used.add((r,c))
                        st.stop()

# ---------------- SUCCESS MESSAGE ----------------
if st.session_state.solved:
    st.markdown(
        "<div class='success-box'>✔ Success! The solution was found.</div>",
        unsafe_allow_html=True
    )
