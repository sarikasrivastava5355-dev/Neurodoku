import streamlit as st
import numpy as np
import copy
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="NEURODOKU", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.cell input {
    width: 45px !important;
    height: 45px !important;
    text-align: center;
    font-size: 20px;
    padding: 0;
}
.success-box {
    background-color: #d1fae5;
    padding: 10px;
    border-radius: 6px;
    color: #065f46;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# ---------------- CENTER LOGO ----------------
def center_logo(path):
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"<div style='display:flex;justify-content:center;'><img src='data:image/png;base64,{encoded}' width='120'></div>",
        unsafe_allow_html=True
    )

center_logo("logo.png")
st.markdown("<h1 style='text-align:center;'>NEURODOKU</h1>", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "grid" not in st.session_state:
    st.session_state.grid = np.zeros((9, 9), dtype=int)
    st.session_state.original = np.zeros((9, 9), dtype=int)
    st.session_state.selected = None
    st.session_state.hints_used = set()
    st.session_state.solved = False
    st.session_state.history = []
    st.session_state.hint_mode = False

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
                        if solve(grid): return True
                        grid[r][c] = 0
                return False
    return True

# ---------------- MAIN LAYOUT ----------------
grid_col, pad_col = st.columns([1.3, 1])

# ---------------- GRID (TRUE TABLE) ----------------
with grid_col:
    st.subheader("Sudoku Grid")

    for r in range(9):
        cols = st.columns(9)
        for c in range(9):
            disabled = st.session_state.original[r][c] != 0
            val = st.session_state.grid[r][c]
            val = "" if val == 0 else str(val)

            new = cols[c].text_input(
                "",
                value=val,
                key=f"cell-{r}-{c}",
                max_chars=1,
                disabled=disabled
            )

            if new.isdigit():
                st.session_state.grid[r][c] = int(new)
                st.session_state.selected = (r, c)

# ---------------- NUMBER PAD & CONTROLS ----------------
with pad_col:
    st.subheader("Controls")

    # Undo / Erase / Clear
    a, b, c = st.columns(3)
    if a.button("Undo") and st.session_state.history:
        st.session_state.grid = st.session_state.history.pop()
    if b.button("Erase") and st.session_state.selected:
        r, cx = st.session_state.selected
        st.session_state.grid[r][cx] = 0
    if c.button("Clear All"):
        st.session_state.clear()
        st.experimental_rerun()

    st.write("")

    # NUMBER PAD
    st.subheader("Number Pad")
    numbers = [1,2,3,4,5,6,7,8,9]
    for i in range(0, 9, 3):
        cols = st.columns(3)
        for j in range(3):
            n = numbers[i+j]
            if cols[j].button(str(n)):
                if st.session_state.selected:
                    r, cx = st.session_state.selected
                    st.session_state.history.append(copy.deepcopy(st.session_state.grid))
                    st.session_state.grid[r][cx] = n

    st.write("")

    if st.button("🔒 Lock Cells"):
        st.session_state.original = copy.deepcopy(st.session_state.grid)

    d, e = st.columns(2)

    if d.button("Hint"):
        sol = copy.deepcopy(st.session_state.grid)
        if solve(sol):
            for r in range(9):
                for c in range(9):
                    if st.session_state.grid[r][c] == 0 and (r,c) not in st.session_state.hints_used:
                        st.session_state.grid[r][c] = sol[r][c]
                        st.session_state.hints_used.add((r,c))
                        st.session_state.hint_mode = True
                        st.success("✔ Hint applied successfully")
                        st.stop()

    if e.button("Solve"):
        sol = copy.deepcopy(st.session_state.grid)
        if solve(sol):
            st.session_state.grid = sol
            st.session_state.solved = True

# ---------------- POST ACTIONS ----------------
if st.session_state.hint_mode:
    x, y, z = st.columns(3)
    if x.button("➕ More Hint"):
        st.session_state.hint_mode = False
    if y.button("🧠 Solve Full Puzzle"):
        sol = copy.deepcopy(st.session_state.grid)
        solve(sol)
        st.session_state.grid = sol
        st.session_state.solved = True
    if z.button("🔄 Restart"):
        st.session_state.clear()
        st.experimental_rerun()

if st.session_state.solved:
    st.markdown("<div class='success-box'>✔ Success! The solution was found.</div>", unsafe_allow_html=True)
    if st.button("🔄 Restart"):
        st.session_state.clear()
        st.experimental_rerun()

# ---------------- INFO SECTION ----------------
st.markdown("---")
st.subheader("Why Neurodoku?")
st.write("Neurodoku is a clean, logic-based Sudoku solver designed for students and puzzle enthusiasts.")

st.subheader("How to use Neurodoku?")
st.write("""
1. Enter numbers in the grid  
2. Lock given cells  
3. Use hints if needed  
4. Solve fully or restart anytime
""")
