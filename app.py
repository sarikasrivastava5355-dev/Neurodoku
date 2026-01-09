import streamlit as st
import numpy as np
import copy
import base64

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="NEURODOKU", layout="wide")

# ---------------- CSS ----------------
st.markdown("""
<style>
.grid {
    display: grid;
    grid-template-columns: repeat(9, 50px);
    gap: 2px;
}
.cell input {
    text-align: center;
    font-size: 20px;
    height: 48px;
    width: 48px;
}
.box-border {
    border-right: 3px solid #444;
}
.box-bottom {
    border-bottom: 3px solid #444;
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
        f"<div style='display:flex;justify-content:center;'><img src='data:image/png;base64,{encoded}' width='110'></div>",
        unsafe_allow_html=True
    )

center_logo("logo.png")
st.markdown("<h1 style='text-align:center;'>NEURODOKU</h1>", unsafe_allow_html=True)

# ---------------- SESSION STATE ----------------
if "grid" not in st.session_state:
    st.session_state.grid = np.zeros((9,9), dtype=int)
    st.session_state.original = np.zeros((9,9), dtype=int)
    st.session_state.selected = None
    st.session_state.hints_used = set()
    st.session_state.solved = False
    st.session_state.history = []

# ---------------- SUDOKU LOGIC ----------------
def valid(grid, r, c, n):
    if n in grid[r]: return False
    if n in grid[:,c]: return False
    br, bc = r//3*3, c//3*3
    return n not in grid[br:br+3, bc:bc+3]

def solve(grid):
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                for n in range(1,10):
                    if valid(grid, r, c, n):
                        grid[r][c] = n
                        if solve(grid): return True
                        grid[r][c] = 0
                return False
    return True

# ---------------- MAIN LAYOUT ----------------
grid_col, pad_col = st.columns([1.4, 1])

# ---------------- SUDOKU GRID ----------------
with grid_col:
    st.subheader("Sudoku Grid")
    for r in range(9):
        cols = st.columns(9)
        for c in range(9):
            disabled = st.session_state.original[r][c] != 0
            value = st.session_state.grid[r][c]
            value = "" if value == 0 else value

            if cols[c].button(value if value else " ", key=f"cell-{r}-{c}"):
                st.session_state.selected = (r,c)

# ---------------- NUMBER PAD + CONTROLS ----------------
with pad_col:
    st.subheader("Controls")

    # Undo / Erase / Clear
    a,b,c = st.columns(3)
    if a.button("Undo") and st.session_state.history:
        st.session_state.grid = st.session_state.history.pop()
    if b.button("Erase") and st.session_state.selected:
        r,cx = st.session_state.selected
        st.session_state.grid[r][cx] = 0
    if c.button("Clear All"):
        st.session_state.grid[:] = 0
        st.session_state.original[:] = 0
        st.session_state.hints_used.clear()
        st.session_state.history.clear()
        st.session_state.solved = False

    st.write("")

    # NUMBER PAD
    st.subheader("Number Pad")
    numbers = [1,2,3,4,5,6,7,8,9]
    for i in range(0,9,3):
        cols = st.columns(3)
        for j in range(3):
            n = numbers[i+j]
            if cols[j].button(str(n)):
                if st.session_state.selected:
                    r,cx = st.session_state.selected
                    st.session_state.history.append(copy.deepcopy(st.session_state.grid))
                    st.session_state.grid[r][cx] = n

    st.write("")

    if st.button("🔒 Lock Cells"):
        st.session_state.original = copy.deepcopy(st.session_state.grid)

    # Hint / Solve
    d,e = st.columns(2)

    if d.button("Hint"):
        sol = copy.deepcopy(st.session_state.grid)
        if solve(sol):
            for r in range(9):
                for c in range(9):
                    if st.session_state.grid[r][c] == 0 and (r,c) not in st.session_state.hints_used:
                        st.session_state.grid[r][c] = sol[r][c]
                        st.session_state.hints_used.add((r,c))
                        st.success("✔ Hint applied successfully")
                        st.stop()

    if e.button("Solve"):
        sol = copy.deepcopy(st.session_state.grid)
        if solve(sol):
            st.session_state.grid = sol
            st.session_state.solved = True

# ---------------- POST SOLVE ----------------
if st.session_state.solved:
    st.markdown("<div class='success-box'>✔ Success! The solution was found.</div>", unsafe_allow_html=True)
    if st.button("🔄 Restart"):
        st.session_state.clear()
        st.experimental_rerun()

# ---------------- INFO SECTION ----------------
st.markdown("---")
st.subheader("Why Neurodoku?")
st.write("Neurodoku blends logical reasoning with a clean, modern interface designed for students and puzzle lovers.")

st.subheader("How to use Neurodoku?")
st.write("""
• Select a cell  
• Use number pad to enter digits  
• Lock given cells  
• Use hints if stuck  
• Solve or restart anytime
""")
