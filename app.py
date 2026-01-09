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
    text-align: center;
    font-size: 22px;
    height: 48px;
}
.success-box {
    background-color: #d1fae5;
    padding: 12px;
    border-radius: 8px;
    color: #065f46;
    font-weight: 600;
    margin-top: 10px;
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
    st.session_state.hints_used = set()
    st.session_state.solved = False
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

# ---------------- GRID (TABLE FORMAT) ----------------
st.subheader("Sudoku Grid")

for r in range(9):
    cols = st.columns(9)
    for c in range(9):
        disabled = st.session_state.original[r][c] != 0
        value = st.session_state.grid[r][c]
        value = "" if value == 0 else value
        new = cols[c].text_input(
            "",
            value=value,
            key=f"{r}{c}",
            max_chars=1,
            disabled=disabled
        )
        if new.isdigit():
            st.session_state.grid[r][c] = int(new)

# ---------------- BUTTONS ----------------
c1, c2, c3 = st.columns(3)

if c1.button("🔒 Lock Cells"):
    st.session_state.original = copy.deepcopy(st.session_state.grid)

if c2.button("💡 Hint"):
    solution = copy.deepcopy(st.session_state.grid)
    if solve(solution):
        for r in range(9):
            for c in range(9):
                if st.session_state.grid[r][c] == 0 and (r,c) not in st.session_state.hints_used:
                    st.session_state.grid[r][c] = solution[r][c]
                    st.session_state.hints_used.add((r,c))
                    st.session_state.hint_mode = True
                    st.success("✔ Hint applied successfully")
                    st.stop()

if c3.button("🧠 Solve"):
    solved = copy.deepcopy(st.session_state.grid)
    if solve(solved):
        st.session_state.grid = solved
        st.session_state.solved = True
        st.success("✔ Success! The solution was found.")

# ---------------- POST ACTION BUTTONS ----------------
if st.session_state.hint_mode:
    b1, b2, b3 = st.columns(3)
    if b1.button("➕ More Hint"):
        st.session_state.hint_mode = False
    if b2.button("🧠 Solve Full Puzzle"):
        solved = copy.deepcopy(st.session_state.grid)
        solve(solved)
        st.session_state.grid = solved
        st.session_state.solved = True
    if b3.button("🔄 Restart"):
        st.session_state.clear()
        st.experimental_rerun()

if st.session_state.solved:
    if st.button("🔄 Restart"):
        st.session_state.clear()
        st.experimental_rerun()

# ---------------- INFO SECTION ----------------
st.markdown("---")
st.subheader("Why Neurodoku?")
st.write(
    "Neurodoku combines logical problem solving with a clean, distraction-free interface. "
    "It is designed for students, puzzle lovers, and AI enthusiasts."
)

st.subheader("How to use Neurodoku?")
st.write("""
1. Enter numbers into the grid  
2. Lock the given cells  
3. Use Hint if stuck  
4. Solve the full puzzle anytime  
5. Restart to try again
""")

st.subheader("Is Neurodoku free?")
st.write("Yes, Neurodoku is completely free and open-source.")
