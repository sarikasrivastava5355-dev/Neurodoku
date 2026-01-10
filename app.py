import streamlit as st
import copy
import time

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Neurodoku", layout="centered")

# ---------------- SESSION STATE ----------------
if "grid" not in st.session_state:
    st.session_state.grid = [[0]*9 for _ in range(9)]

if "selected" not in st.session_state:
    st.session_state.selected = None

if "solution" not in st.session_state:
    st.session_state.solution = None

if "hinted" not in st.session_state:
    st.session_state.hinted = set()

if "history" not in st.session_state:
    st.session_state.history = []

if "status" not in st.session_state:
    st.session_state.status = None

# ---------------- CSS ----------------
st.markdown("""
<style>
.center {
    text-align: center;
}

.sudoku-grid {
    display: grid;
    grid-template-columns: repeat(9, 42px);
    gap: 6px;
    justify-content: center;
}

.sudoku-grid button {
    width: 42px !important;
    height: 42px !important;
    font-size: 18px !important;
}

.number-pad {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
}

@media (max-width: 768px) {
    .sudoku-grid {
        grid-template-columns: repeat(9, 1fr);
    }
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOGO + TITLE ----------------
st.markdown("""
<div class="center">
    <img src="logo.png" width="90">
    <h1>NEURODOKU</h1>
</div>
""", unsafe_allow_html=True)

# ---------------- VALIDATION ----------------
def is_valid(grid, r, c, n):
    for i in range(9):
        if grid[r][i] == n or grid[i][c] == n:
            return False
    br, bc = r//3*3, c//3*3
    for i in range(3):
        for j in range(3):
            if grid[br+i][bc+j] == n:
                return False
    return True

# ---------------- SOLVER ----------------
def solve(grid):
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                for n in range(1, 10):
                    if is_valid(grid, r, c, n):
                        grid[r][c] = n
                        if solve(grid):
                            return True
                        grid[r][c] = 0
                return False
    return True

# ---------------- MAIN LAYOUT ----------------
grid_col, control_col = st.columns([3, 2])

# ---------------- GRID ----------------
with grid_col:
    st.subheader("Sudoku Grid")
    st.markdown('<div class="sudoku-grid">', unsafe_allow_html=True)

    for r in range(9):
        for c in range(9):
            value = st.session_state.grid[r][c]
            label = str(value) if value != 0 else ""
            if st.button(label, key=f"cell-{r}-{c}"):
                st.session_state.selected = (r, c)

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- CONTROLS ----------------
st.subheader("Controls")

edit_col, solve_col = st.columns(2)

# ---------- EDIT CONTROLS ----------
with edit_col:
    st.markdown("### ✏️ Edit")

    if st.button("↩️ Undo", use_container_width=True):
        if st.session_state.history:
            st.session_state.grid = st.session_state.history.pop()

    if st.button("🧹 Erase", use_container_width=True):
        if st.session_state.selected:
            r, c = st.session_state.selected
            st.session_state.history.append(copy.deepcopy(st.session_state.grid))
            st.session_state.grid[r][c] = 0

    if st.button("🔄 Clear All", use_container_width=True):
        st.session_state.grid = [[0]*9 for _ in range(9)]
        st.session_state.history = []
        st.session_state.hinted = set()
        st.session_state.solution = None
        st.session_state.selected = None
        st.session_state.status = ("success", "Grid cleared")

# ---------- SOLVE CONTROLS ----------
with solve_col:
    st.markdown("### 🤖 AI Assist")

    if st.button("💡 Hint", use_container_width=True):
        if not st.session_state.solution:
            sol = copy.deepcopy(st.session_state.grid)
            solve(sol)
            st.session_state.solution = sol

        for r in range(9):
            for c in range(9):
                if st.session_state.grid[r][c] == 0 and (r, c) not in st.session_state.hinted:
                    st.session_state.history.append(copy.deepcopy(st.session_state.grid))
                    st.session_state.grid[r][c] = st.session_state.solution[r][c]
                    st.session_state.hinted.add((r, c))
                    st.session_state.status = ("success", "Hint applied")
                    break
            else:
                continue
            break

    if st.button("✅ Solve", use_container_width=True):
        temp = copy.deepcopy(st.session_state.grid)
        if solve(temp):
            st.session_state.grid = temp
            st.session_state.status = ("success", "Puzzle solved")
        else:
            st.session_state.status = ("error", "No solution exists")

    if st.button("🔁 Restart", use_container_width=True):
        st.session_state.grid = [[0]*9 for _ in range(9)]
        st.session_state.history = []
        st.session_state.hinted = set()
        st.session_state.solution = None
        st.session_state.selected = None
        st.session_state.status = ("success", "Game restarted")

# ---------------- NUMBER PAD ----------------
st.markdown("---")
st.subheader("Number Pad")

num_cols = st.columns(3)
num = 1

for row in range(3):
    for col in range(3):
        with num_cols[col]:
            if st.button(str(num), key=f"num-{num}", use_container_width=True):
                if st.session_state.selected:
                    r, c = st.session_state.selected
                    if is_valid(st.session_state.grid, r, c, num):
                        st.session_state.history.append(copy.deepcopy(st.session_state.grid))
                        st.session_state.grid[r][c] = num
                        st.session_state.status = ("success", f"Placed {num}")
                    else:
                        st.session_state.status = ("error", "Invalid move")
        num += 1

# ---------------- STATUS MESSAGE ----------------
if st.session_state.status:
    t, msg = st.session_state.status
    if t == "success":
        st.success(msg)
    else:
        st.error(msg)

# -------------------- INFORMATION SECTION --------------------
st.markdown("---")

st.markdown("## ❓ How to use Neurodoku?")
st.write("""
1. Click on any empty cell in the Sudoku grid.
2. Use the Number Pad to enter digits.
3. Use **Hint** to get help for one cell.
4. Use **Solve** to get the complete solution.
5. **Undo**, **Erase**, and **Restart** help control your progress.
""")

st.markdown("## ❓ What is Neurodoku?")
st.write("""
Neurodoku is an intelligent Sudoku-solving web application that assists users
by validating moves, providing hints, and generating complete solutions.
""")

st.markdown("## ❓ How is Neurodoku an AI Model?")
st.write("""
Neurodoku mimics human problem-solving by applying logical reasoning,
constraint satisfaction, and decision-making algorithms to solve Sudoku puzzles.
It demonstrates Artificial Intelligence through automated reasoning and rule-based learning.
""")
