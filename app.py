import streamlit as st
import time
import copy
import base64

def load_logo_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()
# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="NEURODOKU", layout="wide")

# -------------------- SESSION STATE --------------------
if "grid" not in st.session_state:
    st.session_state.grid = [[0]*9 for _ in range(9)]

if "selected_cell" not in st.session_state:
    st.session_state.selected_cell = (0, 0)

if "history" not in st.session_state:
    st.session_state.history = []

st.markdown("""
<style>
/* Force proper mobile layout */
@media (max-width: 768px) {
    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .sudoku-grid {
        display: grid;
        grid-template-columns: repeat(9, 1fr);
        gap: 6px;
        justify-content: center;
    }

    .number-pad {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
        margin-top: 10px;
    }

    .controls-stack button {
        width: 100%;
        margin-bottom: 8px;
    }
}
</style>
""", unsafe_allow_html=True)
# -------------------- SUDOKU LOGIC --------------------
def is_valid(grid, row, col, num):
    for i in range(9):
        if grid[row][i] == num or grid[i][col] == num:
            return False
    box_x, box_y = col // 3, row // 3
    for i in range(box_y*3, box_y*3+3):
        for j in range(box_x*3, box_x*3+3):
            if grid[i][j] == num:
                return False
    return True

def solve_sudoku(grid):
    for row in range(9):
        for col in range(9):
            if grid[row][col] == 0:
                for num in range(1, 10):
                    if is_valid(grid, row, col, num):
                        grid[row][col] = num
                        if solve_sudoku(grid):
                            return True
                        grid[row][col] = 0
                return False
    return True

def get_hint(grid):
    temp = copy.deepcopy(grid)
    if solve_sudoku(temp):
        for r in range(9):
            for c in range(9):
                if grid[r][c] == 0:
                    return r, c, temp[r][c]
    return None

def place_number(num):
    r, c = st.session_state.selected_cell
    st.session_state.history.append(copy.deepcopy(st.session_state.grid))
    st.session_state.grid[r][c] = num
    st.rerun()

# -------------------- HEADER (PERFECT CENTER + VISIBLE LOGO) --------------------
logo_base64 = load_logo_base64("logo.png")

st.markdown(
    f"""
    <div style="
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        margin-bottom: 20px;
    ">
        <img src="data:image/png;base64,{logo_base64}" width="100"/>
        <h1 style="text-align:center; margin-top:10px;">NEURODOKU</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------- MAIN LAYOUT --------------------
left, right = st.columns([3, 1])

# -------------------- SUDOKU GRID --------------------
with left:
    st.subheader("Sudoku Grid")
    for r in range(9):
        cols = st.columns(9)
        for c in range(9):
            value = st.session_state.grid[r][c]
            label = str(value) if value != 0 else ""
            if cols[c].button(label, key=f"cell_{r}_{c}"):
                st.session_state.selected_cell = (r, c)
                st.rerun()

# ---------------- CONTROLS + NUMBER PAD + SOLVE ----------------

st.subheader("Controls")

# ---- REQUIRED STATES ----
if "history" not in st.session_state:
    st.session_state.history = []

if "status_msg" not in st.session_state:
    st.session_state.status_msg = None

# ---- CONTROL BUTTONS ----
if st.button("↩️ Undo", use_container_width=True):
    if st.session_state.history:
        st.session_state.grid = st.session_state.history.pop()

if st.button("🧹 Erase", use_container_width=True):
    if st.session_state.selected:
        r, c = st.session_state.selected
        st.session_state.history.append(copy.deepcopy(st.session_state.grid))
        st.session_state.grid[r][c] = 0

if st.button("🔄 Restart", use_container_width=True):
    st.session_state.grid = [[0]*9 for _ in range(9)]
    st.session_state.history = []
    st.session_state.hinted = set()
    st.session_state.solution = None
    st.session_state.selected = None
    st.session_state.status_msg = ("success", "Puzzle restarted successfully")

st.markdown("---")

# ---- SOLVE & HINT ----
if st.button("✅ Solve", use_container_width=True):
    board_copy = copy.deepcopy(st.session_state.grid)
    time.sleep(2)
    if solve(board_copy):
        st.session_state.grid = board_copy
        st.session_state.status_msg = ("success", "Sudoku solved successfully ✅")
    else:
        st.session_state.status_msg = ("error", "No solution exists ❌")

if st.button("💡 Hint", use_container_width=True):
    time.sleep(2)

    if not st.session_state.solution:
        sol = copy.deepcopy(st.session_state.grid)
        if solve(sol):
            st.session_state.solution = sol

    for r in range(9):
        for c in range(9):
            if (
                st.session_state.grid[r][c] == 0
                and (r, c) not in st.session_state.hinted
            ):
                st.session_state.history.append(copy.deepcopy(st.session_state.grid))
                st.session_state.grid[r][c] = st.session_state.solution[r][c]
                st.session_state.hinted.add((r, c))
                st.session_state.status_msg = ("success", "Hint applied successfully 💡")
                break
        else:
            continue
        break

st.markdown("---")
st.subheader("Number Pad")

# ---- NUMBER PAD ----
pad = st.columns(3)
num = 1

for i in range(3):
    for j in range(3):
        if pad[j].button(str(num), key=f"num-{num}", use_container_width=True):
            if st.session_state.selected:
                r, c = st.session_state.selected
                st.session_state.history.append(copy.deepcopy(st.session_state.grid))

                if is_valid(st.session_state.grid, r, c, num):
                    st.session_state.grid[r][c] = num
                    st.session_state.status_msg = ("success", f"Placed {num}")
                else:
                    st.session_state.status_msg = ("error", "❌ Invalid move")
        num += 1

# ---- STATUS MESSAGE ----
if st.session_state.status_msg:
    t, msg = st.session_state.status_msg
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
