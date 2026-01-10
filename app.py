import streamlit as st
import time
import copy

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="NEURODOKU", layout="wide")

# -------------------- SESSION STATE --------------------
if "grid" not in st.session_state:
    st.session_state.grid = [[0]*9 for _ in range(9)]

if "selected_cell" not in st.session_state:
    st.session_state.selected_cell = (0, 0)

if "history" not in st.session_state:
    st.session_state.history = []

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

# -------------------- HEADER --------------------
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
st.image("logo.png", width=90)
st.markdown("<h1>NEURODOKU</h1>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

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

# -------------------- CONTROLS & NUMBER PAD --------------------
with right:
    st.subheader("Controls")

    if st.button("↩ Undo"):
        if st.session_state.history:
            st.session_state.grid = st.session_state.history.pop()
            st.rerun()

    if st.button("🧹 Erase"):
        place_number(0)

    if st.button("🗑 Clear All"):
        st.session_state.grid = [[0]*9 for _ in range(9)]
        st.session_state.history.clear()
        st.rerun()

    st.subheader("Number Pad")
    for i in range(1, 10, 3):
        cols = st.columns(3)
        for j in range(3):
            num = i + j
            if cols[j].button(str(num), key=f"num_{num}"):
                place_number(num)

    st.markdown("---")

    if st.button("💡 Hint", use_container_width=True):
        with st.spinner("Finding hint..."):
            time.sleep(1)
            hint = get_hint(st.session_state.grid)
            if hint:
                r, c, v = hint
                st.session_state.grid[r][c] = v
        st.rerun()

    if st.button("✅ Solve", use_container_width=True):
        with st.spinner("Solving Sudoku..."):
            time.sleep(1.5)
            solved = copy.deepcopy(st.session_state.grid)
            solve_sudoku(solved)
            st.session_state.grid = solved
        st.rerun()

    if st.button("🔁 Restart", use_container_width=True):
        st.session_state.grid = [[0]*9 for _ in range(9)]
        st.session_state.history.clear()
        st.rerun()

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
