import streamlit as st
import pandas as pd
import copy

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="NEURODOKU", layout="wide")

# ---------------- LOGO + TITLE ----------------
st.markdown(
    """
    <div style="text-align:center;">
        <img src="https://github.com/sarikasrivastava5355-dev/Neurodoku/blob/305099c891349845486fe0f0f6b85faaf19d5ff6/logo.png" width="90">
        <h1>NEURODOKU</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------- SESSION STATE ----------------
if "grid" not in st.session_state:
    st.session_state.grid = pd.DataFrame([[0]*9 for _ in range(9)])

if "solution" not in st.session_state:
    st.session_state.solution = None

if "history" not in st.session_state:
    st.session_state.history = []

if "selected_cell" not in st.session_state:
    st.session_state.selected_cell = (0, 0)

if "hinted_cells" not in st.session_state:
    st.session_state.hinted_cells = set()

# ---------------- SUDOKU LOGIC ----------------
def is_valid(grid, r, c, num):
    if num in grid.iloc[r].values:
        return False
    if num in grid.iloc[:, c].values:
        return False
    br, bc = (r//3)*3, (c//3)*3
    if num in grid.iloc[br:br+3, bc:bc+3].values:
        return False
    return True

def solve_sudoku(grid):
    for r in range(9):
        for c in range(9):
            if grid.iat[r, c] == 0:
                for num in range(1, 10):
                    if is_valid(grid, r, c, num):
                        grid.iat[r, c] = num
                        if solve_sudoku(grid):
                            return True
                        grid.iat[r, c] = 0
                return False
    return True

# ---------------- LAYOUT ----------------
left, right = st.columns([3, 1])

# ---------------- GRID ----------------
with left:
    st.subheader("Sudoku Grid")

    for r in range(9):
        cols = st.columns(9)
        for c in range(9):
            val = st.session_state.grid.iat[r, c]
            label = str(val) if val != 0 else ""
            if cols[c].button(label, key=f"cell-{r}-{c}", use_container_width=True):
                st.session_state.selected_cell = (r, c)

# ---------------- CONTROLS ----------------
with right:
    st.subheader("Controls")

    c1, c2, c3 = st.columns(3)

    if c1.button("↩ Undo"):
        if st.session_state.history:
            st.session_state.grid = st.session_state.history.pop()

    if c2.button("🧹 Erase"):
        r, c = st.session_state.selected_cell
        st.session_state.history.append(st.session_state.grid.copy())
        st.session_state.grid.iat[r, c] = 0

    if c3.button("🗑 Clear All"):
        st.session_state.grid = pd.DataFrame([[0]*9 for _ in range(9)])
        st.session_state.history.clear()
        st.session_state.hinted_cells.clear()
        st.session_state.solution = None

    st.markdown("### Number Pad")

    for i in range(0, 9, 3):
        row = st.columns(3)
        for j in range(3):
            num = i + j + 1
            if row[j].button(str(num), use_container_width=True):
                r, c = st.session_state.selected_cell
                if is_valid(st.session_state.grid, r, c, num):
                    st.session_state.history.append(st.session_state.grid.copy())
                    st.session_state.grid.iat[r, c] = num
                else:
                    st.warning("Invalid move")

    st.markdown("---")

    if st.button("💡 Hint", use_container_width=True):
        temp = st.session_state.grid.copy()
        solved = temp.copy()
        if solve_sudoku(solved):
            for r in range(9):
                for c in range(9):
                    if temp.iat[r, c] == 0 and (r, c) not in st.session_state.hinted_cells:
                        st.session_state.history.append(st.session_state.grid.copy())
                        st.session_state.grid.iat[r, c] = solved.iat[r, c]
                        st.session_state.hinted_cells.add((r, c))
                        st.success("Hint applied successfully!")
                        st.stop()
            st.info("No more hints available")

    if st.button("✅ Solve", use_container_width=True):
        st.session_state.history.append(st.session_state.grid.copy())
        solved = st.session_state.grid.copy()
        if solve_sudoku(solved):
            st.session_state.grid = solved
            st.success("Success! The solution was found.")

    if st.button("🔁 Restart", use_container_width=True):
        st.session_state.grid = pd.DataFrame([[0]*9 for _ in range(9)])
        st.session_state.history.clear()
        st.session_state.hinted_cells.clear()
        st.session_state.solution = None

# ---------------- CONTENT SECTION ----------------
st.markdown("---")
st.header("❓ What is Neurodoku?")
st.write(
    "Neurodoku is an intelligent Sudoku-solving web application that assists users "
    "by validating moves, providing hints, and generating complete solutions."
)

st.header("📘 How to use Neurodoku?")
st.write("""
1. Click on a cell in the grid  
2. Enter numbers using the Number Pad  
3. Invalid moves are blocked  
4. Use Hint for guidance  
5. Use Solve to complete the puzzle  
6. Restart anytime  
""")

st.header("🤖 How is Neurodoku an AI Model?")
st.write("""
Neurodoku is an AI-based system because it simulates human-like logical reasoning.
It uses constraint satisfaction, backtracking search, and rule-based decision making
to evaluate valid moves and reach optimal solutions.
""")

st.markdown("### 🔍 Why Neurodoku?")
st.write("""
• Logical Reasoning  
• Decision Making  
• Automation  
• Problem Solving  
• Human-like Strategy Simulation  
""")
