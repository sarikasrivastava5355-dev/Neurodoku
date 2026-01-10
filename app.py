import streamlit as st
import pandas as pd
import copy

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="NEURODOKU", layout="wide")

st.title("🧠 NEURODOKU")

# -------------------------------------------------
# INITIAL STATE
# -------------------------------------------------
if "grid" not in st.session_state:
    st.session_state.grid = pd.DataFrame([[0]*9 for _ in range(9)])
    st.session_state.history = []
    st.session_state.message = ""

# -------------------------------------------------
# SUDOKU SOLVER (BACKTRACKING)
# -------------------------------------------------
def is_valid(board, r, c, num):
    if num in board.iloc[r].values:
        return False
    if num in board.iloc[:, c].values:
        return False
    br, bc = (r//3)*3, (c//3)*3
    if num in board.iloc[br:br+3, bc:bc+3].values:
        return False
    return True

def solve_sudoku(board):
    for r in range(9):
        for c in range(9):
            if board.iat[r, c] == 0:
                for n in range(1, 10):
                    if is_valid(board, r, c, n):
                        board.iat[r, c] = n
                        if solve_sudoku(board):
                            return True
                        board.iat[r, c] = 0
                return False
    return True

# -------------------------------------------------
# MAIN LAYOUT
# -------------------------------------------------
left, right = st.columns([2.2, 1])

# ================= GRID =================
with left:
    st.subheader("Sudoku Grid")

    display_grid = st.session_state.grid.replace(0, "")
    st.dataframe(
        display_grid,
        hide_index=True,
        use_container_width=True
    )

    st.markdown("### Select Cell")
    r1, r2 = st.columns(2)
    row = r1.selectbox("Row", range(1, 10)) - 1
    col = r2.selectbox("Column", range(1, 10)) - 1

# ================= CONTROLS =================
with right:
    st.subheader("Controls")

    c1, c2, c3 = st.columns(3)

    if c1.button("↩ Undo"):
        if st.session_state.history:
            st.session_state.grid = st.session_state.history.pop()

    if c2.button("🧽 Erase"):
        st.session_state.history.append(st.session_state.grid.copy())
        st.session_state.grid.iat[row, col] = 0

    if c3.button("🗑 Clear All"):
        st.session_state.history.append(st.session_state.grid.copy())
        st.session_state.grid = pd.DataFrame([[0]*9 for _ in range(9)])

    st.markdown("### Number Pad")

    pad = st.columns(3)
    num = 1
    for i in range(3):
        for j in range(3):
            if pad[j].button(str(num), key=f"num{num}"):
                if is_valid(st.session_state.grid, row, col, num):
                    st.session_state.history.append(st.session_state.grid.copy())
                    st.session_state.grid.iat[row, col] = num
                else:
                    st.warning("Invalid move")
            num += 1

    st.markdown("---")

    if st.button("💡 Hint"):
        temp = st.session_state.grid.copy()
        if solve_sudoku(temp):
            st.session_state.history.append(st.session_state.grid.copy())
            st.session_state.grid.iat[row, col] = temp.iat[row, col]

    if st.button("✅ Solve"):
        st.session_state.history.append(st.session_state.grid.copy())
        solved = st.session_state.grid.copy()
        if solve_sudoku(solved):
            st.session_state.grid = solved
            st.session_state.message = "🎉 Success! The solution was found."

# -------------------------------------------------
# SUCCESS MESSAGE
# -------------------------------------------------
if st.session_state.message:
    st.success(st.session_state.message)

# -------------------------------------------------
# AI EXPLANATION SECTION
# -------------------------------------------------
st.markdown("---")
st.header("🤖 How is Neurodoku an AI Model?")

st.markdown("""
**Neurodoku is an AI-based system because it mimics human problem-solving and decision-making using algorithms.**

### 🔹 Reasons why Neurodoku is considered an AI model:

1. **Logical Reasoning**  
   Neurodoku applies constraint satisfaction rules similar to human reasoning to decide valid placements.

2. **Backtracking Algorithm**  
   The model explores multiple possible states, learns from wrong paths, and corrects itself — a key AI trait.

3. **Decision Making**  
   At every step, Neurodoku evaluates possible numbers and selects the optimal valid choice.

4. **Automation of Intelligence**  
   It automatically solves Sudoku without human intervention, simulating intelligent behavior.

5. **Future AI Scope**  
   The model can be extended using **Computer Vision** and **Neural Networks** to detect Sudoku from images.

➡️ Hence, Neurodoku qualifies as an **Artificial Intelligence application based on rule-based reasoning and search algorithms**.
""")
