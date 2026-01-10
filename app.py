import streamlit as st
import pandas as pd

st.set_page_config(page_title="NEURODOKU", layout="wide")

# ---------- HEADER ----------
st.markdown("<div style='text-align:center;'>", unsafe_allow_html=True)
st.image("logo.png", width=90)
st.markdown("<h1>NEURODOKU</h1>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------- SESSION STATE ----------
if "grid" not in st.session_state:
    st.session_state.grid = [[0]*9 for _ in range(9)]
if "selected" not in st.session_state:
    st.session_state.selected = (0, 0)
if "history" not in st.session_state:
    st.session_state.history = []
if "hinted" not in st.session_state:
    st.session_state.hinted = set()
if "hint_used" not in st.session_state:
    st.session_state.hint_used = False

# ---------- LOGIC ----------
def valid(grid, r, c, n):
    if n in grid[r]: return False
    if n in [grid[i][c] for i in range(9)]: return False
    br, bc = (r//3)*3, (c//3)*3
    for i in range(br, br+3):
        for j in range(bc, bc+3):
            if grid[i][j] == n:
                return False
    return True

def solve(grid):
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                for n in range(1,10):
                    if valid(grid, r, c, n):
                        grid[r][c] = n
                        if solve(grid):
                            return True
                        grid[r][c] = 0
                return False
    return True

# ---------- LAYOUT ----------
left, right = st.columns([3,1])

# ---------- GRID ----------
with left:
    st.subheader("Sudoku Grid")
    for r in range(9):
        cols = st.columns(9)
        for c in range(9):
            v = st.session_state.grid[r][c]
            label = str(v) if v != 0 else ""
            if cols[c].button(label, key=f"cell{r}{c}", use_container_width=True):
                st.session_state.selected = (r, c)

# ---------- CONTROLS ----------
with right:
    st.subheader("Controls")

    if st.button("↩ Undo", use_container_width=True):
        if st.session_state.history:
            st.session_state.grid = st.session_state.history.pop()

    if st.button("🧹 Erase", use_container_width=True):
        r,c = st.session_state.selected
        st.session_state.history.append([row[:] for row in st.session_state.grid])
        st.session_state.grid[r][c] = 0

    if st.button("🗑 Clear All", use_container_width=True):
        st.session_state.grid = [[0]*9 for _ in range(9)]
        st.session_state.history.clear()
        st.session_state.hinted.clear()
        st.session_state.hint_used = False

    st.markdown("### Number Pad")
    for i in range(0,9,3):
        row = st.columns(3)
        for j in range(3):
            num = i+j+1
            if row[j].button(str(num), use_container_width=True):
                r,c = st.session_state.selected
                if valid(st.session_state.grid, r, c, num):
                    st.session_state.history.append([row[:] for row in st.session_state.grid])
                    st.session_state.grid[r][c] = num
                else:
                    st.warning("Invalid move")

    st.markdown("---")

    if st.button("💡 Hint", use_container_width=True):
        temp = [row[:] for row in st.session_state.grid]
        solve(temp)
        for r in range(9):
            for c in range(9):
                if st.session_state.grid[r][c] == 0 and (r,c) not in st.session_state.hinted:
                    st.session_state.history.append([row[:] for row in st.session_state.grid])
                    st.session_state.grid[r][c] = temp[r][c]
                    st.session_state.hinted.add((r,c))
                    st.session_state.hint_used = True
                    st.success("Hint applied successfully!")
                    break
            else:
                continue
            break

    if st.session_state.hint_used:
        if st.button("More Hint", use_container_width=True):
            st.session_state.hint_used = False
            st.experimental_rerun()

        if st.button("Solve Full Puzzle", use_container_width=True):
            solve(st.session_state.grid)
            st.success("Puzzle solved successfully!")

        if st.button("Restart", use_container_width=True):
            st.session_state.grid = [[0]*9 for _ in range(9)]
            st.session_state.history.clear()
            st.session_state.hinted.clear()
            st.session_state.hint_used = False

    if st.button("✅ Solve", use_container_width=True):
        solve(st.session_state.grid)
        st.success("Success! The solution was found.")

# ---------- INFO SECTION ----------
st.markdown("---")

st.header("📘 How to use Neurodoku?")
st.write("""
• Click a cell  
• Enter digits using Number Pad  
• Use Hint for guidance  
• Solve for full solution  
• Restart anytime  
""")

st.header("❓ What is Neurodoku?")
st.write("Neurodoku is an intelligent Sudoku-solving web application that assists users by validating moves, providing hints, and generating complete solutions.")

st.header("🤖 How is Neurodoku an AI Model?")
st.write("""
Neurodoku mimics human logical reasoning using constraint satisfaction and backtracking.
It evaluates possibilities, eliminates invalid options, and makes optimal decisions —
core characteristics of Artificial Intelligence systems.
""")
