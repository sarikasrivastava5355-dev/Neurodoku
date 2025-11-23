# app.py — NEURODOKU (6x6, 9x9, 16x16 Sudoku Solver + Tesseract OCR + Blue Tech UI)

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
import cv2
import pytesseract
import re
import math

# Tesseract path for Streamlit Cloud
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

st.set_page_config(page_title="NEURODOKU", layout="centered")

# ---------- Blue Tech CSS ----------
st.markdown("""
    <style>
    body { background: linear-gradient(135deg, #0e1a2b, #112b45); color: #e6eef8; font-family: 'Segoe UI',sans-serif;}
    .main-title{font-size:38px; font-weight:800; text-align:center;
        background:linear-gradient(90deg,#61dafb,#1e88e5); -webkit-background-clip:text; color:transparent;}
    .subtitle {text-align:center; color:#9fd6ff; margin-bottom:18px;}
    .card { background: rgba(255,255,255,0.03); padding:18px; border-radius:14px; box-shadow:0 8px 24px rgba(3, 169, 244, 0.06); border:1px solid rgba(30,136,229,0.08); }
    .stButton>button { background: linear-gradient(90deg,#0288d1,#0277bd); color:white; border-radius:10px; padding:8px 16px; font-weight:700; }
    table { border-collapse:collapse; margin-top:12px; }
    td { width:40px; height:40px; text-align:center; font-weight:700; font-size:18px; color:#e8f6ff; border:1px solid rgba(30,136,229,0.25); }
    .highlight { background: #ffd54f; color:#001; }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>🧠 NEURODOKU</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload Sudoku image — Supports 6×6, 9×9, 16×16 • OCR Powered</div>", unsafe_allow_html=True)

st.markdown("<div class='card'>", unsafe_allow_html=True)

uploaded = st.file_uploader("Upload Sudoku image (photo / screenshot)", type=["png","jpg","jpeg"])


# ---------- Utility functions ----------

def find_largest_contour(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edge = cv2.Canny(blur, 50, 150)
    cnts,_ = cv2.findContours(edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx)==4:
            return approx.reshape(4,2)
    return None


def order_pts(pts):
    rect = np.zeros((4,2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def warp_to_square(gray, pts):
    rect = order_pts(pts)
    (tl,tr,br,bl) = rect
    widthA = np.linalg.norm(br-bl)
    widthB = np.linalg.norm(tr-tl)
    heightA = np.linalg.norm(tr-br)
    heightB = np.linalg.norm(tl-bl)
    side = int(max(widthA,widthB,heightA,heightB))
    dst = np.array([[0,0],[side-1,0],[side-1,side-1],[0,side-1]],dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(gray, M, (side,side))


def ocr_cell(img, whitelist):
    if len(img.shape)==3:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    _,th = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    pil = Image.fromarray(th)
    config = f'--psm 10 -c tessedit_char_whitelist={whitelist}'
    txt = pytesseract.image_to_string(pil, config=config)
    txt = re.sub(r'[^0-9A-Za-z]', '', txt)
    return txt.strip()


def test_sizes(warp, sizes=(6,9,16)):
    best=None
    side = warp.shape[0]
    for n in sizes:
        cell = side//n
        grid=[[0]*n for _ in range(n)]
        count=0
        for i in range(n):
            for j in range(n):
                y=i*cell; x=j*cell
                m=max(4,cell//12)
                crop=warp[y+m:y+cell-m, x+m:x+cell-m]
                whitelist="0123456789ABCDEFabcdef" if n==16 else "0123456789"
                txt=ocr_cell(crop, whitelist)
                if txt:
                    if n==16:
                        ch=txt[0].upper()
                        try:
                            val=int(ch,16)
                        except:
                            val=0
                    else:
                        try: val=int(txt[0])
                        except: val=0
                    if val!=0:
                        grid[i][j]=val
                        count+=1
        ratio=count/(n*n)
        if best is None or ratio>best[2]:
            best=(n,grid,ratio)
    return best


def convert_to_strings(grid, n):
    out=[]
    for r in grid:
        row=[]
        for v in r:
            if v==0:
                row.append("")
            else:
                if n==16 and v>9:
                    row.append(format(v,"X"))
                else:
                    row.append(str(v))
        out.append(row)
    return out


def box_shape(n):
    if n==6: return 2,3
    root=int(math.sqrt(n))
    if root*root==n: return root,root
    for r in range(root,0,-1):
        if n%r==0:
            return r,n//r
    return 1,n


def find_empty(board,n):
    for i in range(n):
        for j in range(n):
            if board[i][j]==0:
                return (i,j)
    return None


def valid(board,r,c,num,n,br,bc):
    for x in range(n):
        if board[r][x]==num or board[x][c]==num:
            return False
    rs=(r//br)*br
    cs=(c//bc)*bc
    for i in range(rs,rs+br):
        for j in range(cs,cs+bc):
            if board[i][j]==num:
                return False
    return True


def solve(board,n,br,bc):
    pos=find_empty(board,n)
    if not pos: return True
    r,c=pos
    for num in range(1,n+1):
        if valid(board,r,c,num,n,br,bc):
            board[r][c]=num
            if solve(board,n,br,bc): return True
            board[r][c]=0
    return False


def display_grid(grid,title="Grid",highlight=None):
    size=len(grid)
    html="<table>"
    for i in range(size):
        html+="<tr>"
        for j in range(size):
            val=grid[i][j]
            is_h = highlight and (i,j)==highlight
            style = "background:#ffd54f;color:#001;" if is_h else ""
            html+=f"<td style='{style}'>{val}</td>"
        html+="</tr>"
    html+="</table>"
    st.write(f"### {title}")
    st.write(html, unsafe_allow_html=True)



# ---------- Main App Logic ----------
if uploaded:
    img=Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    arr=cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    gray=cv2.cvtColor(arr,cv2.COLOR_BGR2GRAY)

    pts=find_largest_contour(gray)
    if pts is None:
        st.error("Could not detect Sudoku grid. Try cropping closer.")
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        warp=warp_to_square(gray,pts)
        n,grid,ratio = test_sizes(warp,(6,9,16))

        st.info(f"Detected {n}×{n} puzzle • {ratio*100:.1f}% digits detected")

        grid_str=convert_to_strings(grid,n)
        display_grid(grid_str,"Extracted Grid")

        if ratio<0.12:
            st.warning("Very few digits detected. Upload a clearer image.")
        else:
            c1,c2,c3=st.columns(3)
            if c1.button("Solve Puzzle"):
                board=[row[:] for row in grid]
                br,bc=box_shape(n)
                ok=solve(board,n,br,bc)
                if ok:
                    disp=convert_to_strings(board,n)
                    display_grid(disp,"Solved Grid")
                    st.session_state["solution"]=board
                else:
                    st.error("Sudoku not solvable (OCR error likely).")

            if c2.button("Hint"):
                if "solution" not in st.session_state:
                    board=[row[:] for row in grid]
                    br,bc=box_shape(n)
                    if solve(board,n,br,bc):
                        st.session_state["solution"]=board
                    else:
                        st.error("Cannot solve → cannot give hint.")
                        st.stop()

                sol=st.session_state["solution"]
                for i in range(n):
                    for j in range(n):
                        if grid[i][j]==0:
                            val=sol[i][j]
                            val_disp = format(val,"X") if n==16 and val>9 else val
                            display_grid(convert_to_strings(sol,n),"Hint (Highlighted)",highlight=(i,j))
                            st.info(f"Hint → place **{val_disp}** at row {i+1}, column {j+1}")
                            st.stop()
                st.success("Puzzle already complete.")

            if c3.button("Reset"):
                for k in list(st.session_state.keys()):
                    del st.session_state[k]
                st.experimental_rerun()

st.markdown("</div>", unsafe_allow_html=True)
