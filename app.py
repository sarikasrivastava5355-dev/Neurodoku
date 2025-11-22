import streamlit as st
from PIL import Image
import numpy as np
import cv2
import pytesseract
import re

# Set the Tesseract executable path (important for Colab/non-standard installs)
# In Colab, tesseract is usually in /usr/bin/tesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def set_custom_style():
    """Applies custom CSS for background, text styles, sizes, and colors."""
    custom_css = """
    <style>
    body {
        background-color: beige;
    }
    h1 {
        color: #4B0082; /* Indigo */
        text-align: center; /* Center the title */
        font-family: "Black Chancery", cursive; /* Black Chancery font style */
        -webkit-text-stroke: 1px black; /* Black outline for the title */
        color: white; /* White color for the title text */
    }
    p {
        color: #FFB6C1; /* Light pink pastel color for welcome message */
        text-align: center; /* Center the welcome message */
        font-family: "High Tower Text", serif; /* High Tower font style */
        font-size: 16px; /* Font size 16 for welcome message */
    }
    .stMarkdown, .stException, .stWarning, .stInfo, .stSuccess, .stError {
        font-family: "Arial", sans-serif; /* Arial font style for other statements */
        font-size: 12px; /* Font size 12 for other statements */
        color: white; /* White color for statements */
    }
     /* Style for text inside buttons like structure */
    .stButton>button, .stRadio > label, .stFileUploader label {
        font-family: "Georgia", serif; /* Georgia font style for buttons and labels */
        color: #7CFC00; /* LawnGreen - a brighter green */
    }

    /* Add styles for other elements as needed */
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

def extract_sudoku_grid(image):
    """
    Extracts a 9x9 Sudoku grid from an image using OpenCV and Tesseract OCR.
    Returns a 2D list representing the grid, with 0 for empty cells.
    """
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # In a real application, you would find contours, the largest square, perform perspective transform,
    # and then process individual cells. For this example, we'll assume the grid is straight and use
    # image dimensions to infer cell boundaries.

    height, width = thresh.shape
    cell_h = height // 9
    cell_w = width // 9

    sudoku_grid = [[0 for _ in range(9)] for _ in range(9)]

    # Use pytesseract to find all digits in the thresholded image
    # output_type=pytesseract.Output.DICT gives a dictionary with bounding box info and text
    data = pytesseract.image_to_data(thresh, output_type=pytesseract.Output.DICT, config='--psm 6 digits')

    n_boxes = len(data['text'])
    for i in range(n_boxes):
        if int(data['conf'][i]) > 70: # Confidence threshold
            text = data['text'][i]
            # Use regex to ensure we only get single digits
            match = re.match(r'^\d$', text)
            if match:
                digit = int(match.group(0))
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]

                # Calculate center of the digit bounding box
                center_x = x + w // 2
                center_y = y + h // 2

                # Determine grid cell based on center coordinates
                col = min(center_x // cell_w, 8)
                row = min(center_y // cell_h, 8)

                # Place the digit in the grid
                if 1 <= digit <= 9:
                    sudoku_grid[row][col] = digit

    return sudoku_grid

def solve_sudoku(grid, get_hint=False, previous_hint=None):
    """
    Placeholder function to solve a Sudoku grid.

    Args:
        grid: A 2D list or numpy array representing the Sudoku grid (0 for empty cells).
        get_hint: If True, return a hint (e.g., the next step); otherwise, return the full solution.
        previous_hint: The hint previously provided, to avoid suggesting the same hint again.

    Returns:
        The solved grid (2D list/array) if get_hint is False, or a hint (e.g., tuple (row, col, value)) if get_hint is True.
        Returns None if the puzzle is unsolvable or no hint is available.
    """
    # --- Placeholder for the actual Sudoku solving logic (NEURODOKU model integration) ----
    # For demonstration purposes, let's create a dummy solved grid and a dummy hint
    if get_hint:
        dummy_hints = [
            (0, 0, 5), (0, 2, 4), (1, 1, 7), (1, 2, 2), (1, 6, 3), (1, 7, 4), (1, 8, 8),
            (2, 0, 1), (2, 3, 3), (2, 4, 4), (2, 5, 2), (2, 8, 7),
            (3, 1, 5), (3, 2, 9), (3, 3, 7), (3, 5, 1), (3, 7, 2),
            (4, 1, 2), (4, 2, 6), (4, 6, 7), (4, 7, 9),
            (5, 1, 1), (5, 2, 3), (5, 3, 9), (5, 5, 4), (5, 6, 8), (5, 7, 5),
            (6, 2, 1), (6, 3, 5), (6, 4, 3), (6, 5, 7), (6, 8, 4),
            (7, 0, 2), (7, 1, 8), (7, 2, 7), (7, 6, 6), (7, 7, 3), (7, 8, 5),
            (8, 0, 3), (8, 1, 4), (8, 2, 5), (8, 5, 6), (8, 6, 1), (8, 7, 7)
        ]
        # Filter out hints for cells that are already filled or were previously suggested
        available_hints = [h for h in dummy_hints if grid[h[0]][h[1]] == 0 and h != previous_hint]

        if available_hints:
            hint = available_hints[0]
            return hint
        else:
            return None

    else:
        solved_grid = [
            [5, 3, 4, 6, 7, 8, 9, 1, 2],
            [6, 7, 2, 1, 9, 5, 3, 4, 8],
            [1, 9, 8, 3, 4, 2, 5, 6, 7],
            [8, 5, 9, 7, 6, 1, 4, 2, 3],
            [4, 2, 6, 8, 5, 3, 7, 9, 1],
            [7, 1, 3, 9, 2, 4, 8, 5, 6],
            [9, 6, 1, 5, 3, 7, 2, 8, 4],
            [2, 8, 7, 4, 1, 9, 6, 3, 5],
            [3, 4, 5, 2, 8, 6, 1, 7, 9]
        ]
        return solved_grid

def display_grid(grid):
    """Displays the Sudoku grid in a formatted table."""
    st.write("Current Grid:")
    import pandas as pd
    df_grid = pd.DataFrame(grid)
    st.table(df_grid)


# --- Main Streamlit App Code ---
set_custom_style()

st.title("Neurodoku")

st.markdown("<p style='color: #FFB6C1; text-align: center; font-family: \"High Tower Text\", serif; font-size: 16px;'>Welcome to NEURODOKU! Your personal Sudoku assistant.</p>", unsafe_allow_html=True)


# Initialize session state variables
if 'puzzle_loaded' not in st.session_state:
    st.session_state.puzzle_loaded = False
if 'hint_provided' not in st.session_state:
    st.session_state.hint_provided = False
if 'solution_provided' not in st.session_state:
    st.session_state.solution_provided = False
if 'extracted_grid' not in st.session_state:
    st.session_state.extracted_grid = None
if 'solved_result' not in st.session_state:
    st.session_state.solved_result = None
if 'hint_result' not in st.session_state:
    st.session_state.hint_result = None
if 'previous_hint' not in st.session_state:
    st.session_state.previous_hint = None


# File uploader
uploaded_file = st.file_uploader("Choose a Sudoku image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and not st.session_state.puzzle_loaded:
    st.session_state.puzzle_loaded = True
    st.session_state.hint_provided = False
    st.session_state.solution_provided = False
    st.session_state.extracted_grid = None # Clear previous grid
    st.session_state.solved_result = None
    st.session_state.hint_result = None
    st.session_state.previous_hint = None # Clear previous hint

    image = Image.open(uploaded_file)
    st.write("File uploaded and puzzle loaded successfully!")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Extract grid using OCR
    with st.spinner('Extracting Sudoku grid from image...'):
        st.session_state.extracted_grid = extract_sudoku_grid(image)

    if st.session_state.extracted_grid:
        st.write("Sudoku grid extracted:")
        display_grid(st.session_state.extracted_grid)
    else:
        st.write("Could not extract Sudoku grid. Please try another image.")
        st.session_state.puzzle_loaded = False # Reset if extraction fails


# Display options only if a puzzle is loaded
if st.session_state.puzzle_loaded and st.session_state.extracted_grid:
    # Use buttons for hint or full solution
    col_hint, col_solution = st.columns(2)
    with col_hint:
        hint_button = st.button("Get a Hint")
    with col_solution:
        solution_button = st.button("Get Full Solution")

    if hint_button:
        st.write("Getting a hint...")
        if st.session_state.extracted_grid:
            hint_result = solve_sudoku(st.session_state.extracted_grid, get_hint=True, previous_hint=st.session_state.previous_hint)
            st.session_state.hint_result = hint_result
            st.session_state.hint_provided = True
            st.session_state.solution_provided = False
            if hint_result:
                st.session_state.previous_hint = hint_result # Store the provided hint
                st.write(f"Hint: Try placing **{hint_result[2]}** at row **{hint_result[0]+1}**, column **{hint_result[1]+1}**.")
            else:
                st.write("Could not find a new hint or puzzle is solved.")
        else:
            st.write("No puzzle grid available to provide a hint.")

    elif solution_button:
        st.write("Solving the full puzzle...")
        if st.session_state.extracted_grid:
            solved_result = solve_sudoku(st.session_state.extracted_grid, get_hint=False)
            st.session_state.solved_result = solved_result
            st.session_state.solution_provided = True
            st.session_state.hint_provided = False
            if solved_result:
                st.write("Here is the full solution:")
                display_grid(solved_result)
            else:
                st.write("Could not solve the puzzle.")
        else:
            st.write("No puzzle grid available to provide a solution.")

    # Add buttons for next steps after a hint or solution is provided
    if st.session_state.hint_provided or st.session_state.solution_provided:
        st.write("What would you like to do next?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Try Another Puzzle"):
                st.session_state.puzzle_loaded = False
                st.session_state.hint_provided = False
                st.session_state.solution_provided = False
                st.session_state.extracted_grid = None
                st.session_state.solved_result = None
                st.session_state.hint_result = None
                st.session_state.previous_hint = None
                st.experimental_rerun()

        with col2:
            if st.button("Continue with Current Puzzle"):
                 st.session_state.hint_provided = False
                 st.session_state.solution_provided = False
                 st.write("Continuing with the current puzzle. Choose an option above.")
                 st.experimental_rerun()

else:
    st.write("Please upload an image of a Sudoku puzzle to get started.")
