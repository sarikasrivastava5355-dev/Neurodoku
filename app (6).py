import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tensorflow as tf
import copy # Import copy for deepcopy

# Load the pre-trained CNN model only once into Streamlit's session state
@st.cache_resource
def load_cnn_model():
    return tf.keras.models.load_model('digit_model.h5')

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

def order_points(pts):
    """Orders a list of 4 points in top-left, top-right, bottom-right, bottom-left fashion."""
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_sudoku_grid(image_cv):
    """Finds the largest square contour (assumed to be the Sudoku grid) in an image."""
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    largest_contour = None
    max_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) == 4:
                if area > max_area:
                    largest_contour = approx
                    max_area = area
    return largest_contour

def extract_sudoku_grid(image):
    """
    Extracts a 9x9 Sudoku grid from an image using OpenCV for grid localization
    and a CNN for digit recognition. Returns a 2D list representing the grid,
    with 0 for empty cells, or None if grid not found.
    """
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    grid_contour = find_sudoku_grid(img_cv)

    if grid_contour is None:
        return None # No Sudoku grid found

    # Get the ordered four corner points of the grid
    pts = grid_contour.reshape(4, 2)
    rect = order_points(pts)

    # Set desired output size for the warped image (e.g., 450x450 for better cell resolution)
    output_size = 450 # Standard size, divisible by 9
    dst = np.array([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]], dtype = "float32")

    # Compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(img_cv, M, (output_size, output_size))

    # Preprocess the warped image for digit extraction
    warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    warped_blur = cv2.GaussianBlur(warped_gray, (5, 5), 0)
    _, warped_thresh = cv2.threshold(warped_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    cell_h = output_size // 9
    cell_w = output_size // 9

    sudoku_grid = [[0 for _ in range(9)] for _ in range(9)]

    cnn_model = load_cnn_model()

    for r in range(9):
        for c in range(9):
            x1 = c * cell_w
            y1 = r * cell_h
            x2 = (c + 1) * cell_w
            y2 = (r + 1) * cell_h

            cell_img = warped_thresh[y1:y2, x1:x2]

            cell_contours, _ = cv2.findContours(cell_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if len(cell_contours) > 0:
                largest_digit_contour = max(cell_contours, key=cv2.contourArea)

                x_digit, y_digit, w_digit, h_digit = cv2.boundingRect(largest_digit_contour)

                if w_digit < 10 or h_digit < 10 or w_digit > cell_w * 0.8 or h_digit > cell_h * 0.8:
                    continue

                pad = 4
                x_digit_start = max(0, x_digit - pad)
                y_digit_start = max(0, y_digit - pad)
                x_digit_end = min(cell_img.shape[1], x_digit + w_digit + pad)
                y_digit_end = min(cell_img.shape[0], y_digit + h_digit + pad)

                digit_roi = cell_img[y_digit_start:y_digit_end, x_digit_start:x_digit_end]

                if digit_roi.shape[0] > 0 and digit_roi.shape[1] > 0:
                    digit_resized = cv2.resize(digit_roi, (28, 28), interpolation=cv2.INTER_AREA)
                    digit_normalized = digit_resized.astype('float32') / 255.0
                    digit_input = np.expand_dims(digit_normalized, axis=(0, -1))

                    predictions = cnn_model.predict(digit_input, verbose=0)
                    predicted_digit = np.argmax(predictions[0])
                    confidence = np.max(predictions[0])

                    if confidence > 0.9 and predicted_digit != 0:
                        sudoku_grid[r][c] = predicted_digit

    return sudoku_grid

# Helper function 1: Find an empty cell
def find_empty(grid):
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                return (r, c)  # (row, col)
    return None

# Helper function 2: Check if placing a number is valid
def is_valid(grid, num, pos):
    row, col = pos

    # Check row
    for c in range(9):
        if grid[row][c] == num and col != c:
            return False

    # Check column
    for r in range(9):
        if grid[r][col] == num and row != r:
            return False

    # Check 3x3 box
    box_x = col // 3
    box_y = row // 3

    for r_box in range(box_y * 3, box_y * 3 + 3):
        for c_box in range(box_x * 3, box_x * 3 + 3):
            if grid[r_box][c_box] == num and (r_box, c_box) != pos:
                return False

    return True

# Main Sudoku solver function (backtracking)
def solve(grid):
    find = find_empty(grid)
    if not find:
        return True  # Puzzle solved
    else:
        row, col = find

    for num in range(1, 10):
        if is_valid(grid, num, (row, col)):
            grid[row][col] = num

            if solve(grid):
                return True

            grid[row][col] = 0  # Backtrack

    return False

def solve_sudoku(grid, get_hint=False, previous_hint=None):
    """
    Solves a Sudoku grid using backtracking.
    If get_hint is True, returns a valid next step; otherwise, returns the full solution.
    """
    if get_hint:
        copied_grid = copy.deepcopy(grid)
        empty_cell = find_empty(copied_grid)

        if empty_cell:
            r, c = empty_cell
            for num in range(1, 10):
                if is_valid(copied_grid, num, (r, c)):
                    return (r, c, num)
        return None

    else:
        copied_grid = copy.deepcopy(grid)
        if solve(copied_grid):
            return copied_grid
        else:
            return None

def display_grid(grid):
    """Displays the Sudoku grid in a formatted table."""
    st.write("Current Grid:")
    import pandas as pd
    display_grid_data = [[str(cell) if cell != 0 else '' for cell in row] for row in grid]
    df_grid = pd.DataFrame(display_grid_data)
    st.table(df_grid)


# --- Main Streamlit App Code ---
set_custom_style()

st.title("Neurodoku")

# Add the logo here
st.image("logo.png", caption="NEURODOKU Logo", width=150)

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

    # Extract grid using CNN model
    with st.spinner('Extracting Sudoku grid from image using CNN...'):
        st.session_state.extracted_grid = extract_sudoku_grid(image)

    if st.session_state.extracted_grid:
        st.write("Sudoku grid extracted:")
        display_grid(st.session_state.extracted_grid)
    else:
        st.error("Could not extract Sudoku grid. Please try another image with a clear Sudoku puzzle.")
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
            # Pass the current grid to solve_sudoku for hinting
            hint_result = solve_sudoku(st.session_state.extracted_grid, get_hint=True, previous_hint=st.session_state.previous_hint)
            st.session_state.hint_result = hint_result
            st.session_state.hint_provided = True
            st.session_state.solution_provided = False
            if hint_result:
                st.session_state.previous_hint = hint_result # Store the provided hint
                st.write(f"Hint: Try placing **{hint_result[2]}** at row **{hint_result[0]+1}**, column **{hint_result[1]+1}**.")
            else:
                st.write("Could not find a new hint or puzzle is already solved.")
        else:
            st.write("No puzzle grid available to provide a hint.")

    elif solution_button:
        st.write("Solving the full puzzle...")
        if st.session_state.extracted_grid:
            # Pass the extracted grid to solve_sudoku for full solution
            solved_result = solve_sudoku(st.session_state.extracted_grid, get_hint=False)
            st.session_state.solved_result = solved_result
            st.session_state.solution_provided = True
            st.session_state.hint_provided = False
            if solved_result:
                st.write("Here is the full solution:")
                display_grid(solved_result)
            else:
                st.error("Could not solve the puzzle. It might be invalid or too difficult.")
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
