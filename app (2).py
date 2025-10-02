import streamlit as st
from PIL import Image
import numpy as np
import cv2

def set_custom_style():
    """Applies custom CSS for background and text styles."""
    custom_css = """
    <style>
    body {
        background-color: beige;
    }
    h1 {
        color: #4B0082; /* Indigo */
        text-align: center; /* Center the title */
    }
    p {
        color: #556B2F; /* DarkOliveGreen */
        text-align: center; /* Center the welcome message */
    }
    .stRadio > label {
        color: #556B2F; /* DarkOliveGreen for radio button labels */
    }
    /* Add styles for other elements as needed */
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

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
    # --- Placeholder for the actual Sudoku solving logic (NEURODOKU model integration) ---
    # This is where you would integrate your NEURODOKU model or a standard Sudoku solver.
    # The solver would take the 'grid' as input and find the solution.

    # For demonstration purposes, let's create a dummy solved grid and a dummy hint
    # In a real implementation, this would be the output of your solver.

    if get_hint:
        # Dummy hint logic: Provide a different hint each time (up to a limit)
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
            # Return the first available hint (can be randomized in a real app)
            hint = available_hints[0]
            return hint
        else:
            return None # No new hints available or puzzle is solved

    else:
        # Dummy solved grid: A simple valid Sudoku grid for demonstration
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
    # Create a pandas DataFrame for better table display
    import pandas as pd
    df_grid = pd.DataFrame(grid)
    st.table(df_grid)


# --- Main Streamlit App Code ---
set_custom_style()

st.title("NEURODOKU")

# Placeholder for a logo
# st.image("path/to/your/logo.png", caption="NEURODOKU Logo", width=100)

# Display a welcome message with different color
st.markdown("<p style='color: #FF4500; text-align: center;'>Welcome to NEURODOKU! Your personal Sudoku assistant.</p>", unsafe_allow_html=True)


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
    # Process the image and store the extracted grid in session_state
    image = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # --- Placeholder for Sudoku Grid Extraction ---
    # In a real app, extracted_grid would come from image processing
    st.session_state.extracted_grid = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ] # Replace with actual extraction result
    # --- End of Placeholder ---

    st.write("File uploaded and puzzle loaded successfully!")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    display_grid(st.session_state.extracted_grid)


# Display options only if a puzzle is loaded
if st.session_state.puzzle_loaded:
    st.write("Sudoku grid extracted.")

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
                # Optionally display the grid with the hint highlighted (requires more complex rendering)
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
                display_grid(solved_result) # Display the solved grid using the function
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
                # Clear session state for a new puzzle
                st.session_state.puzzle_loaded = False
                st.session_state.hint_provided = False
                st.session_state.solution_provided = False
                st.session_state.extracted_grid = None
                st.session_state.solved_result = None
                st.session_state.hint_result = None
                st.session_state.previous_hint = None
                st.experimental_rerun() # Rerun the app to show file uploader

        with col2:
            if st.button("Continue with Current Puzzle"):
                 # This will effectively just leave the current state as is and re-display options
                 st.session_state.hint_provided = False # Reset these to allow choosing again
                 st.session_state.solution_provided = False
                 st.write("Continuing with the current puzzle. Choose an option above.")
                 st.experimental_rerun() # Rerun to show options again

else:
    # If no puzzle loaded, show the welcome message and file uploader
    st.write("Please upload an image of a Sudoku puzzle to get started.")
