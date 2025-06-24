# Chess AI

#### Chess_AI.ipynb:
This project develops a simple yet effective Chess AI using Python, the python-chess library, and TensorFlow/Keras. The AI learns to evaluate chess board positions and select the best move based on a trained neural network. A key feature of this project is the exploration of phase-specific models (opening, midgame, endgame) to enhance the AI's strategic understanding throughout a game.

## Features
- Chess Board Representation: Encodes standard chess board positions into a numerical format suitable for neural networks.
- Neural Network for Board Evaluation: Uses a simple feed-forward neural network to predict the "score" (evaluation) of a given board state.
- Move Selection: Determines the best move by evaluating all legal moves and choosing the one that leads to the most favorable board state according to the neural network.
- Phase-Specific Models: Divides the game into opening, midgame, and endgame phases based on material count, training a specialized neural network for each phase to improve strategic play.
- Interactive Play: Allows a human player to play against the AI through a command-line interface.

## How it Works
### Board Encoding
Chess boards are represented using a one-hot encoding scheme. Each square on the 8x8 board is encoded into a 13-element vector:

- 12 elements for each type of chess piece (Pawn, Knight, Bishop, Rook, Queen, King for both White and Black).
- 1 element for an empty square.
This results in an 8x8x13 NumPy array for each board state.

### Neural Network Architecture
The project uses a simple feed-forward neural network implemented with Keras:

- Flatten(): Flattens the 8x8x13 input into a single vector.
- Dense(1024, activation='relu'): A hidden layer with 1024 neurons and ReLU activation.
- Dense(64, activation='relu') (in the initial model): Another hidden layer with 64 neurons and ReLU activation.
- Dense(1): An output layer with a single neuron, predicting the board's evaluation (black's score).
The model is compiled with the rmsprop optimizer and mean_squared_error loss function.

### Phase-Specific Models
To account for the different strategic nuances in various stages of a chess game, the training data is segmented based on the total_material (sum of piece values for both sides) remaining on the board:

- Endgame Model: For boards with total_material < 30.
- Midgame Model: For boards with 30 <= total_material <= 60.
- Opening Model: For boards with total_material > 60.
A separate neural network is trained for each of these phases. During gameplay, the AI dynamically selects the appropriate model based on the current board's material count.

### Move Selection
For a given board state, the AI:

1. Iterates through all legal moves.
2. For each legal move, it simulates the move to create a candidate_board.
3. Encodes the candidate_board into the numerical format.
4. Feeds the encoded candidate_board into the appropriate trained neural network (opening, midgame, or endgame model) to get a score (evaluation).
5. If it's Black's turn, it chooses the move that maximizes the score (higher score is better for Black).
6. If it's White's turn, it chooses the move that minimizes the score (lower score is better for White, as the score represents Black's advantage).

### Training
The models are trained using the provided train.csv dataset. The training process involves:

- Loading FEN strings and black_score from the CSV.
- Encoding FEN strings into the 3D numpy array format.
- Splitting data into training and validation sets.
- Training the neural networks for a specified number of epochs (e.g., 20 or 40).
- Visualizing the training and validation loss curves to monitor performance.

### Playing Against the AI
The play_game(ai_function) function allows you to play against the trained AI:

- The human player plays as White, and the AI plays as Black.
- You enter your moves in Standard Algebraic Notation (e.g., e2e4, g1f3).
- The game continues until a checkmate, stalemate, or resignation.

### Results and Observations
- The loss curves (matplotlib.pyplot) provide insights into the training process, showing how well the models generalize to unseen data.
- The phase-specific models (play_nn2) are expected to exhibit more strategic play compared to the single general model (play_nn), as they are trained on data relevant to specific game stages.
- The AI's performance is limited by the simplicity of the model architecture and the dataset. More complex networks, larger datasets, and advanced techniques (e.g., MCTS, Alpha-Beta pruning combined with NN evaluation) would yield stronger play.

### Future Enhancements
- More Complex Neural Networks: Experiment with CNNs (Convolutional Neural Networks) for better spatial feature extraction from the board.
- Reinforcement Learning: Train the AI using reinforcement learning techniques (e.g., Q-learning, Policy Gradients) by having it play against itself.
- Search Algorithms: Integrate search algorithms like Minimax or Alpha-Beta pruning, using the neural network as an evaluation function, to explore deeper into possible move sequences.
- Larger Dataset: Utilize a much larger dataset of professional chess games for training.
- Advanced Features: Incorporate opening books, endgame tablebases, and more sophisticated evaluation features.
- GUI: Develop a graphical user interface for a more intuitive playing experience.

![Screenshot (184)](https://github.com/user-attachments/assets/01d96183-e0b1-44f6-9d59-c30b7ca88c4d)

# Enhanced Chess AI with CNN, Alpha-Beta Search, and GUI

#### Enhanced_Chess_AI_with_CNN,_Alpha_Beta_Search,_and_GUI.ipynb:
This project takes a significant leap forward in developing a Chess AI using Python, python-chess, TensorFlow/Keras, and Pygame. Building upon a previous iteration, this version integrates a more sophisticated Convolutional Neural Network (CNN) for board evaluation, implements the Alpha-Beta pruning algorithm for intelligent move selection, and introduces a basic graphical user interface (GUI) for interactive play (simulated in Colab).

The primary aim of this project is for learning and demonstrating advanced AI concepts applied to chess, specifically in neural network architecture, search algorithms, and basic GUI integration in a challenging environment like Google Colab.

## Features
- Convolutional Neural Network (CNN) for Board Evaluation: Leverages CNNs to better extract spatial features and patterns from the chessboard, leading to more nuanced board evaluations compared to simple feed-forward networks.
- Minimax with Alpha-Beta Pruning: Implements a powerful search algorithm that uses the trained CNN as its evaluation function to explore move sequences and select the optimal move, significantly enhancing the AI's strategic depth.
- Pygame-based Graphical User Interface (GUI): Provides a visual representation of the chessboard. In Google Colab, this GUI renders board states as images, allowing for a simulated interactive experience where the user inputs moves via text.
- Human vs. AI Play: Allows a human player to choose their side (White or Black) and play against the AI, inputting moves using Universal Chess Interface (UCI) notation (e.g., e2e4, g1f3).
- Board Encoding: Continues to use a one-hot encoding scheme to represent board positions as 3D NumPy arrays (8x8x13) for neural network input.

## How it Works
1. Data Preparation and Feature Engineering
The project uses a dataset (downloaded from a Kaggle competition) containing chess board FEN strings and corresponding black_score evaluations.

- One-hot Encoding: Each chess piece (including an empty square) is one-hot encoded into a 13-element vector.
- Board Encoding: An entire chessboard is transformed into an 8x8x13 NumPy array, suitable as input for a Convolutional Neural Network.
- Dataset Split: The train.csv dataset is split into training (90%) and validation (10%) sets.
2. Convolutional Neural Network (CNN)
Instead of simple feed-forward networks, this version employs a CNN to learn more complex spatial relationships on the board.

- Architecture:
  - Conv2D(32, (3, 3), padding='same', input_shape=(8, 8, 13)): First convolutional layer with 32 filters, 3x3 kernel, and ReLU activation.
  - Conv2D(32, (3, 3), padding='same'): Second convolutional layer with 32 filters, 3x3 kernel, and ReLU activation.
  - Flatten(): Flattens the output for the dense layers.
  - Dense(64, activation='relu'): A hidden dense layer with 64 neurons and ReLU activation.
  - Dense(1): Output layer with a single neuron predicting the black_score (evaluation from Black's perspective).
- Compilation: The model is compiled with the rmsprop optimizer and mean_squared_error loss function.
- Training: The CNN is trained for 15 epochs, monitoring training and validation loss.
3. Search Algorithm (Minimax with Alpha-Beta Pruning)
The trained CNN serves as the evaluation function for a Minimax search algorithm enhanced with Alpha-Beta pruning.

- evaluate_board(board): Takes a chess.Board object, encodes it, and uses the cnn_model.predict to get a score. The score is adjusted to be from the perspective of the board.turn (current player).
- minimax_alpha_beta(board, depth, alpha, beta, maximizing_player): Recursively explores the game tree to a specified depth.
  - depth: How many moves ahead the AI looks.
  - alpha, beta: Used for pruning branches of the search tree that won't affect the final decision, significantly speeding up the search.
  - maximizing_player: Boolean indicating if the current call is trying to maximize (for the current player) or minimize (for the opponent).
- get_ai_move(board, depth): Iterates through all legal moves, calls minimax_alpha_beta for each, and selects the move that yields the best evaluation for the current player.
4. Graphical User Interface (GUI)
A basic GUI is implemented using Pygame to visualize the board. Important Note for Colab Users: Due to the nature of Google Colab's remote execution environment, direct mouse interaction (like dragging pieces) is not possible. The GUI works by:

- Drawing the board and pieces to an off-screen Pygame surface.
- Converting this surface to an image (PNG).
- Displaying this image directly in the Colab output cell.
- User interaction for moves is handled via text input (UCI format) in the Colab cell prompt.

## Results and Observations
- The loss curves generated during training provide insights into the CNN model's learning progress and generalization ability.
- The integration of Alpha-Beta pruning allows the AI to make more strategic moves by looking ahead, even at a relatively shallow depth=2.
- The Pygame GUI visually updates the board after each move, offering a clear representation of the game state within the Colab output.
- The AI's performance can be further improved with deeper search depths (at the cost of longer computation times) and more extensive training data.

## Future Enhancements
- Optimized CNN Architecture: Experiment with more advanced CNN layers, architectures (e.g., Residual connections), and hyperparameter tuning.
- Larger Dataset: Utilize a significantly larger dataset of professional chess games to train a more robust evaluation function.
- Transposition Tables: Implement transposition tables in the Alpha-Beta search to store previously evaluated board states and avoid redundant calculations.
- Iterative Deepening: Combine iterative deepening with Alpha-Beta to find the best move within a time limit.
- Opening Books and Endgame Tablebases: Integrate pre-computed opening move sequences and perfect endgame solutions for common positions.
- Reinforcement Learning: Explore training the AI using reinforcement learning (e.g., AlphaZero-style self-play) for potentially superhuman performance.

## Technologies Used
- Python
- python-chess: For chess board representation and move generation.
- pygame: For the graphical board visualization (running in dummy mode for Colab).
- tensorflow.keras: For building and training the Convolutional Neural Network.
- numpy: For numerical operations, especially board encoding.
- pandas: For data handling (CSV file).
- matplotlib.pyplot: For visualizing training loss.
- IPython.display: For displaying images and managing Colab output.
- PIL (Pillow): For image manipulation to display Pygame output in Colab.

![Screenshot (185)](https://github.com/user-attachments/assets/966b4a3d-c5f9-4c14-8afe-cf64eb54cfed)
