# Chess AI
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
