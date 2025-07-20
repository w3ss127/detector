import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import statsapi
import sys

# NIPALS PCA implementation
def nipals_pca(X, n_components=3, tol=1e-6, max_iter=100):
    X = X.copy()  # Avoid modifying original data
    n_samples, n_features = X.shape
    # Center the data
    X_mean = np.mean(X, axis=0)
    X -= X_mean
    # Initialize outputs
    T = np.zeros((n_samples, n_components))  # Scores
    P = np.zeros((n_features, n_components))  # Loadings
    for i in range(n_components):
        # Initialize random score vector
        t = X[:, 0].copy()
        for _ in range(max_iter):
            # Project X onto t to get loading p
            p = X.T @ t / (t.T @ t)
            p /= np.sqrt(p.T @ p)  # Normalize
            # Project X onto p to get new score t_new
            t_new = X @ p
            # Check convergence
            if np.linalg.norm(t_new - t) < tol:
                break
            t = t_new
        # Store score and loading
        T[:, i] = t
        P[:, i] = p
        # Deflate X
        X -= np.outer(t, p)
    return T, P, X_mean

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Sample data for training (bypassing pybaseball issues)
data = {
    'home_team_win_rate': [0.600, 0.550, 0.650, 0.580, 0.620, 0.570, 0.610, 0.590, 0.630, 0.560, 0.640, 0.585, 0.605, 0.575, 0.595],
    'home_team_era': [3.50, 4.00, 3.20, 3.70, 3.60, 3.90, 3.55, 3.85, 3.45, 4.10, 3.30, 3.75, 3.65, 3.95, 3.80],
    'home_team_batting_avg': [0.300, 0.280, 0.310, 0.290, 0.295, 0.285, 0.305, 0.275, 0.315, 0.270, 0.320, 0.290, 0.300, 0.280, 0.310],
    'away_team_win_rate': [0.550, 0.600, 0.500, 0.570, 0.580, 0.560, 0.590, 0.610, 0.540, 0.620, 0.530, 0.600, 0.570, 0.580, 0.550],
    'away_team_era': [4.00, 3.80, 4.20, 3.90, 3.85, 4.10, 3.95, 4.05, 3.75, 4.15, 3.70, 4.00, 3.90, 4.05, 3.80],
    'away_team_batting_avg': [0.280, 0.290, 0.270, 0.285, 0.300, 0.275, 0.290, 0.280, 0.295, 0.270, 0.305, 0.285, 0.290, 0.280, 0.300],
    'outcome': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
}
df = pd.DataFrame(data)

# Prepare data
X = df[['home_team_win_rate', 'home_team_era', 'home_team_batting_avg',
        'away_team_win_rate', 'away_team_era', 'away_team_batting_avg']].values
y = df['outcome'].values

# Handle missing values
X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
y = np.nan_to_num(y, nan=0)

# Apply NIPALS PCA
n_components = 3
T, P, X_mean = nipals_pca(X, n_components=n_components)
X_pca = T  # Use scores as new features

# Normalize PCA features
X_pca_mean = X_pca.mean(axis=0)
X_pca_std = X_pca.std(axis=0)
X_pca_std[X_pca_std == 0] = 1  # Avoid division by zero
X_pca = (X_pca - X_pca_mean) / X_pca_std

# Convert to PyTorch tensors
X_pca = torch.tensor(X_pca, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(device)

# Define the model
class MLBModel(nn.Module):
    def __init__(self):
        super(MLBModel, self).__init__()
        self.fc1 = nn.Linear(n_components, 16)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(16, 8)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

model = MLBModel().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
try:
    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        outputs = model(X_pca)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/50, Loss: {loss.item():.4f}")
except Exception as e:
    print(f"Error training model: {e}")
    sys.exit(1)

# Save the model
try:
    torch.save(model.state_dict(), 'mlb_model.pth')
except Exception as e:
    print(f"Error saving model: {e}")
    sys.exit(1)

# Fetch today's games (July 19, 2025)
try:
    games = statsapi.schedule(date='2025-07-19', sportId=1)
except Exception as e:
    print(f"Error fetching games: {e}")
    sys.exit(1)

if not games:
    print("No games found for July 19, 2025. Try a regular-season date like 2025-04-01.")
    sys.exit(1)

# Predict outcomes for up to 15 games
print("\nMLB Game Predictions for July 19, 2025:")
print("-" * 50)
model.eval()
with torch.no_grad():
    for game in games[:15]:
        # Sample input (replace with real stats if available)
        new_game = np.array([[0.600, 3.50, 0.300, 0.550, 4.00, 0.280]])
        # Apply PCA transformation
        new_game = new_game - X_mean
        new_game_pca = new_game @ P[:, :n_components]
        new_game_pca = (new_game_pca - X_pca_mean) / X_pca_std
        new_game_pca = torch.tensor(new_game_pca, dtype=torch.float32).to(device)
        try:
            prediction = model(new_game_pca).item()
            home_team = game['home_name']
            away_team = game['away_name']
            result = 'Home Win' if prediction > 0.5 else 'Away Win'
            print(f"{home_team} vs {away_team}: {result} ({prediction:.2%} home win probability)")
        except Exception as e:
            print(f"Error predicting for {game['home_name']} vs {game['away_name']}: {e}")