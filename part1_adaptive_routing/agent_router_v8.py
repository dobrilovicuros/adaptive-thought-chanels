"""
Adaptive Neural Network — Routing Agent (Version 8)
===================================================
A Q-learning routing agent that learns to activate specific thought channels 
based on the context of the session (short-term memory of inputs and answers).

State (16 values):
  - Memory: Last 4 (input, answer) pairs -> 12 values
  - Channel stats: Accuracy of K0-K3 in the current session -> 4 values

Action: Select channel {0, 1, 2, 3}
Reward: +1 correct answer, -1 incorrect answer
"""

import numpy as np

class RoutingAgent:
    def __init__(self, n_channels=4, memory_len=4, lr=0.01, gamma=0.9, 
                 epsilon=0.3, epsilon_decay=0.9995, hidden=32):
        self.n_channels = n_channels
        self.memory_len = memory_len
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.state_dim = memory_len * 3 + n_channels

        # Q-Network: state_dim -> hidden -> n_channels
        self.W1 = np.random.randn(self.state_dim, hidden) * np.sqrt(2.0 / self.state_dim)
        self.b1 = np.zeros((1, hidden))
        self.W2 = np.random.randn(hidden, n_channels) * np.sqrt(2.0 / hidden)
        self.b2 = np.zeros((1, n_channels))

        self.mem_buffer = np.zeros((memory_len, 3))
        self.ch_correct = np.zeros(n_channels)
        self.ch_total = np.zeros(n_channels)

        self.replay =[]
        self.replay_size = 500
        self.batch_size = 32

    def reset_session(self):
        """Reset session memory when a new task starts."""
        self.mem_buffer = np.zeros((self.memory_len, 3))
        self.ch_correct = np.zeros(self.n_channels)
        self.ch_total = np.zeros(self.n_channels)

    def get_state(self) -> np.ndarray:
        """Builds state vector from memory and channel statistics."""
        mem_flat = self.mem_buffer.flatten()
        safe_total = np.where(self.ch_total > 0, self.ch_total, 1.0)
        ch_acc = np.where(self.ch_total > 0, self.ch_correct / safe_total, 0.5)
        return np.concatenate([mem_flat, ch_acc]).reshape(1, -1)

    def forward_q(self, state: np.ndarray):
        """Calculates Q-values for all channels."""
        h = np.maximum(0, state @ self.W1 + self.b1)
        q = h @ self.W2 + self.b2
        return q, h

    def select_action(self, state: np.ndarray, greedy=False) -> int:
        """Epsilon-greedy channel selection."""
        if not greedy and np.random.rand() < self.epsilon:
            return np.random.randint(self.n_channels)
        q, _ = self.forward_q(state)
        return int(np.argmax(q[0]))

    def update_memory(self, x: np.ndarray, y_true: float):
        """Updates short-term memory with a new (input, answer) pair."""
        new_pair = np.array([x[0], x[1], float(y_true)])
        self.mem_buffer = np.roll(self.mem_buffer, -1, axis=0)
        self.mem_buffer[-1] = new_pair

    def update_stats(self, channel: int, correct: bool):
        """Updates accuracy statistics per channel."""
        self.ch_total[channel] += 1
        self.ch_correct[channel] += float(correct)

    def store_transition(self, state, action, reward, next_state):
        self.replay.append((state.copy(), action, reward, next_state.copy()))
        if len(self.replay) > self.replay_size:
            self.replay.pop(0)

    def train_step(self) -> float:
        """Mini-batch Q-learning update."""
        if len(self.replay) < self.batch_size: return 0.0

        indices = np.random.choice(len(self.replay), self.batch_size, replace=False)
        batch = [self.replay[i] for i in indices]

        total_loss = 0.0
        for (s, a, r, s_next) in batch:
            q_curr, h = self.forward_q(s)
            q_next, _ = self.forward_q(s_next)

            target = r + self.gamma * np.max(q_next[0])
            td_error = target - q_curr[0, a]

            dq = np.zeros((1, self.n_channels))
            dq[0, a] = -td_error
            dW2 = h.T @ dq
            db2 = dq.copy()

            dh_pre = (dq @ self.W2.T) * (h > 0)
            dW1 = s.T @ dh_pre
            db1 = dh_pre.copy()

            # Update weights
            self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1
            self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
            total_loss += td_error ** 2

        return total_loss / self.batch_size

    def decay_epsilon(self):
        self.epsilon = max(0.05, self.epsilon * self.epsilon_decay)

# Implementation demo of Q-learning logic decoupled from the massive ChannelNet
if __name__ == "__main__":
    print("Routing Agent v8 Module loaded successfully.")
    print("Can be attached to any multi-output ChannelNet via select_action().")
