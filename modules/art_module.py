import streamlit as st
import numpy as np

class ART1:
    def __init__(self, inputs, num_input, num_clusters, top_down_init, num_vectors, vig_pram, learning_rate):
        self.inputs = inputs
        self.num_input = num_input
        self.num_clusters = num_clusters
        self.top_down_init = top_down_init
        self.num_vectors = num_vectors
        self.vig_pram = vig_pram
        self.learning_rate = learning_rate

        self.bottom_up_weights_val = 1 / (1 + self.num_input)
        self.bottom_up_weights = np.ones((self.num_input, self.num_clusters)) * self.bottom_up_weights_val
        self.top_down_weights = np.ones((self.num_clusters, self.num_input)) * self.top_down_init

        self.train()

    def train(self):
        for i in range(self.num_vectors):
            f1_s = self.inputs[i]
            norm_s = np.sum(f1_s)
            f1_x = f1_s

            winner = self.find_winner(range(self.num_clusters), f1_x)
            f1_x = f1_s * self.top_down_weights[winner, :]
            norm_X = np.sum(f1_x)
            vigilance = norm_X / norm_s

            if vigilance > self.vig_pram:
                for i in range(len(f1_x)):
                    z = f1_x[i]
                    self.bottom_up_weights[i, winner] = (self.learning_rate * z) / (self.learning_rate - 1 + norm_X)
                    self.top_down_weights[winner, i] = (self.learning_rate * z) / (self.learning_rate - 1 + norm_X)
            else:
                remaining = list(range(self.num_clusters))
                remaining.remove(winner)
                self.find_winner(remaining, f1_x)

        st.subheader("Bottom-Up Weights:")
        st.dataframe(self.bottom_up_weights)

        st.subheader("Top-Down Weights:")
        st.dataframe(self.top_down_weights)

    def find_winner(self, clusters, f1_x):
        winner = None
        winner_val = -1
        for cluster in clusters:
            match_val = np.sum(f1_x * self.bottom_up_weights[:, cluster])
            if match_val > winner_val:
                winner_val = match_val
                winner = cluster
        return winner


def run_art():
    st.header("🧠 ART1 Network")

    num_input = st.number_input("Number of inputs per vector", min_value=1, value=4, step=1, key="art_inputs")
    num_vectors = st.number_input("Number of input vectors", min_value=1, value=4, step=1, key="art_vectors")

    if num_input and num_vectors:
        st.subheader("Input Vectors")
        full_input = []
        for i in range(num_vectors):
            cols = st.columns(num_input)
            row = []
            for j in range(num_input):
                with cols[j]:
                    val = st.number_input(f"X[{i + 1},{j + 1}]", key=f"art_{i}_{j}", min_value=0.0)
                    row.append(val)
            full_input.append(row)

        inputs = np.array(full_input)

        num_clusters = st.number_input("Number of clusters", min_value=1, value=3, step=1, key="art_clusters")
        top_down_value = st.number_input("Top-down initial weight value", min_value=0.1, value=2.0, step=0.1, key="art_topdown")
        vig_pram = st.number_input("Vigilance parameter", min_value=0.0, max_value=1.0, value=0.7, step=0.05, key="art_vig")
        learning_rate = st.number_input("Learning rate", min_value=0.1, value=2.0, step=0.1, key="art_lr")

        if st.button("Run ART1 Network"):
            st.subheader("Input Matrix:")
            st.dataframe(inputs)
            ART1(inputs, num_input, num_clusters, top_down_value, num_vectors, vig_pram, learning_rate)
