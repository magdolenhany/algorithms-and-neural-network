import streamlit as st
import numpy as np

def run_som():
    st.header("🔵 Self-Organizing Map (SOM)")

    iterations = st.number_input("Number of iterations", min_value=1, step=1, key="som_iter")
    num_input_cols = st.number_input("Number of input columns", min_value=1, step=1, key="som_input_cols")

    inputs = []
    for i in range(iterations):
        row = []
        cols = st.columns(num_input_cols)
        for j in range(num_input_cols):
            with cols[j]:
                row.append(st.number_input(f"Input ({i+1},{j+1})", key=f"som_input_{i}_{j}"))
        inputs.append(row)

    num_weight_cols = st.number_input("Number of weight columns", min_value=1, step=1, key="som_weight_cols")
    weights = []
    for i in range(num_input_cols):
        row = []
        cols = st.columns(num_weight_cols)
        for j in range(num_weight_cols):
            with cols[j]:
                row.append(st.number_input(f"Weight ({i+1},{j+1})", key=f"som_weight_{i}_{j}"))
        weights.append(row)

    learning_rate = st.number_input("Learning Rate", min_value=0.01, step=0.01, value=0.5, key="som_lr")

    if st.button("Run SOM"):
        inputs = np.array(inputs)
        weights = np.array(weights)
        st.text(f"The Inputs Values:\n{inputs}")
        st.text(f"The Weights Values:\n{weights}")

        for iteration in range(iterations):
            st.subheader(f"Iteration {iteration + 1}")
            input_vector = inputs[iteration]
            st.text(f"Input Vector: {input_vector}")

            distances = np.sum((weights - input_vector.reshape(-1, 1)) ** 2, axis=0)
            winner = np.argmin(distances)
            st.text(f"Winner Cluster: {winner + 1}")
            weights[:, winner] += learning_rate * (input_vector - weights[:, winner])
            st.text(f"Updated Weights for Cluster {winner + 1}:\n{weights[:, winner]}")
            learning_rate /= 2

        st.success(f"Final Weights:\n{weights}")
