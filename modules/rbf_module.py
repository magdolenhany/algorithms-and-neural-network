import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

def rbf(x, c, sigma):
    r_squared = np.sum((x - c) ** 2, axis=1)
    return np.exp(-r_squared / (2 * sigma)), r_squared

def run_rbf():
    st.header("🔴 Radial Basis Function (RBF) Classification")

    inputs_len = st.number_input("Number of input points", min_value=1, step=1, key="rbf_len")
    points = []

    for i in range(inputs_len):
        cols = st.columns(2)
        row = []
        for j in range(2):
            with cols[j]:
                val = st.number_input(f"Input ({i+1},{j+1})", key=f"rbf_{i}_{j}")
                row.append(val)
        points.append(row)

    sigma = st.number_input("Sigma (spread)", value=1.0, step=0.1, key="rbf_sigma")

    c1 = [st.number_input("C1[0]", value=0.0, key="c1_0"),
          st.number_input("C1[1]", value=0.0, key="c1_1")]
    c2 = [st.number_input("C2[0]", value=2.0, key="c2_0"),
          st.number_input("C2[1]", value=2.0, key="c2_1")]

    if st.button("Run RBF"):
        points = np.array(points)
        C1 = np.array(c1)
        C2 = np.array(c2)

        phi1, r1_squared = rbf(points, C1, sigma)
        phi2, r2_squared = rbf(points, C2, sigma)
        labels = np.where(phi1 > phi2, 0, 1)

        df = pd.DataFrame(points, columns=['x1', 'x2'])
        df['r1^2'] = r1_squared
        df['r2^2'] = r2_squared
        df['Phi1'] = phi1
        df['Phi2'] = phi2
        df['Label'] = labels

        st.dataframe(df)

        fig = plt.figure(figsize=(7, 7))
        for i, (x, y) in enumerate(points):
            plt.scatter(x, y, color='red' if labels[i] == 0 else 'blue', s=100)
        plt.scatter(*C1, color='black', marker='x', s=200, label="C1")
        plt.scatter(*C2, color='black', marker='x', s=200, label="C2")

        xx, yy = np.meshgrid(np.linspace(-1, 4, 100), np.linspace(-1, 4, 100))
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        Z = np.where(rbf(grid_points, C1, sigma)[0] > rbf(grid_points, C2, sigma)[0], 0, 1)
        Z = Z.reshape(xx.shape)

        plt.contour(xx, yy, Z, levels=[0.5], colors='green', linewidths=2)
        plt.legend()
        plt.title("RBF Classification")
        plt.xlabel("X1")
        plt.ylabel("X2")
        st.pyplot(fig)
