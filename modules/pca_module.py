import numpy as np
import pandas as pd
import streamlit as st

def run_pca():
    st.header("📊 Principal Component Analysis (PCA)")

    iterations = st.number_input("Enter the number of points (rows)", min_value=1, step=1, key="pca_rows")
    num_of_inputs = st.number_input("Enter the number of inputs per point (columns)", min_value=1, step=1, key="pca_cols")

    if iterations and num_of_inputs:
        X = []
        st.subheader("Input Data Matrix")
        for i in range(iterations):
            row = []
            cols = st.columns(num_of_inputs)
            for j in range(num_of_inputs):
                with cols[j]:
                    value = st.number_input(f"X[{i+1},{j+1}]", key=f"pca_{i}_{j}")
                    row.append(value)
            X.append(row)

        X = np.array(X)
        mean = np.mean(X, axis=0)
        X_centered = X - mean
        cov_matrix = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        pc1 = eigenvectors[:, 0]
        projected_data = X_centered @ pc1

        df_result = pd.DataFrame(X, columns=[f"X{i+1}" for i in range(X.shape[1])])
        st.subheader("Original Data")
        st.dataframe(df_result)

        st.subheader("Mean of each feature")
        st.write(mean)

        st.subheader("Covariance Matrix")
        st.write(cov_matrix)

        st.subheader("Eigenvalues")
        st.write(eigenvalues)

        st.subheader("Eigenvectors")
        st.write(eigenvectors)

        # Transformed matrix
        transformed = X_centered @ -eigenvectors
        df_transformed = pd.DataFrame(transformed, columns=[f"PC{i+1}" for i in range(transformed.shape[1])])
        st.subheader("Transformed Matrix")
        st.dataframe(df_transformed.round(5))
