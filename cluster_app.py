import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

def main():
    st.title("Resource and Project Clustering App")
    st.markdown(
        """
        This app allows you to upload a CSV file (using a semicolon as delimiter and Latin-1 encoding)
        and specify the name of the column that contains the resource names (e.g. **Resources**).
        The app then:
        - Creates and displays a cluster map for the resources.
        - Computes hierarchical clusters for both resources (rows) and projects (columns).
        - Reorders the data accordingly.
        - Adds an additional row (for project clusters) and a new sequential index column.
        - Lets you preview and download the final CSV.
        """
    )

    # File uploader and text input for resource column name.
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    resource_col = st.text_input("Enter the name of the resource column", value="Resources")

    if uploaded_file is not None:
        try:
            # Read CSV using the expected encoding and separator.
            df = pd.read_csv(uploaded_file, encoding='latin-1', sep=';')
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return

        # Check that the resource column exists.
        if resource_col not in df.columns:
            st.error(f"The column '{resource_col}' was not found in the uploaded CSV file.")
            return

        # Set the resource column as the index and fill NaN with 0.
        df = df.set_index(resource_col)
        df = df.fillna(0)

        st.write("### Clustermap of Resources")
        try:
            # Create a clustermap using seaborn.
            # Increase figure size to better see the labels.
            g = sns.clustermap(df, method='ward', cmap="vlag", figsize=(15, 10), yticklabels=True)
            plt.setp(g.ax_heatmap.get_ymajorticklabels(), rotation=0, fontsize=6)
            g.ax_heatmap.set_title("Clustermap of Resources")
            st.pyplot(g.fig)
        except Exception as e:
            st.error(f"Error generating clustermap: {e}")
            return

        # -----------------------------
        # 1. Compute Resource Clusters
        # -----------------------------
        try:
            Z = linkage(df, method='ward')
            resource_cluster_labels = fcluster(Z, t=0.9, criterion='inconsistent')
            df['Cluster'] = resource_cluster_labels
        except Exception as e:
            st.error(f"Error computing resource clusters: {e}")
            return

        # -----------------------------
        # 2. Reorder DataFrame by Clustermap
        # -----------------------------
        try:
            # Use the clustermap dendrogram to reorder the rows.
            row_order = g.dendrogram_row.reordered_ind
            df_reordered = df.iloc[row_order, :]
        except Exception as e:
            st.error(f"Error reordering rows: {e}")
            return

        # -----------------------------
        # 3. Prepare the Final Resource DataFrame
        # -----------------------------
        try:
            df_final = df_reordered.reset_index()

            # Ensure that the 'Cluster' column appears right after the resource column.
            cols = list(df_final.columns)
            if 'Cluster' in cols:
                cols.remove('Cluster')
                # Insert the 'Cluster' column right after the first column (resource column).
                cols.insert(1, 'Cluster')
                df_final = df_final[cols]

            # Insert a new sequential index column at the beginning.
            df_final.insert(0, 'New_Index', range(1, len(df_final) + 1))
        except Exception as e:
            st.error(f"Error preparing final resource DataFrame: {e}")
            return

        # -----------------------------
        # 4. Cluster the Projects (Columns)
        # -----------------------------
        try:
            # Drop the resource cluster column to focus on project data.
            project_df = df_reordered.drop('Cluster', axis=1)
            # Transpose so that each project becomes a row.
            project_data = project_df.T
            Z_proj = linkage(project_data, method='ward')
            proj_cluster_labels = fcluster(Z_proj, t=0.8, criterion='inconsistent')
            project_data['Cluster_project'] = proj_cluster_labels
        except Exception as e:
            st.error(f"Error clustering projects: {e}")
            return

        # -----------------------------
        # 5. Reorder Projects Based on Clustering
        # -----------------------------
        try:
            dendro_proj = dendrogram(Z_proj, labels=project_data.index, no_plot=True)
            proj_order = dendro_proj['leaves']
            project_data_ordered = project_data.iloc[proj_order, :]
            # Get the ordered project column names.
            ordered_project_cols = list(project_data_ordered.index)
            # Create a one-row DataFrame of project cluster labels.
            project_cluster_row = project_data_ordered[['Cluster_project']].T
        except Exception as e:
            st.error(f"Error reordering projects: {e}")
            return

        # -----------------------------
        # 6. Reorder Final DataFrame Columns
        # -----------------------------
        try:
            # The first three columns in df_final are: New_Index, resource column, and Cluster.
            resource_cols = list(df_final.columns[:3])
            # Reorder so that project columns come in the clustered order.
            df_final = df_final[resource_cols + ordered_project_cols]
        except Exception as e:
            st.error(f"Error reordering final DataFrame columns: {e}")
            return

        # -----------------------------
        # 7. Add Additional Row for Project Cluster Labels
        # -----------------------------
        try:
            # Create an empty row with the same columns.
            empty_row = pd.DataFrame([[""] * len(df_final.columns)], columns=df_final.columns)
            # Fill in the project cluster labels.
            for col in ordered_project_cols:
                empty_row.at[0, col] = project_cluster_row.at['Cluster_project', col]
            # Mark the resource column in this row.
            empty_row.at[0, resource_cols[1]] = "Project_Cluster"
            # Concatenate the additional row on top.
            df_final_with_proj = pd.concat([empty_row, df_final], ignore_index=True)
        except Exception as e:
            st.error(f"Error adding project cluster row: {e}")
            return

        # -----------------------------
        # 8. Display and Allow Download of Final CSV
        # -----------------------------
        st.write("### Final DataFrame with Cluster Information")
        st.dataframe(df_final_with_proj)

        try:
            csv = df_final_with_proj.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Final CSV",
                data=csv,
                file_name="final_cluster.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Error preparing CSV download: {e}")
            return

if __name__ == '__main__':
    main()
