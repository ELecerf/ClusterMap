import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram

def main():
    st.title("Resource and Project Clustering App")
    st.markdown(
        """
        This app allows you to upload a CSV file and specify:
        - The name of the column containing the resource names (e.g. **Resources**).
        - The CSV delimiter (comma or semicolon).
        - Clustering thresholds for both resources (lines) and projects (columns).

        The app will:
        - Display the uploaded CSV file.
        - Create and display a clustermap for the resources.
        - Compute hierarchical clusters for both resources and projects.
        - Reorder the data accordingly.
        - Add an additional row (for project clusters) and a new sequential index column.
        - Let you preview and download the final CSV.
        """
    )

    # File uploader, resource column name input, and delimiter selector.
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    resource_col = st.text_input("Enter the name of the first column", value="Resources")
    delimiter = st.radio("Select CSV Delimiter", options=[",", ";"], index=1)

    if uploaded_file is not None:
        try:
            # Read CSV using the selected delimiter and Latin-1 encoding.
            df = pd.read_csv(uploaded_file, encoding='latin-1', sep=delimiter)
            st.write("### Uploaded CSV File")
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            return

        # Verify that the resource column exists.
        if resource_col not in df.columns:
            st.error(f"The column '{resource_col}' was not found in the CSV file.")
            return

        # Let the user choose clustering thresholds using sliders.
        line_threshold = st.slider("Line Clustering Threshold (for Resources)", 
                                   min_value=0.0, max_value=10.0, value=0.9, step=0.01)
        col_threshold = st.slider("Column Clustering Threshold (for Projects)", 
                                  min_value=0.0, max_value=10.0, value=0.8, step=0.01)

        # Set the resource column as index and fill missing values.
        df = df.set_index(resource_col)
        df = df.fillna(0)

        # -------------------------------
        # 1. Create and Display Clustermap (Resources)
        # -------------------------------
        st.write("### Clustermap of Resources")
        try:
            g = sns.clustermap(df, method='ward', cmap="vlag", figsize=(15, 10), yticklabels=True)
            plt.setp(g.ax_heatmap.get_ymajorticklabels(), rotation=0, fontsize=6)
            g.ax_heatmap.set_title("Clustermap of Resources")
            st.pyplot(g.fig)
        except Exception as e:
            st.error(f"Error generating clustermap: {e}")
            return

        # -------------------------------
        # 2. Compute Resource Cluster Labels
        # -------------------------------
        try:
            Z = linkage(df, method='ward')
            resource_cluster_labels = fcluster(Z, t=line_threshold, criterion='inconsistent')
            df['Cluster'] = resource_cluster_labels
        except Exception as e:
            st.error(f"Error computing resource clusters: {e}")
            return

        # -------------------------------
        # 3. Reorder the DataFrame Based on the Clustermap (Resources)
        # -------------------------------
        try:
            row_order = g.dendrogram_row.reordered_ind
            df_reordered = df.iloc[row_order, :]
        except Exception as e:
            st.error(f"Error reordering rows: {e}")
            return

        # -------------------------------
        # 4. Prepare the Final Resource DataFrame
        # -------------------------------
        try:
            df_final = df_reordered.reset_index()
            # Ensure 'Cluster' column appears right after the resource column.
            cols = list(df_final.columns)
            if 'Cluster' in cols:
                cols.remove('Cluster')
                cols.insert(1, 'Cluster')
                df_final = df_final[cols]
            # Insert a new sequential index column at the beginning.
            df_final.insert(0, 'New_Index', range(1, len(df_final) + 1))
        except Exception as e:
            st.error(f"Error preparing final resource DataFrame: {e}")
            return

        # -------------------------------
        # 5. Cluster the Projects and Assign Project Cluster Labels
        # -------------------------------
        try:
            # Remove the resource cluster column.
            project_df = df_reordered.drop('Cluster', axis=1)
            # Transpose so each project becomes a row.
            project_data = project_df.T
            Z_proj = linkage(project_data, method='ward')
            proj_cluster_labels = fcluster(Z_proj, t=col_threshold, criterion='inconsistent')
            project_data['Cluster_project'] = proj_cluster_labels
        except Exception as e:
            st.error(f"Error clustering projects: {e}")
            return

        # -------------------------------
        # 6. Reorder the Projects Based on Their Clustering
        # -------------------------------
        try:
            dendro_proj = dendrogram(Z_proj, labels=project_data.index, no_plot=True)
            proj_order = dendro_proj['leaves']
            project_data_ordered = project_data.iloc[proj_order, :]
            ordered_project_cols = list(project_data_ordered.index)
            # Create a one-row DataFrame of project cluster labels.
            project_cluster_row = project_data_ordered[['Cluster_project']].T
        except Exception as e:
            st.error(f"Error reordering projects: {e}")
            return

        # -------------------------------
        # 7. Reorder the Final DataFrame Columns
        # -------------------------------
        try:
            resource_cols = list(df_final.columns[:3])  # New_Index, resource name, and Cluster.
            df_final = df_final[resource_cols + ordered_project_cols]
        except Exception as e:
            st.error(f"Error reordering final DataFrame columns: {e}")
            return

        # -------------------------------
        # 8. Create the "Cluster_project" Additional Row
        # -------------------------------
        try:
            empty_row = pd.DataFrame([[""] * len(df_final.columns)], columns=df_final.columns)
            for col in ordered_project_cols:
                empty_row.at[0, col] = project_cluster_row.at['Cluster_project', col]
            # Label the resource column for this row.
            empty_row.at[0, resource_cols[1]] = "Project_Cluster"
            df_final_with_proj = pd.concat([empty_row, df_final], ignore_index=True)
        except Exception as e:
            st.error(f"Error adding project cluster row: {e}")
            return

        # -------------------------------
        # 9. Display and Allow Download of Final CSV
        # -------------------------------
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
