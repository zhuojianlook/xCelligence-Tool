import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
import plotly

def load_layout(file_path):
    """Load layout data from the 'Layout' sheet."""
    return pd.read_excel(file_path, sheet_name='Layout')

def generate_well_options(layout_df):
    """Generate options for wells based on the layout."""
    wells = []
    for index, row in layout_df.iterrows():
        row_label = row.iloc[0]
        for col_index in range(1, len(layout_df.columns)):
            cell_value = row.iloc[col_index]
            if pd.notnull(cell_value):
                well_position = f"{row_label}{layout_df.columns[col_index]}"
                wells.append(well_position)
    return wells

def load_and_process_well_graph(file_path, wells_to_exclude):
    well_graph_df = pd.read_excel(file_path, sheet_name='Well Graph', skiprows=52)
    well_graph_df.columns = well_graph_df.iloc[0].map(str)  # Convert column names to strings
    well_graph_df = well_graph_df[1:].reset_index(drop=True)
    well_graph_df.columns = well_graph_df.columns.astype(str)
    first_column_name = well_graph_df.columns[0]
    well_graph_df[first_column_name] = pd.to_numeric(well_graph_df[first_column_name], errors='coerce')
    well_graph_df.set_index(first_column_name, inplace=True)
    well_graph_df.drop(columns=[str(well) for well in wells_to_exclude], errors='ignore', inplace=True)
    well_graph_df.columns = well_graph_df.columns.map(str)  # Ensure column names are strings after processing
    return well_graph_df

def map_sample_to_wells(layout_df):
    sample_to_wells = {}
    for _, row in layout_df.iterrows():
        for col_index, cell_value in enumerate(row.iloc[1:], start=1):
            if pd.notnull(cell_value):
                well_id = f"{row.iloc[0]}{layout_df.columns[col_index]}"
                sample_to_wells.setdefault(cell_value.strip(), []).append(well_id)
    return sample_to_wells

def normalize_data(well_graph_data, sample_to_wells, control_sample):
    control_wells = [well for sample_name, wells in sample_to_wells.items() if sample_name == control_sample for well in wells]
    control_wells = [f"{well}" for well in control_wells if f"{well}" in well_graph_data.columns]

    if not control_wells:
        st.error("No control wells found. Check if the correct control sample is selected.")
        return well_graph_data

    control_avg = well_graph_data[control_wells].mean(axis=1)
    normalized_data = well_graph_data.sub(control_avg, axis='index')
    normalized_data.fillna(0, inplace=True)

    return normalized_data

def perform_ratio_transformation(normalized_data, base_time):
    """Perform ratio transformation on the normalized data based on a specified base-time."""
    # Convert index to Series to use abs() method
    time_diff = pd.Series(normalized_data.index, index=normalized_data.index) - base_time
    closest_time_idx = time_diff.abs().argmin()  # Now operates on a Series, allowing use of abs()
    base_time_row = normalized_data.iloc[closest_time_idx]
    
    # Perform ratio transformation
    ratio_transformed_data = normalized_data.div(base_time_row)
    
    return ratio_transformed_data



def format_time_index(time_index):
    formatted_time = time_index.map(lambda x: f"{int(x // 3600):02d}:{int((x % 3600) / 60):02d}")
    return formatted_time


def plot_average_bar_graph_at_time(data, sample_to_wells, selected_time_in_seconds, title='Average Sample Measurement', yaxis_title='Average Measurement'):
    closest_time_idx = data.index.get_loc(selected_time_in_seconds, method='nearest')
    actual_time_point = data.index[closest_time_idx]

    averages = []
    std_devs = []
    samples = []

    for sample, wells in sample_to_wells.items():
        sample_columns = [f"{sample} - {well}" for well in wells if f"{sample} - {well}" in data.columns]
        if sample_columns:
            avg = data.loc[actual_time_point, sample_columns].mean()
            std_dev = data.loc[actual_time_point, sample_columns].std()

            samples.append(sample)
            averages.append(avg)
            std_devs.append(std_dev)

    # Create the bar graph with error bars for standard deviation
    fig = go.Figure(data=[
        go.Bar(
            x=samples,
            y=averages,
            error_y=dict(
                type='data',  # Use actual data values for error bars
                array=std_devs,  # Standard deviation values
                visible=True,  # Make sure error bars are visible
                color='gray',  # Customize error bars color here
                thickness=1.5,  # Customize error bars thickness here
            ),
            # Removing explicit color assignment to let Plotly manage colors automatically
            width=0.4  # Customize bar width here
        )
    ])

    hours, remainder = divmod(actual_time_point, 3600)
    minutes = remainder // 60
    fig.update_layout(
        title=f"{title} (Closest Time: {int(hours)}h {int(minutes)}m)",
        xaxis_title="Sample",
        yaxis_title=yaxis_title,
        hovermode="closest",
        margin=dict(l=40, r=0, t=40, b=30),
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)





def save_figure_as_tiff(fig, filename="plot.tiff"):
    pio.write_image(fig, filename, format='tiff')

def update_column_names(data, sample_to_wells):
    """Update column names of the DataFrame to reflect '{sample} - {well}' format."""
    new_column_names = {}
    for sample, wells in sample_to_wells.items():
        for well in wells:
            original_col_name = well  # Assuming the original column name is the well ID
            new_col_name = f"{sample} - {well}"
            if original_col_name in data.columns:
                new_column_names[original_col_name] = new_col_name
    # Rename columns based on the mapping created
    return data.rename(columns=new_column_names)

def plot_data(data, sample_to_wells, plot_average=False):
    fig = go.Figure()
    if plot_average:
        # Plot the average of replicates for each selected sample
        for sample, wells in sample_to_wells.items():
            sample_columns = [f"{sample} - {well}" for well in wells if f"{sample} - {well}" in data.columns]
            if sample_columns:
                # Calculate the mean of the sample columns
                mean_series = data[sample_columns].mean(axis=1)
                fig.add_trace(go.Scatter(
                    x=mean_series.index,
                    y=mean_series,
                    mode='lines+markers',
                    name=f"Average - {sample}"
                ))
    else:
        # Plot individual wells
        for sample, wells in sample_to_wells.items():
            for well in wells:
                column_name = f"{sample} - {well}"
                if column_name in data.columns:
                    fig.add_trace(go.Scatter(
                        x=data.index,
                        y=data[column_name],
                        mode='lines+markers',
                        name=column_name
                    ))

    # Update layout (keeping your layout settings here)
    fig.update_layout(
        title="Normalized and Ratio Transformed Cell Index Over Time",
        xaxis_title="Time (Hours)",
        yaxis_title="Normalized Cell Index",
        legend_title="Sample - Well ID",
        legend=dict(yanchor="middle", y=-0.2, xanchor="right", x=1),
        hovermode="closest",
        width=800, height=600,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    fig.update_xaxes(automargin=True)
    y_min = data.min().min()
    y_max = data.max().max()
    y_range = np.arange(np.floor(y_min), np.ceil(y_max) + 1, 1)  # Creating a range from min to max

    fig.update_yaxes(tickvals=y_range, ticktext=[str(int(val)) for val in y_range])


    st.plotly_chart(fig, use_container_width=False)


def main():
    st.title("xCelligence E-Plate 96 Data Analysis")
    st.markdown("""
        ## Instructions
        
        Instructions: (Make sure that the perimeter of the 96 well plate is not used)
        
        1. **Upload your Excel file** using the 'Upload your Excel file' button, make sure you export the file from with the layout in the matrix format, then save the file as an .xlsx (not xls).
        2. **Select wells to reject** from the multiselect dropdown, these can include wells that have obviously outlier readings or if you want to exclude replicates.
        3. **Choose the sample to normalize to** from the dropdown, e.g. triplicate of wells with no cells added.
        4. **Select the base-time for ratio transformation** follows the R package RTCA, select a base time where all well will have a cell index of 1.
        5. **Choose samples for plotting** and whether to plot averages of replicates.
        6. **View the generated plots** and download the results if needed.
        
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your Excel file", type=['xlsx'])

    if uploaded_file is not None:
        layout_df = load_layout(uploaded_file)
        st.write("Layout:")
        st.dataframe(layout_df)

        sample_to_wells = map_sample_to_wells(layout_df)
        unique_samples = list(sample_to_wells.keys())
        selected_wells_to_reject = st.multiselect("Select wells to reject:", options=generate_well_options(layout_df))
        control_sample = st.selectbox("Select the sample to normalize to:", unique_samples)
        
        well_graph_data = load_and_process_well_graph(uploaded_file, selected_wells_to_reject)

        if not well_graph_data.empty:
            normalized_data = normalize_data(well_graph_data, sample_to_wells, control_sample)
            formatted_time_index = format_time_index(well_graph_data.index)
            base_time_selection = st.selectbox("Select base-time for ratio transformation:", options=formatted_time_index.unique(), index=0)
            hours, minutes = map(int, base_time_selection.split(':'))
            selected_base_time_seconds = (hours * 3600) + (minutes * 60)
            ratio_transformed_data = perform_ratio_transformation(normalized_data, selected_base_time_seconds)
            ratio_transformed_data = update_column_names(ratio_transformed_data, sample_to_wells)
            
            st.write("Ratio Transformed Well Graph Data:")
            st.dataframe(ratio_transformed_data)
            samples_for_plotting = st.multiselect("Select samples for plotting:", options=unique_samples, default=unique_samples[:min(3, len(unique_samples))])
            plot_average = st.checkbox("Plot average of replicates instead of individual wells")

            if samples_for_plotting:
                data_to_plot_columns = [f"{sample} - {well}" for sample in samples_for_plotting for well in sample_to_wells[sample] if f"{sample} - {well}" in ratio_transformed_data.columns]
                data_to_plot = ratio_transformed_data[data_to_plot_columns]
                data_to_plot.index = data_to_plot.index / 3600
                plot_data(data_to_plot, {sample: sample_to_wells[sample] for sample in samples_for_plotting}, plot_average)

            # For the bar graph, use the same formatted time index for user selection
            time_point_selection_for_bar_graph = st.selectbox("Select Time Point for Bar Graph:", options=formatted_time_index.unique(), index=0)
            hours, minutes = map(int, time_point_selection_for_bar_graph.split(':'))
            selected_time_in_seconds_for_bar_graph = (hours * 3600) + (minutes * 60)

            if st.checkbox('Show Average Bar Graph at Selected Time Point'):
                plot_average_bar_graph_at_time(ratio_transformed_data, {sample: sample_to_wells[sample] for sample in samples_for_plotting}, selected_time_in_seconds_for_bar_graph)

if __name__ == "__main__":
    main()
