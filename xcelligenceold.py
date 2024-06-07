import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

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
    formatted_time = time_index.map(lambda x: f"{int(x // 3600)}:{int((x % 3600) / 60):02d}")
    return formatted_time

def to_scalar(val):
    if isinstance(val, pd.Series):
        if val.size == 1:
            return val.iloc[0]
        else:
            st.write(f"Series with size != 1: {val}")
            raise ValueError("Expected scalar value but got array of size != 1.")
    elif np.isscalar(val):
        return val
    else:
        st.write(f"Non-scalar value encountered: {val}")
        raise ValueError("Expected scalar value.")

def plot_average_bar_graph_at_time(data, sample_to_wells, selected_time_in_seconds, custom_sample_names=None, custom_colors=None, title='Time Point:', yaxis_title='Normalized Cell Index', significance_level=0.05):
    common_time_index = data.index  # Assuming data has been interpolated to a common time index
    abs_diff = np.abs(common_time_index - selected_time_in_seconds / 3600)  # Convert seconds to hours for matching
    closest_time_idx = abs_diff.argmin()
    actual_time_point = common_time_index[closest_time_idx]

    # Debug print statement
    st.write(f"Actual Time Point: {actual_time_point}")

    averages = []
    std_devs = []
    samples = []
    p_values = []
    comparisons = []  # Initialize comparisons list

    for sample, wells in sample_to_wells.items():
        sample_columns = [f"{sample} - {well}" for well in wells if f"{sample} - {well}" in data.columns]
        if sample_columns:
            avg = data.loc[actual_time_point, sample_columns].astype(float).mean()
            std_dev = data.loc[actual_time_point, sample_columns].astype(float).std()

            custom_label = custom_sample_names.get(sample, sample) if custom_sample_names else sample
            samples.append(custom_label)
            averages.append(avg)
            std_devs.append(std_dev)

    # Debug print statements
    st.write(f"Samples: {samples}")
    st.write(f"Averages: {averages}")
    st.write(f"Standard Deviations: {std_devs}")

    # Ensure averages and std_devs are scalar values
    averages = [to_scalar(avg) for avg in averages]
    std_devs = [to_scalar(std) for std in std_devs]

    # Debug print statements after conversion
    st.write(f"Converted Averages: {averages}")
    st.write(f"Converted Standard Deviations: {std_devs}")

    # Create the bar graph with error bars for standard deviation
    bar_colors = [custom_colors.get(sample, 'skyblue') for sample in samples] if custom_colors else 'skyblue'
    
    plt.figure(figsize=(11.69, 8.27))  # A4 size in landscape
    ax = plt.gca()

    plt.bar(samples, averages, yerr=std_devs, capsize=5, color=bar_colors, edgecolor='grey')
    
    # Add individual data points with random jitter
    jitter_strength = 0.1  # Adjust this value to control the amount of jitter
    for i, sample in enumerate(samples):
        original_sample = next(key for key, value in custom_sample_names.items() if value == sample) if custom_sample_names else sample
        sample_columns = [f"{original_sample} - {well}" for well in sample_to_wells[original_sample] if f"{original_sample} - {well}" in data.columns]
        if sample_columns:
            individual_data = data.loc[actual_time_point, sample_columns]
            jitter = np.random.uniform(-jitter_strength, jitter_strength, size=len(individual_data))
            plt.scatter(np.full_like(individual_data, i) + jitter, individual_data, color='black', edgecolor='black', marker='D', zorder=2)

    plt.title(f"{title} {int(actual_time_point)}h", fontsize=14, fontname='Arial')
    plt.xlabel("")
    plt.ylabel(yaxis_title, fontsize=12, fontname='Arial')
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(fontsize=10, fontname='Arial')
    plt.yticks(fontsize=10, fontname='Arial')
    plt.xticks(rotation=45, ha="right")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.tick_params(width=1)

    # Adjust figure height to make room for significance indicators
    if len(averages) > 0:
        max_y = max([avg + std for avg, std in zip(averages, std_devs) if not np.isnan(avg) and not np.isnan(std)])
        if not np.isnan(max_y) and not np.isinf(max_y):
            plt.ylim(0, max_y + max_y * 1.0)  # Adding 100% padding to the top for significance indicators
        else:
            plt.ylim(0, 1)  # Fallback in case there are no valid averages
    else:
        plt.ylim(0, 1)  # Fallback in case there are no valid averages

    # Create comparison checkboxes
    st.markdown("### Select Comparisons to Display Significance")
    comparison_checks = {}
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            sample1 = samples[i]
            sample2 = samples[j]
            comparison_label = f"{sample1} vs {sample2}"
            comparison_checks[comparison_label] = st.checkbox(comparison_label, value=True)
            comparisons.append((sample1, sample2, comparison_label))

    # Significance indicators
    significance_levels = [0.00005, 0.0005, 0.005, 0.05]
    significance_stars = ['****', '***', '**', '*']
    significance_offsets = [0.3, 0.25, 0.2, 0.15]  # Increased distance for the significance lines

    current_offset = 0
    for (sample1, sample2, comparison_label) in comparisons:
        if not comparison_checks[comparison_label]:
            continue
        i = samples.index(sample1)
        j = samples.index(sample2)
        original_sample1 = next(key for key, value in custom_sample_names.items() if value == sample1) if custom_sample_names else sample1
        original_sample2 = next(key for key, value in custom_sample_names.items() if value == sample2) if custom_sample_names else sample2
        sample1_values = data.loc[actual_time_point, [col for col in data.columns if original_sample1 in col]]
        sample2_values = data.loc[actual_time_point, [col for col in data.columns if original_sample2 in col]]
        _, p_value = ttest_ind(sample1_values, sample2_values)
        p_values.append({'Sample1': sample1, 'Sample2': sample2, 'p_value': round(p_value, 6)})

        for level, stars, offset in zip(significance_levels, significance_stars, significance_offsets):
            if p_value < level:
                y_max = max(averages[i] + std_devs[i], averages[j] + std_devs[j])
                line_y = y_max + (current_offset + offset) * max_y
                text_y = line_y + 0.02 * max_y
                ax.plot([i, j], [line_y, line_y], lw=1.5, color='black')
                ax.text((i + j) * .5, text_y, stars, ha='center', va='bottom', color='black', fontname='Arial')
                current_offset += 0.1  # Add more space for the next set of significance indicators
                break

    st.pyplot(plt)
    
    return plt, pd.DataFrame({'Sample': samples, 'Average': averages, 'Std Dev': std_devs}), pd.DataFrame(p_values)

def save_figure_as_tiff(fig, filename="plot.tiff"):
    fig.savefig(filename, format='tiff', dpi=300)

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

def plot_data_with_matplotlib(data, sample_to_wells, custom_sample_names=None, custom_colors=None, plot_average=False):
    plt.figure(figsize=(11.69, 8.27))  # A4 size in landscape
    ax = plt.gca()

    if plot_average:
        # Plot the average of replicates for each selected sample
        for sample, wells in sample_to_wells.items():
            sample_columns = [f"{sample} - {well}" for well in wells if f"{sample} - {well}" in data.columns]
            if sample_columns:
                mean_series = data[sample_columns].mean(axis=1)
                custom_label = custom_sample_names.get(sample, sample) if custom_sample_names else sample
                color = custom_colors.get(custom_label, None) if custom_colors else None
                plt.plot(mean_series.index, mean_series, marker='o', label=custom_label, color=color, linewidth=1.5)
    else:
        # Plot individual wells
        for sample, wells in sample_to_wells.items():
            for well in wells:
                column_name = f"{sample} - {well}"
                if column_name in data.columns:
                    custom_label = custom_sample_names.get(sample, column_name) if custom_sample_names else column_name
                    color = custom_colors.get(custom_label, None) if custom_colors else None
                    plt.plot(data.index, data[column_name], marker='o', label=custom_label, color=color, linewidth=1.5)

    # Nature publication guidelines for the plot
    plt.title("Normalized and Ratio Transformed Cell Index Over Time", fontsize=14, fontname='Arial')
    plt.xlabel("Time (Hours)", fontsize=12, fontname='Arial')
    plt.ylabel("Normalized Cell Index", fontsize=12, fontname='Arial')
    plt.legend(title="Sample - Well ID", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, frameon=False, prop={'family': 'Arial'})
    plt.grid(True)
    plt.tight_layout()
    plt.xticks(fontsize=10, fontname='Arial')
    plt.yticks(fontsize=10, fontname='Arial')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    ax.tick_params(width=1)
    
    st.pyplot(plt)
    
    return plt

def round_time_index(time_index):
    """Round the time index to the nearest hour."""
    rounded_index = pd.to_timedelta(time_index, unit='s').round('H')
    return rounded_index

def interpolate_data_to_common_time(data, common_time_index):
    """Interpolate data to a common time index."""
    data.index = pd.to_timedelta(data.index, unit='s')
    interpolated_data = data.reindex(common_time_index).interpolate(method='time')
    interpolated_data.index = interpolated_data.index.total_seconds() / 3600  # Convert back to hours
    return interpolated_data

def adjust_timepoints(data):
    """Adjust timepoints by subtracting the first non-zero timepoint."""
    time_index = data.index
    first_non_zero_time = time_index[time_index > 0].min()
    adjusted_time_index = time_index - first_non_zero_time
    data.index = adjusted_time_index
    return data

def main():
    st.title("xCelligence E-Plate 96 Data Analysis")
    st.markdown("""
        ## Instructions
        ...
    """, unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload your Excel file(s)", type=['xlsx'], accept_multiple_files=True)

    if uploaded_files:
        all_sample_to_wells = {}
        all_ratio_transformed_data = {}
        all_custom_sample_names = {}
        all_custom_colors = {}

        columns = st.columns(len(uploaded_files))

        for idx, uploaded_file in enumerate(uploaded_files):
            with columns[idx]:
                file_name = uploaded_file.name
                layout_df = load_layout(uploaded_file)
                st.write(f"Layout for {file_name}:")
                st.dataframe(layout_df)

                sample_to_wells = map_sample_to_wells(layout_df)
                unique_samples = list(sample_to_wells.keys())
                selected_wells_to_reject = st.multiselect(f"Select wells to reject for {file_name}:", options=generate_well_options(layout_df))
                control_sample = st.selectbox(f"Select the sample to normalize to for {file_name}:", unique_samples)
                
                well_graph_data = load_and_process_well_graph(uploaded_file, selected_wells_to_reject)
                well_graph_data = adjust_timepoints(well_graph_data)

                if not well_graph_data.empty:
                    normalized_data = normalize_data(well_graph_data, sample_to_wells, control_sample)
                    formatted_time_index = format_time_index(well_graph_data.index)
                    if '05:00' in formatted_time_index.values:
                        base_time_index = formatted_time_index.get_loc('05:00')
                    else:
                        base_time_index = 0  # Fallback to the first index if '05:00' is not found
                    base_time_selection = st.selectbox(f"Select base-time for ratio transformation for {file_name}:", options=formatted_time_index.unique(), index=base_time_index)
                    hours, minutes = map(int, base_time_selection.split(':'))
                    selected_base_time_seconds = (hours * 3600) + (minutes * 60)
                    ratio_transformed_data = perform_ratio_transformation(normalized_data, selected_base_time_seconds)
                    ratio_transformed_data = update_column_names(ratio_transformed_data, sample_to_wells)
                    
                    st.write(f"Ratio Transformed Well Graph Data for {file_name}:")
                    st.dataframe(ratio_transformed_data)
                    samples_for_plotting = st.multiselect(f"Select samples for plotting from {file_name}:", options=unique_samples, default=unique_samples[:min(3, len(unique_samples))])
                    plot_average = st.checkbox(f"Plot average of replicates instead of individual wells for {file_name}")

                    # Custom names and colors for samples
                    custom_sample_names = {}
                    custom_colors = {}
                    color_schemes = {'Inferno': 'inferno', 'Viridis': 'viridis'}
                    color_scheme = st.selectbox(f"Choose color scheme for {file_name}:", list(color_schemes.keys()))
                    color_palette = sns.color_palette(color_schemes[color_scheme], len(samples_for_plotting))
                    for i, sample in enumerate(samples_for_plotting):
                        custom_sample_names[sample] = st.text_input(f"Custom name for {sample} in {file_name}", value=sample)
                        custom_colors[sample] = st.color_picker(f"Pick a color for {sample} in {file_name}", value=f"#{''.join([format(int(x*255), '02x') for x in color_palette[i]])}")

                    all_sample_to_wells[file_name] = sample_to_wells
                    all_ratio_transformed_data[file_name] = ratio_transformed_data
                    all_custom_sample_names[file_name] = custom_sample_names
                    all_custom_colors[file_name] = custom_colors

                    if samples_for_plotting:
                        data_to_plot_columns = [f"{sample} - {well}" for sample in samples_for_plotting for well in sample_to_wells[sample] if f"{sample} - {well}" in ratio_transformed_data.columns]
                        data_to_plot = ratio_transformed_data[data_to_plot_columns]
                        data_to_plot.index = pd.to_timedelta(data_to_plot.index, unit='s') / pd.Timedelta(hours=1)
                        matplotlib_fig = plot_data_with_matplotlib(data_to_plot, {sample: sample_to_wells[sample] for sample in samples_for_plotting}, custom_sample_names, custom_colors, plot_average)

                        if st.button(f'Save Matplotlib Figure as TIFF for {file_name}'):
                            save_figure_as_tiff(matplotlib_fig, f"matplotlib_figure_{file_name}.tiff")
                            st.success(f"Matplotlib figure saved as matplotlib_figure_{file_name}.tiff")

                        st.write(f"Line Graph Data for {file_name}:")
                        st.dataframe(data_to_plot)

        # Combine data for bar graph
        selected_samples = {file_name: st.multiselect(f"Select samples from {file_name} to include in the combined plot:", options=list(all_sample_to_wells[file_name].keys()), default=list(all_sample_to_wells[file_name].keys())[:min(3, len(all_sample_to_wells[file_name].keys()))]) for file_name in [file.name for file in uploaded_files]}

        combined_custom_sample_names = {sample: all_custom_sample_names[file_name][sample] for file_name in selected_samples.keys() for sample in selected_samples[file_name]}
        combined_custom_colors = {all_custom_sample_names[file_name].get(sample, sample): all_custom_colors[file_name].get(sample, 'skyblue') for file_name in selected_samples.keys() for sample in selected_samples[file_name]}

        aggregated_data = pd.concat([all_ratio_transformed_data[file_name][[f"{sample} - {well}" for sample in selected_samples[file_name] for well in all_sample_to_wells[file_name][sample] if f"{sample} - {well}" in all_ratio_transformed_data[file_name].columns]] for file_name in selected_samples.keys()], axis=1)
        
        common_time_index = round_time_index(pd.concat([all_ratio_transformed_data[file_name].index.to_series() for file_name in all_ratio_transformed_data.keys()]).sort_values().unique())

        # Debug print statement
        st.write(f"Common Time Index: {common_time_index}")

        interpolated_data = {file_name: interpolate_data_to_common_time(all_ratio_transformed_data[file_name], common_time_index) for file_name in all_ratio_transformed_data.keys()}
        aggregated_interpolated_data = pd.concat([interpolated_data[file_name][[f"{sample} - {well}" for sample in selected_samples[file_name] for well in all_sample_to_wells[file_name][sample] if f"{sample} - {well}" in interpolated_data[file_name].columns]] for file_name in selected_samples.keys()], axis=1)
        
        time_point_options = [(t.total_seconds(), f"{int(t.total_seconds() // 3600)}:{int((t.total_seconds() % 3600) // 60):02d}") for t in pd.to_timedelta(common_time_index, unit='h')]
        time_point_selection_for_bar_graph = st.selectbox("Select Time Point for Combined Bar Graph:", options=[option[1] for option in time_point_options], index=0)
        selected_time_in_seconds_for_bar_graph = time_point_options[[option[1] for option in time_point_options].index(time_point_selection_for_bar_graph)][0]

        # Debug print statements
        st.write(f"Selected Time Point: {time_point_selection_for_bar_graph}")
        st.write(f"Selected Time in Seconds: {selected_time_in_seconds_for_bar_graph}")

        custom_bar_title = st.text_input("Custom Bar Graph Title", value="Time Point:")

        if st.checkbox('Show Combined Average Bar Graph at Selected Time Point'):
            matplotlib_bar_fig, bar_graph_data, plotly_p_values = plot_average_bar_graph_at_time(aggregated_interpolated_data, {sample: all_sample_to_wells[file_name][sample] for file_name in selected_samples.keys() for sample in selected_samples[file_name]}, selected_time_in_seconds_for_bar_graph, combined_custom_sample_names, combined_custom_colors, title=custom_bar_title)

            if st.button('Save Combined Bar Graph as TIFF'):
                save_figure_as_tiff(matplotlib_bar_fig, "combined_bar_graph.tiff")
                st.success("Combined bar graph saved as combined_bar_graph.tiff")

            st.write("Combined Bar Graph Data:")
            st.dataframe(bar_graph_data)

            st.write("Combined Significance Testing and P-Values:")
            st.dataframe(plotly_p_values)

if __name__ == "__main__":
    main()
