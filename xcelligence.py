import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
from functools import reduce
import numpy as np
import uuid 
from matplotlib import rcParams
import scipy.stats as stats
from scipy.stats import f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd

st.set_page_config(layout="wide")

def load_data(file_path):
    """Load data from the 'Layout' and 'Cell Index' sheets."""
    layout_df = pd.read_excel(file_path, sheet_name='Layout')
    cell_index_df = pd.read_excel(file_path, sheet_name='Cell Index')
    return layout_df, cell_index_df

def create_well_sample_mapping(layout_df):
    """Create a mapping from well positions to sample names."""
    well_sample_mapping = {}
    for _, row in layout_df.iterrows():
        row_label = row.iloc[0]
        for col_index, cell_value in enumerate(row.iloc[1:], start=1):
            well_position = f"{row_label}{layout_df.columns[col_index]}"
            well_sample_mapping[well_position] = cell_value.strip() if pd.notnull(cell_value) else None
    return well_sample_mapping

def process_blocks(cell_index_df, well_sample_mapping):
    """Process all blocks of data in the Cell Index sheet, ignoring the first two blocks."""
    processed_blocks = []
    block_start_indices = cell_index_df[cell_index_df.iloc[:, 0].str.startswith('Cell Index at: ', na=False)].index[2:]

    for row_index in block_start_indices:
        time_point = cell_index_df.iloc[row_index, 0].replace('Cell Index at: ', '')
        time_point_timedelta = pd.to_timedelta(time_point)
        data_block = cell_index_df.iloc[row_index + 2:row_index + 10, 1:13]
        data_block.columns = range(1, 13)
        data_block.index = list('ABCDEFGH')
        data_block = data_block.stack().reset_index()
        data_block.columns = ['Row', 'Column', 'Value']
        data_block['Well'] = data_block['Row'] + data_block['Column'].astype(str)
        data_block['Sample'] = data_block['Well'].map(well_sample_mapping)
        data_block = data_block.dropna(subset=['Sample'])
        data_block['Value'] = pd.to_numeric(data_block['Value'], errors='coerce')
        data_block['Time'] = time_point_timedelta.total_seconds()
        data_block['Sample_Well'] = data_block['Sample'] + " - " + data_block['Well']
        processed_blocks.append(data_block)

    if processed_blocks:
        combined_data = pd.concat(processed_blocks).sort_values(by='Time')
        combined_data['Time_str'] = combined_data['Time'].apply(lambda x: f'{int(x // 3600)}h {int((x % 3600) // 60)}m')
        combined_data_pivot = combined_data.pivot(index='Time', columns='Sample_Well', values='Value')
        combined_data_pivot['Time_str'] = combined_data_pivot.index.map(lambda x: f'{int(x // 3600)}h {int((x % 3600) // 60)}m')
        sorted_columns = sorted([col for col in combined_data_pivot.columns if col != 'Time_str'], key=lambda x: (x.split(" - ")[0], x.split(" - ")[1]))
        combined_data_pivot = combined_data_pivot[sorted_columns + ['Time_str']]
        return combined_data_pivot, combined_data
    else:
        return pd.DataFrame(), pd.DataFrame()

def normalize_to_sample(processed_data_pivot, sample_name):
    """Normalize data to the selected sample."""
    sample_columns = [col for col in processed_data_pivot.columns if sample_name in col]
    sample_averages = processed_data_pivot[sample_columns].mean(axis=1)
    normalized_data = processed_data_pivot.apply(lambda x: x - sample_averages, axis=0)
    return normalized_data

def perform_ratio_transformation(normalized_data, base_time_seconds):
    """Perform ratio transformation based on a specified base-time in seconds."""
    time_diff = normalized_data.index.total_seconds() - base_time_seconds
    closest_time_idx = abs(time_diff).argmin()
    base_time_row = normalized_data.iloc[closest_time_idx]
    normalized_data = normalized_data.apply(pd.to_numeric, errors='coerce')
    ratio_transformed_data = normalized_data.div(base_time_row)
    return ratio_transformed_data, closest_time_idx

def round_to_nearest_quarter_hour(td):
    """Round a timedelta to the nearest 15 minutes."""
    total_minutes = (td.total_seconds() // 60)
    remainder = total_minutes % 15
    total_minutes = total_minutes - remainder if remainder < 7.5 else total_minutes + (15 - remainder)
    return pd.to_timedelta(total_minutes, unit='m')

def adjust_time_index(ratio_transformed_data):
    """Adjust the time index by deducting the first row time value from all other time row values."""
    first_time_value = pd.to_timedelta(ratio_transformed_data.index[0], unit='s')
    adjusted_index = pd.to_timedelta(ratio_transformed_data.index, unit='s') - first_time_value
    adjusted_index = adjusted_index.map(round_to_nearest_quarter_hour)
    ratio_transformed_data.index = adjusted_index
    ratio_transformed_data['Adjusted_Time_str'] = adjusted_index.map(lambda x: f'{x.components.days * 24 + x.components.hours}h {x.components.minutes}m')
    return ratio_transformed_data

def convert_time_str_to_hours(time_str):
    """Convert time string 'Hh Mm' to total hours as float."""
    parts = time_str.split(' ')
    hours = int(parts[0].replace('h', ''))
    minutes = int(parts[1].replace('m', '')) if len(parts) > 1 else 0
    return hours + minutes / 60.0

def sort_time_strings(time_strings):
    """Sort the time strings in the format 'Hh Mm'."""
    def parse_time_string(ts):
        parts = ts.split(' ')
        hours = int(parts[0].replace('h', ''))
        minutes = int(parts[1].replace('m', '')) if len(parts) > 1 else 0
        return hours * 60 + minutes  # Convert to total minutes for sorting
    
    return sorted(time_strings, key=parse_time_string)

def get_significance_stars(p_value):
    """Return the significance stars based on p-value."""
    if p_value < 0.00005:
        return '****'
    elif p_value < 0.0005:
        return '***'
    elif p_value < 0.005:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'

def plot_combined_data(combined_data, selected_samples, plot_average, show_std_dev, fig_title, x_label, y_label, palette, base_times, base_time_labels, dashes, show_grid, show_base_lines, custom_names, custom_colors, point_size, point_style, enhance_visibility, fig_width, fig_height):
    sns.set(font_scale=1.2)
    rcParams['font.family'] = 'arial'
    if show_grid:
        sns.set_style("whitegrid")
    else:
        sns.set_style("white")

    # Set up the figure with dynamic size
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # Determine the color palette
    if not isinstance(palette, list):
        sns.set_palette(palette)
        palette = sns.color_palette(palette, n_colors=len(selected_samples))

    # Convert time string to hours
    time_hours = combined_data.index.map(convert_time_str_to_hours)

    # Iterate over each selected sample for plotting
    for idx, sample in enumerate(selected_samples):
        label = custom_names.get(sample, sample)
        color = custom_colors.get(sample, palette[idx % len(palette)])
        
        # Define marker style
        edgecolor = 'black' if enhance_visibility else color
        linewidth = 1.5 if enhance_visibility else 1.0
        marker_style = {'marker': point_style, 'markersize': point_size, 'color': color, 'linestyle': '-', 'linewidth': linewidth, 'markeredgecolor': edgecolor}

        # Plot either averaged data or individual data points
        if plot_average:
            mean_series = combined_data[f"{sample}_mean"]
            if show_std_dev:
                std_series = combined_data[f"{sample}_std"]
                ax.errorbar(time_hours, mean_series, yerr=std_series, fmt=point_style, label=f'{label} ± SD', capsize=5, **marker_style)
            else:
                ax.plot(time_hours, mean_series, label=label, **marker_style)
        else:
            sample_columns = [col for col in combined_data.columns if sample in col and '_mean' not in col and '_std' not in col]
            for col in sample_columns:
                well_id = col.split(' - ')[1]
                specific_label = f"{label} ({well_id})"
                sns.lineplot(x=time_hours, y=combined_data[col], label=specific_label, **marker_style)

    # Plot base time lines if enabled
    if show_base_lines:
        for i, base_time_sec in enumerate(base_times):
            ax.axvline(x=base_time_sec / 3600, color='red', linestyle=dashes[i % len(dashes)], linewidth=1, label=f'Base Time: {base_time_labels[i]}')

    # Set the axis labels, title, and legend
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(fig_title)
    ax.set_xticks(np.arange(0, max(time_hours) + 1, 6))
    ax.set_xticklabels([str(int(t)) for t in np.arange(0, max(time_hours) + 1, 6)])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)

    # Display the plot in the Streamlit app
    st.pyplot(fig)

    # Save the plot to a buffer for download
    buf = io.BytesIO()
    fig.savefig(buf, format='tiff')
    buf.seek(0)
    download_key = f"download_{uuid.uuid4()}"
    st.download_button(label="Download TIFF", data=buf, file_name="combined_line_graph.tiff", mime="image/tiff", key=download_key)

def process_and_adjust(file, base_time_selection):
    """Process and adjust time index for each uploaded file."""
    layout_df, cell_index_df = load_data(file)
    well_sample_mapping = create_well_sample_mapping(layout_df)
    processed_data_pivot, combined_data = process_blocks(cell_index_df, well_sample_mapping)

    if not processed_data_pivot.empty:
        sample_names = sorted(set(col.split(" - ")[0] for col in processed_data_pivot.columns if col != 'Time_str'))
        selected_sample = sample_names[0]  # For demonstration, we'll select the first sample for normalization

        normalized_data = normalize_to_sample(processed_data_pivot.drop(columns='Time_str'), selected_sample)
        adjusted_time_data = adjust_time_index(normalized_data)

        selected_base_time_seconds = pd.to_timedelta(base_time_selection).total_seconds()

        ratio_transformed_data, base_time_idx = perform_ratio_transformation(normalized_data, selected_base_time_seconds)
        adjusted_ratio_data = adjust_time_index(ratio_transformed_data)

        base_time_str = adjusted_ratio_data['Adjusted_Time_str'][base_time_idx]

        return adjusted_ratio_data, selected_base_time_seconds, base_time_str
    else:
        return pd.DataFrame(), None, None

def concatenate_dataframes(dataframes):
    """Concatenate dataframes horizontally using 'Adjusted_Time_str' as the index."""
    for df in dataframes:
        if 'Adjusted_Time_str' not in df.columns:
            df['Adjusted_Time_str'] = df.index.map(lambda x: f'{int(x.total_seconds() // 3600)}h {int((x % 3600) // 60)}m')
        df.set_index('Adjusted_Time_str', inplace=True)

    combined_df = pd.concat(dataframes, axis=1)
    # Calculate mean values for each sample
    sample_names = sorted(set([col.split(" - ")[0] for col in combined_df.columns if " - " in col]))
    for sample in sample_names:
        sample_columns = [col for col in combined_df.columns if sample in col]
        combined_df[f'{sample}_mean'] = combined_df[sample_columns].mean(axis=1)
        combined_df[f'{sample}_std'] = combined_df[sample_columns].std(axis=1)

    return combined_df

def process_and_display(file, col):
    """Process and display data for each uploaded file."""
    layout_df, cell_index_df = load_data(file)
    col.write("Layout Data:")
    col.dataframe(layout_df)
    col.write("Cell Index Data:")
    col.dataframe(cell_index_df.head(33))

    well_sample_mapping = create_well_sample_mapping(layout_df)
    processed_data_pivot, combined_data = process_blocks(cell_index_df, well_sample_mapping)

    if not processed_data_pivot.empty:
        columns_to_drop = col.multiselect("Select columns to drop:", options=processed_data_pivot.columns.tolist(), key=file.name)
        if columns_to_drop:
            processed_data_pivot = processed_data_pivot.drop(columns=columns_to_drop)

        col.write("Processed Data:")
        col.dataframe(processed_data_pivot)

        sample_names = sorted(set(col.split(" - ")[0] for col in processed_data_pivot.columns if col != 'Time_str'))
        selected_sample = col.selectbox("Select sample for normalization:", options=sample_names, key=file.name + '_sample')

        if selected_sample:
            normalized_data = normalize_to_sample(processed_data_pivot.drop(columns='Time_str'), selected_sample)
            col.write("Normalized Data:")
            normalized_data_display = normalized_data.copy()
            normalized_data_display.index = pd.to_timedelta(normalized_data_display.index, unit='s').map(lambda x: f'{int(x.total_seconds() // 3600)}h {int((x.total_seconds() % 3600) // 60)}m')
            col.dataframe(normalized_data_display)

            adjusted_time_data = adjust_time_index(normalized_data)
            time_options = sort_time_strings(adjusted_time_data['Adjusted_Time_str'].unique().tolist())
            base_time_selection = col.selectbox("Select base-time for ratio transformation:", options=time_options, key=file.name + '_base_time', index=0)

            ratio_transformed_data, base_time_idx = perform_ratio_transformation(normalized_data, pd.to_timedelta(base_time_selection).total_seconds())
            ratio_transformed_data_display = ratio_transformed_data.copy()
            ratio_transformed_data_display.index = pd.to_timedelta(ratio_transformed_data_display.index, unit='s').map(lambda x: f'{int(x.total_seconds() // 3600)}h {int((x.total_seconds() % 3600) // 60)}m')
            col.write("Ratio Transformed Data:")
            col.dataframe(ratio_transformed_data_display)

            adjusted_time_data = adjust_time_index(ratio_transformed_data)
            adjusted_time_str = adjusted_time_data['Adjusted_Time_str'].reset_index(drop=True)
            adjusted_time_data = adjusted_time_data.reset_index(drop=True)
            adjusted_time_data['Adjusted_Time_str'] = adjusted_time_str

            col.write("Adjusted Time Data:")
            col.dataframe(adjusted_time_data.drop(columns='Adjusted_Time_str').set_index(adjusted_time_str))

            base_time_seconds = pd.to_timedelta(base_time_selection).total_seconds()

            return processed_data_pivot, adjusted_time_data, base_time_seconds, base_time_selection

    return processed_data_pivot, pd.DataFrame(), None, None

def align_data_frames(all_data):
    """Align and merge all data frames based on 'Adjusted_Time_str'."""
    for data_pivot in all_data:
        if 'Adjusted_Time_str' not in data_pivot.columns:
            data_pivot['Adjusted_Time_str'] = data_pivot.index.map(lambda x: f'{int(x // 3600)}h {int((x % 3600) // 60)}m')
        data_pivot.set_index('Adjusted_Time_str', inplace=True)

    combined_df = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), all_data)
    return combined_df

def print_sample_replicate_data(data, time_point, selected_samples):
    """Fetch replicate data at selected time point for given samples and calculate extended statistics."""
    if time_point not in data.index:
        st.write("Time point not available in the data.")
        return None

    # Filter columns based on selected samples and exclude mean and std columns
    relevant_columns = [col for col in data.columns if any(sample in col for sample in selected_samples) and '_mean' not in col and '_std' not in col]
    specific_time_data = data.loc[time_point, relevant_columns]

    # Creating a DataFrame to display the values along with sample identification
    sample_data_list = []
    for sample in selected_samples:
        for col in relevant_columns:
            if sample in col:
                sample_data_list.append({
                    'Sample': sample,
                    'Well': col.split('-')[-1].strip(),
                    'Value': data.at[time_point, col]
                })

    sample_data_df = pd.DataFrame(sample_data_list)

    # Calculate mean and SD for each well/sample
    summary_stats = sample_data_df.groupby('Sample').agg({'Value': ['mean', 'std', 'count']}).reset_index()
    summary_stats.columns = ['Sample', 'Mean', 'SD', 'N']

    # Perform statistical analysis
    sample_groups = [sample_data_df[sample_data_df['Sample'] == sample]['Value'].dropna() for sample in selected_samples]
    if len(sample_groups) > 1:
        f_val, p_val = stats.f_oneway(*sample_groups)
        if p_val < 0.05:
            data_flat = np.concatenate(sample_groups)
            labels = np.concatenate([[sample]*len(group) for sample, group in zip(selected_samples, sample_groups)])
            tukey_result = pairwise_tukeyhsd(data_flat, labels, 0.05)
            tukey_table = tukey_result.summary().data[1:]  # Skipping the header row
            tukey_df = pd.DataFrame(tukey_table, columns=tukey_result.summary().data[0])
            p_values = dict(zip(tukey_df['group1'] + ' vs ' + tukey_df['group2'], tukey_df['p-adj'].astype(float)))
            summary_stats['P-Value'] = summary_stats['Sample'].apply(lambda x: ', '.join([f"{k}: {v:.3f}" for k, v in p_values.items() if x in k]))
        else:
            summary_stats['P-Value'] = 'n.s.'  # Not significant
    else:
        summary_stats['P-Value'] = 'n.a.'  # Not applicable

    st.write("Detailed Replicate Data:")
    st.dataframe(sample_data_df)

    st.write("Summary Statistics and Statistical Analysis:")
    st.dataframe(summary_stats)

    return sample_data_df, summary_stats

def plot_bar_chart(sample_data_df, summary_stats, fig_title, x_label, y_label, custom_colors, custom_names, fig_width, fig_height, show_grid):
    """Plot a bar chart of the sample data at a specific time point."""
    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 12
    
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    if show_grid:
        ax.grid(True, which='both', linestyle='--', linewidth=0.7)
    else:
        ax.grid(False)
    
    sample_means = summary_stats.set_index('Sample')['Mean']
    sample_errors = summary_stats.set_index('Sample')['SD']
    sample_names = [custom_names.get(sample, sample) for sample in sample_means.index]
    bars = sample_means.plot(kind='bar', yerr=sample_errors, ax=ax, capsize=5, color=[custom_colors[sample] for sample in sample_means.index], edgecolor='black', alpha=0.75)
    
    ax.set_title(fig_title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticklabels(sample_names, rotation=45, ha='right')

    # Overlay a scatter plot with jitter for replicates
    for i, sample in enumerate(sample_means.index):
        jittered_x = np.random.normal(i, 0.05, size=len(sample_data_df[sample_data_df['Sample'] == sample]))
        ax.scatter(jittered_x, sample_data_df[sample_data_df['Sample'] == sample]['Value'], color='black', zorder=10)

    # Add significance lines if Tukey test results are significant
    indicator_positions = []  # To track the positions of previous indicators
    added_comparisons = set()  # To track added comparisons
    max_y = (sample_means + sample_errors).max()  # Get the maximum y value for positioning

    for _, row in summary_stats.iterrows():
        p_values_str = row['P-Value']
        if p_values_str != 'n.s.' and p_values_str != 'n.a.':
            comparisons = p_values_str.split(', ')
            for comp in comparisons:
                groups, p_value = comp.split(': ')
                p_value = float(p_value)
                group1, group2 = groups.split(' vs ')
                if p_value < 0.05:
                    sorted_groups = tuple(sorted([group1, group2]))
                    if sorted_groups in added_comparisons:
                        continue  # Skip if the comparison has already been added
                    x1 = list(sample_means.index).index(group1)
                    x2 = list(sample_means.index).index(group2)
                    y = max_y + 0.1 * max_y  # Start above the highest bar
                    h = 0.1 * max_y  # Increase the height to avoid overlap
                    
                    # Check previous indicator positions to avoid overlap
                    while any(abs(pos - y) < h for pos in indicator_positions):
                        y += h  # Increment the y position to avoid overlap

                    indicator_positions.append(y)  # Add the new position
                    added_comparisons.add(sorted_groups)  # Mark this comparison as added
                    ax.plot([x1, x2], [y, y], lw=1.5, c='black')  # Horizontal line only
                    ax.text((x1 + x2) * .5, y, get_significance_stars(p_value), ha='center', va='bottom', color='black')

    st.pyplot(fig)
    
    # Save the plot to a buffer for download
    buf = io.BytesIO()
    fig.savefig(buf, format='tiff')
    buf.seek(0)
    download_key = f"download_bar_{uuid.uuid4()}"
    st.download_button(label="Download Bar Chart TIFF", data=buf, file_name="bar_chart.tiff", mime="image/tiff", key=download_key)


def perform_statistical_analysis(data, samples):
    """Perform statistical analysis on the given data."""
    # Collect data for each sample into a list of arrays for statistical testing
    groups = [data[data.columns[data.columns.str.contains(sample)]].values.flatten() for sample in samples]
    
    if len(samples) == 2:
        # Perform t-test for two groups
        t_stat, p_value = stats.ttest_ind(*groups, nan_policy='omit')
        return ("T-test", p_value)
    else:
        # Perform ANOVA and Tukey's test for more than two groups
        f_stat, p_value = f_oneway(*groups)
        if p_value < 0.05:  # Proceed with post-hoc test if ANOVA is significant
            data_flat = np.concatenate(groups)
            labels = np.concatenate([[sample] * len(group) for sample, group in zip(samples, groups)])
            tukey_result = pairwise_tukeyhsd(data_flat, labels, 0.05)
            return ("ANOVA + Tukey's", p_value, tukey_result.summary())
        else:
            return ("ANOVA", p_value)

def main():
    st.title("xCelligence E-Plate 96 Data Analysis Tool")
    st.markdown("Upload your Excel files to start.")
    with st.sidebar:
        st.header("How to Use This Application")
        st.markdown("""
        ### Prerequisites
        - Export an .xls file from the xCelligence Machine with the layout and in the 'matrix' format.
        - Save the .xls file as an .xlsx file.
        - Do not include '-', ':' and avoid special characters in the sample names.
        - If loading multiple .xlsx files, they should have the same xCelligence measurement plate read time settings. Loading different times is Untested. 
        
        ### Step 1: Upload Excel Files
        - Click the "Upload Excel files" button.
        - Select one or more Excel files that contain 'Layout' and 'Cell Index' sheets.

        ### Step 2: Normalize and Transform Data
        - Once the files are uploaded, the layout and cell index data for each file will be displayed.
        - Check the displayed data to ensure it has been loaded correctly.
        - Drop wells that have bad readings.
        - Select a sample for normalization from the dropdown menu for each .xlsx file, usually this is the control.
        - Choose a base-time for ratio transformation.
        - The application will automatically process and adjust the time index for the uploaded files.
        
        ### Step 3: Adjust Settings
        - Use the controls to adjust various settings for your analysis.
        - You can toggle gridlines, adjust figure dimensions, and customize sample names and colors.

        ### Step 5: Visualize Data
        - Select the samples you want to include in the combined line graph.
        - Customize the appearance of the graph, including point size, point style, and colors.
        - The line graph will be generated automatically.

        ### Step 6: Analyze Specific Time Points
        - Select a specific time point from the dropdown menu to generate a bar graph.
        - Significance P $\leq$ {0.05, 0.005, 0.0005, 0.00005} = {* | ** | *** | ****} are automatically indicated.
        - Customize the bar graph appearance like in the line graph.

        ### Step 7: Download Results
        - Use the provided download buttons to export the generated graphs as TIFF files.
        - You can also download the adjusted data tables used for plotting.

        ### Tips for Best Results
        - Ensure your Excel files are properly formatted with the necessary sheets.
        - Use consistent sample naming conventions across files for easier analysis.
        - Experiment with different settings to find the best visualization for your data.
        """)

    uploaded_files = st.file_uploader("Upload Excel files", type="xlsx", accept_multiple_files=True)
    if uploaded_files:
        columns = st.columns(len(uploaded_files))
        all_data = []
        adjusted_time_dataframes = []
        base_times = []
        base_time_labels = []

        for file, col in zip(uploaded_files, columns):
            data_pivot, adjusted_time_data, base_time_seconds, base_time_selection = process_and_display(file, col)
            if not adjusted_time_data.empty:
                adjusted_time_dataframes.append(adjusted_time_data)
                base_times.append(base_time_seconds)
                base_time_labels.append(base_time_selection)
            all_data.append(data_pivot)

        combined_data = align_data_frames(all_data)

        if adjusted_time_dataframes:
            combined_adjusted_time_data = concatenate_dataframes(adjusted_time_dataframes)
            st.write("Combined Adjusted Time Data:")
            st.dataframe(combined_adjusted_time_data)

        plot_average_combined = st.checkbox("Plot average of replicates for combined graph")
        show_std_dev = st.checkbox("Show standard deviation for combined graph", value=True) if plot_average_combined else False
        grid_toggle_line = st.checkbox("Show Gridlines for Line Graph", value=True)
        show_base_lines = st.checkbox("Show Base Time Lines", value=True)

        available_samples = sorted(set([col.replace('_mean', '').replace('_std', '') for col in combined_adjusted_time_data.columns if '_mean' in col or '_std' in col]))
        selected_samples = st.multiselect("Select samples for combined line graph:", options=available_samples, key="samples_selection")

        if selected_samples:
        
            st.subheader("Customize Line Graph Sample Names and Colors")
            custom_names_line = {}
            custom_colors = {}
            for sample in selected_samples:
                custom_name_line = st.text_input(f"Custom name for {sample}:", value=sample, key=f"custom_name_{sample}")
                custom_names_line[sample] = custom_name_line
                custom_color = st.color_picker("Pick a color", value='#000000', key=f"custom_color_{sample}")
                custom_colors[sample] = custom_color

            fig_title = st.text_input("Figure Title", value="Normalized Cell Index vs Time")
            x_label = st.text_input("X-axis Label", value="Time (Hours)")
            y_label = st.text_input("Y-axis Label", value="Normalized Cell Index")
            use_palette = st.checkbox("Use preset palette", value=True)
            enhance_visibility = st.checkbox("Enhance point visibility", value=True)
            palette_options = list(sns.palettes.SEABORN_PALETTES.keys())
            point_size = st.slider("Select point size:", min_value=1, max_value=10, value=5)
            point_types = ['o', '^', 's', 'p', '*', '+', 'x']
            point_style = st.selectbox("Select point style:", options=point_types, key="line_graph_point_style_selection")
            
            fig_width_line = st.slider("Figure width", 5, 20, 12, key="line_fig_width")
            fig_height_line = st.slider("Figure height", 3, 15, 8, key="line_fig_height")

            if use_palette:
                palette = st.selectbox("Select color palette for line graph:", options=palette_options, key="palette_selectbox_line_graph")
                custom_colors = {sample: color for sample, color in zip(selected_samples, sns.color_palette(palette, n_colors=len(selected_samples)))}
            else:
                palette = list(custom_colors.values())

            dash_patterns = ['--', '-.', ':']
            dashes = [dash_patterns[i % len(dash_patterns)] for i in range(len(base_times))]

            plot_combined_data(combined_adjusted_time_data, selected_samples, plot_average_combined, show_std_dev, fig_title, x_label, y_label, palette, base_times, base_time_labels, dashes, grid_toggle_line, show_base_lines, custom_names_line, custom_colors, point_size, point_style, enhance_visibility, fig_width_line, fig_height_line)

            st.subheader("Customize Bar Graph")
    
            time_points = sort_time_strings(sorted(set(combined_adjusted_time_data.index)))
            selected_time_point = st.selectbox("Select time point for line graph (in hours):", options=time_points, key="line_graph_time_point_selection")
    

            fig_title = st.text_input("Bar Chart Title", value="X Hours X Mins")
            x_label = st.text_input("X-axis Label", value="Conditions")
            y_label = st.text_input("Y-axis Label", value="Normalized Cell Index ")
            fig_width = st.slider("Figure width", 5, 20, 12, key="bar_fig_width")
            fig_height = st.slider("Figure height", 3, 15, 8, key="bar_fig_height")
            show_grid_bar = st.checkbox("Show Gridlines for Bar Graph", value=True)

            custom_bar_colors = {}
            custom_names = {}
            for sample in selected_samples:
                custom_color = st.color_picker(f"Pick a color for {sample}:", value='#1f77b4', key=f"custom_bar_color_{sample}")
                custom_bar_colors[sample] = custom_color
                custom_name = st.text_input(f"Custom name for {sample}:", value=sample, key=f"custom_name_{sample}_bar")
                custom_names[sample] = custom_name

            if st.button("Plot Bar Graph at Selected Timepoint"):
                sample_data_df, summary_stats = print_sample_replicate_data(combined_adjusted_time_data, selected_time_point, selected_samples)
                
                if sample_data_df is not None and summary_stats is not None:
                    plot_bar_chart(sample_data_df, summary_stats, fig_title, x_label, y_label, custom_bar_colors, custom_names, fig_width, fig_height, show_grid_bar)

if __name__ == "__main__":
    main()
