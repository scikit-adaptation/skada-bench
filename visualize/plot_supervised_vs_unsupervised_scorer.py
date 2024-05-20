import argparse
import pandas as pd
import matplotlib.pyplot as plt

# Function to create the scatter plot
def create_scatter_plot(input_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_file)
    
    # Filter out the rows where the scorer is 'supervised'
    supervised_df = df[df['scorer'] == 'supervised']
    other_df = df[df['scorer'] != 'supervised']

    # We remove the best_scorer rows
    other_df = other_df[other_df['scorer'] != 'best_scorer']
    
    # Create a color palette and marker list for unique estimators and scorers
    unique_estimators = other_df['estimator'].unique()
    unique_scorers = other_df['scorer'].unique()

    cmaps = [plt.cm.Reds, plt.cm.Greens, plt.cm.Blues, plt.cm.Purples, plt.cm.Oranges]
    estimator_cmap_dict = {}

    for da_type, cmap in zip(df['type'].unique(), cmaps):
        unique_type_estimators = df['estimator'][df['type'] == da_type].unique()
        for idx, estimator in enumerate(unique_type_estimators):
            estimator_cmap_dict[estimator] = cmap((idx+0.5) / len(unique_type_estimators))

    
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', 'P', '*', 'X', 'd'][:len(unique_scorers)]
    
    # Plot the supervised scores
    fig, ax = plt.subplots()
    
    for idx, scorer in enumerate(unique_scorers):
        for idy, estimator in enumerate(unique_estimators):

            # Filter rows for the specific scorer and estimator
            filtered_df = other_df[(other_df['scorer'] == scorer) & (other_df['estimator'] == estimator)]
            supervised_accuracy = supervised_df[supervised_df['estimator'] == estimator]
            if not filtered_df.empty and supervised_accuracy.size > 0:
                # Scatter plot
                ax.scatter(
                    supervised_accuracy['accuracy-mean'].mean(),
                    filtered_df['accuracy-mean'].mean(),
                    label=f'{estimator}-{scorer}', 
                    color=estimator_cmap_dict[estimator],
                    alpha=0.5,
                    marker=markers[idx],
                )


    # Plot diagonal line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')

    # Labeling the plot
    ax.set_xlabel('Accuracy (Supervised Score)')
    ax.set_ylabel('Accuracy (Other Scores)')
    ax.set_title('Scatter Plot of Accuracy Scores')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # # To display the legend grouped by type
    # handles, labels = plt.gca().get_legend_handles_labels()
    # labels = [label.split('-')[0] for label in labels]

    # # Create a list of handles with shape 'o'
    # circle_handles = [handle for handle, label in zip(handles, labels) if hasattr(handle, 'get_marker') and handle.get_marker() == 'o']


    # # Get the associated labels
    # circle_labels = [label for handle, label in zip(handles, labels) if hasattr(handle, 'get_marker') and handle.get_marker() == 'o']

    # import pdb; pdb.set_trace()

    # plt.legend(circle_handles, circle_labels, loc='upper right', bbox_to_anchor=(1.2, 1))

    plt.tight_layout()
    plt.show()

# Main function to parse arguments and call the plotting function
def main():
    parser = argparse.ArgumentParser(description="Create a scatter plot to compare supervised and unsupervised scorers")
    parser.add_argument('input_file', type=str, help="Path to the input CSV file")
    
    args = parser.parse_args()
    create_scatter_plot(args.input_file)

if __name__ == "__main__":
    main()
