import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the two CSV files
federated_df = pd.read_csv('simulations/fed_val.csv')
centralized_df = pd.read_csv('simulations/centr_val.csv')

full_med = pd.read_csv('simulations/fully_med.csv')
full_25 = pd.read_csv('simulations/fully_25.csv')
full_75 = pd.read_csv('simulations/fully_75.csv')
disc = pd.read_csv('simulations/disc_med.csv')
disc_25 = pd.read_csv('simulations/disc_25.csv')
disc_75 = pd.read_csv('simulations/disc_75.csv')

fully_con = pd.read_csv('simulations/fully_con.csv')
random = pd.read_csv('simulations/random.csv')
discon = pd.read_csv('simulations/discon.csv')
classifieronly = pd.read_csv('simulations/classifieronly.csv')
alignment = pd.read_csv('simulations/alignment.csv')
federated = pd.read_csv('simulations/federated.csv')
centralized = pd.read_csv('simulations/centralized.csv')

# --- Data Cleaning (Important!) ---
# The downloaded CSV might contain non-numeric values (e.g., "NaN") for steps
# where a metric wasn't logged. We'll drop those rows to ensure clean plotting.
# federated_df.dropna(subset=['Step', 'eval/global_knn_accuracy'], inplace=True)
# centralized_df.dropna(subset=['Step', 'eval/knn_accuracy'], inplace=True)

plot = 'learning_curve' # 'global accuracy', 'angles', 'learning_curve'

# --- Plotting ---
if plot == 'global accuracy':
    plt.figure(figsize=(10, 6))

    # Plot the federated run data
    plt.plot(
        federated_df['Step'],
        federated_df['federated_agents_5_dataset_cifar10 - eval/global_knn_accuracy'],
        label='Federated Benchmark Validation 5-Agents',
        color='sandybrown' # Matches your original plot color
    )

    # Plot the centralized run data
    plt.plot(
        centralized_df['Step'],
        centralized_df['centralized_cifar10_epochs_200 - eval/knn_accuracy'],
        label='Centralized Benchmark Validation',
        color='skyblue' # Matches your original plot color
    )

    # --- Customize the Plot ---
    plt.title('Federated vs. Centralized Training Accuracy')
    plt.xlabel('Step')
    plt.ylabel('k-NN Accuracy (%)')
    plt.legend() # Display the labels
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(40, 80)
    plt.tight_layout()

    # Save the figure or display it
    plt.show()

elif plot == 'angles':
    plt.figure(figsize=(10, 6))

    plt.plot(
        full_med['Step'],
        full_med['combo_domain_fully_connected_agents_4 - eval/angle_median'],
        label='Fully Connected Median Angle Misalignment',
        color='blue'
    )
    plt.scatter(
        full_25['Step'],
        full_25['combo_domain_fully_connected_agents_4 - eval/angle_25th_percentile'],
        # label='Fully Connected 25th Percentile',
        color='blue',
        marker='x'
    )

    plt.scatter(
        full_75['Step'],
        full_75['combo_domain_fully_connected_agents_4 - eval/angle_75th_percentile'],
        # label='Fully Connected 75th Percentile',
        color='blue',
        marker='x'
    )

    plt.plot(
        disc['Step'],
        disc['combo_domain_disconnected_agents_4 - eval/angle_median'],
        label='Disconnected Median Angle Misalignment',
        color='red'
    )

    plt.scatter(
        disc_25['Step'],
        disc_25['combo_domain_disconnected_agents_4 - eval/angle_25th_percentile'],
        # label='Disconnected 25th Percentile',
        color='red',
        marker='x'
    )

    plt.scatter(
        disc_75['Step'],
        disc_75['combo_domain_disconnected_agents_4 - eval/angle_75th_percentile'],
        # label='Disconnected 75th Percentile',
        color='red',
        marker='x'
    )

    plt.title('Angle Misalignment for Connected vs Disconnected Graphs')
    plt.xlabel('Communication Round')
    plt.ylabel('Angle (deg)')
    plt.legend()  # Display the labels
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(15, 50)
    plt.tight_layout()

    plt.show()

elif plot == 'learning_curve':
    plt.figure(figsize=(10, 6))

    plt.plot(
        fully_con['Step'],
        fully_con['combo_domain_fully_connected_agents_4 - eval/avg_combo_accuracy'],
        label='Our Method',
        color='blue'
    )

    plt.plot(
        random['Step'],
        random['combo_domain_random_agents_4 - eval/avg_combo_accuracy'],
        label='Random',
        color='green'
    )

    plt.plot(
        discon['Step'],
        discon['combo_domain_disconnected_agents_4 - eval/avg_combo_accuracy'],
        label='Disconnected',
        color='red'
    )

    plt.plot(
        classifieronly['Step'],
        classifieronly['combo_domain_fully_connected_agents_4 - eval/avg_combo_accuracy'],
        label='Classifier Only',
        color='orange'
    )

    plt.plot(
        alignment['Step'],
        alignment['combo_domain_fully_connected_agents_4 - eval/avg_combo_accuracy'],
        label='Alignment Only',
        color='purple'
    )

    plt.plot(
        federated['Step'],
        federated['federated_agents_4_dataset_office_home - eval/global_knn_accuracy'],
        label='Federated',
        color='sandybrown'
    )

    plt.plot(
        centralized['Step'],
        centralized['centralized_office_home_epochs_200 - eval/knn_accuracy'],
        label='Centralized',
        color='skyblue'
    )

    plt.title('Learning Curves for Different Architectures')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()  # Display the labels
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(20, 70)
    plt.tight_layout()
    plt.show()