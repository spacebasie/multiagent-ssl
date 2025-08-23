import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the two CSV files
federated_df = pd.read_csv('simulations/fed_val.csv')
centralized_df = pd.read_csv('simulations/centr_val.csv')

# --- Data Cleaning (Important!) ---
# The downloaded CSV might contain non-numeric values (e.g., "NaN") for steps
# where a metric wasn't logged. We'll drop those rows to ensure clean plotting.
# federated_df.dropna(subset=['Step', 'eval/global_knn_accuracy'], inplace=True)
# centralized_df.dropna(subset=['Step', 'eval/knn_accuracy'], inplace=True)


# --- Plotting ---
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