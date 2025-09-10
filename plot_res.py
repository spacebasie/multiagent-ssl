import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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

inv = pd.read_csv('simulations/inv.csv')
var = pd.read_csv('simulations/var.csv')
cov = pd.read_csv('simulations/cov.csv')
train_loss = pd.read_csv('simulations/train_loss.csv')

fed_alpha05 = pd.read_csv('simulations/fed_alpha05.csv')
fed_alpha5 = pd.read_csv('simulations/fed_alpha5.csv')
fed_alpha100 = pd.read_csv('simulations/fed_alpha100.csv')

lsp15fully = pd.read_csv('simulations/lsp15fully.csv')
lsp15disc = pd.read_csv('simulations/lsp15disc.csv')
lsp15rand = pd.read_csv('simulations/lsp15rand.csv')

lsp10fully = pd.read_csv('simulations/lsp10fully.csv')
lsp10disc = pd.read_csv('simulations/lsp10disc.csv')
lsp10rand = pd.read_csv('simulations/lsp10rand.csv')

lsp5fully = pd.read_csv('simulations/lsp5fully.csv')
lsp5disc = pd.read_csv('simulations/lsp5disc.csv')
lsp5rand = pd.read_csv('simulations/lsp5rand.csv')

ds_rand = pd.read_csv('simulations/ds_rand.csv')
ds_disc = pd.read_csv('simulations/ds_disc.csv')

ali_1 = pd.read_csv('simulations/ali_1.csv')
ali_5 = pd.read_csv('simulations/ali_5.csv')
ali_25 = pd.read_csv('simulations/ali_25.csv')
ali_50 = pd.read_csv('simulations/ali_50.csv')
ali_100 = pd.read_csv('simulations/ali_100.csv')
ali_125 = pd.read_csv('simulations/ali_125.csv')
ali_150 = pd.read_csv('simulations/ali_150.csv')

method_lsp05 = pd.read_csv('simulations/method_lsp05.csv')
method_lsp5 = pd.read_csv('simulations/method_lsp5.csv')
method_lsp100 = pd.read_csv('simulations/method_lsp100.csv')
method_lsp05disc = pd.read_csv('simulations/method_lsp05disc.csv')

lsp_05 = pd.read_csv('simulations/lsp_05.csv')
lsp_5 = pd.read_csv('simulations/lsp_5.csv')
lsp_100 = pd.read_csv('simulations/lsp_100.csv')

# --- Data Cleaning (Important!) ---
# The downloaded CSV might contain non-numeric values (e.g., "NaN") for steps
# where a metric wasn't logged. We'll drop those rows to ensure clean plotting.
# federated_df.dropna(subset=['Step', 'eval/global_knn_accuracy'], inplace=True)
# centralized_df.dropna(subset=['Step', 'eval/knn_accuracy'], inplace=True)

plot = 'lsp' # 'global accuracy', 'angles', 'learning_curve', 'tsne', 'loss_terms', lsp, method_lsp, lsp_alpha

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
    plt.savefig('figures/fed_vs_centr.pdf')

    # Save the figure or display it
    plt.show()

elif plot == 'angles':
    plt.figure(figsize=(10, 6))

    plt.plot(
        full_med['Step'],
        full_med['combo_domain_fully_connected_agents_4 - eval/angle_median'],
        label='Fully Connected Angles',
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
        label='Disconnected Angles',
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

    legend_handles = [
        Line2D([0], [0], color='blue', lw=3, label=r'$\bf{Fully\ Connected\ Angles}$'),  # bold label
        Line2D([0], [0], color='red', lw=3, label='Disconnected Angles')
    ]

    plt.title('Angle Misalignment for Connected vs Disconnected Graphs', fontsize=20)
    plt.xlabel('Communication Round')
    plt.ylabel('Angle (deg)')
    plt.legend(loc='upper left', handles=legend_handles, fontsize=16)  # Display the labels
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(15, 50)
    plt.tight_layout()
    plt.savefig("angles.pdf")  # vector, infinitely scalable
    plt.show()

elif plot == 'learning_curve':
    plt.figure(figsize=(10, 7))

    # Plot lines (normal thickness)
    plt.plot(
        fully_con['Step'],
        fully_con['combo_domain_fully_connected_agents_4 - eval/avg_combo_accuracy'],
        color='blue'
    )

    plt.plot(
        random['Step'],
        random['combo_domain_random_agents_4 - eval/avg_combo_accuracy'],
        color='green'
    )

    plt.plot(
        discon['Step'],
        discon['combo_domain_disconnected_agents_4 - eval/avg_combo_accuracy'],
        color='red'
    )

    plt.plot(
        classifieronly['Step'],
        classifieronly['combo_domain_fully_connected_agents_4 - eval/avg_combo_accuracy'],
        color='orange'
    )

    plt.plot(
        alignment['Step'],
        alignment['combo_domain_fully_connected_agents_4 - eval/avg_combo_accuracy'],
        color='purple'
    )

    # plt.plot(
    #     federated['Step'],
    #     federated['federated_agents_4_dataset_office_home - eval/global_knn_accuracy'],
    #     color='sandybrown'
    # )
    #
    # plt.plot(
    #     centralized['Step'],
    #     centralized['centralized_office_home_epochs_200 - eval/knn_accuracy'],
    #     color='skyblue'
    # )

    # Create custom legend handles with slightly thicker lines
    legend_handles = [
        Line2D([0], [0], color='blue', lw=3, label=r'$\bf{Our\ Method}$'),  # bold label
        Line2D([0], [0], color='green', lw=3, label='Random'),
        Line2D([0], [0], color='red', lw=3, label='Disconnected'),
        Line2D([0], [0], color='orange', lw=3, label='Classifier Only'),
        Line2D([0], [0], color='purple', lw=3, label='Alignment Only'),
        # Line2D([0], [0], color='sandybrown', lw=3, label='Federated'),
        # Line2D([0], [0], color='skyblue', lw=3, label='Centralized'),
    ]

    plt.title('Learning Curves with Our Method', fontsize=20)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')

    # Use the custom handles for the legend with increased font size
    plt.legend(handles=legend_handles, fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(20, 65)
    plt.tight_layout()
    plt.savefig("ablation_learningcurve.pdf")  # vector, infinitely scalable
    plt.show()

elif plot == 'tsne':
    plt.figure(figsize=(11, 6))

    tsne_df = pd.read_csv('simulations/tsne_data_agent_2.csv', header=1)
    class_labels = {
        0: 'Clipart', 1: 'Product', 2: 'Alarm Clock',
        3: 'Alarm Clock', 4: 'Backpack', 5: 'Batteries', 6: 'Bed',
        7: 'Bike', 8: 'Bottle', 9: 'Bucket'
    }
    colors = plt.cm.get_cmap('tab10', 10)

    # Create the scatter plot
    for class_id, class_name in class_labels.items():
        # Filter data for the current class
        class_data = tsne_df[tsne_df['label'] == class_id]
        if not class_data.empty:
            plt.scatter(
                class_data['tsne_dim_1'],
                class_data['tsne_dim_2'],
                label=class_name,
                color=colors(class_id),
                alpha=0.8,
                s=40
            )

    plt.title('t-SNE Feature Embeddings for Disconnected', fontsize=20)
    plt.xlabel('t-SNE Dimension 1', fontsize=14)
    plt.ylabel('t-SNE Dimension 2', fontsize=14)

    # Place legend outside the plot area for clarity
    # plt.legend(title='Classes', loc='lower right', fontsize=12)

    # Use a specific tight_layout to make space for the external legend
    # plt.tight_layout(rect=[0, 0, 0.85, 1])

# Add a grid for all plot types
    plt.grid(True)
    plt.savefig("tsne.pdf")
# Adjust layout for plots other than tsne, which has its own layout adjustment
    plt.tight_layout()
    plt.show()

elif plot == 'loss_terms':
    plt.figure(figsize=(10, 6))

    # plt.plot(
    #     train_loss['Step'],
    #     train_loss['centralized_office_home_epochs_200 - train/loss'],
    #     label='Training Loss',
    #     color='blue'
    # )

    plt.plot(
        inv['Step'],
        inv['centralized_office_home_epochs_200 - train/invariance_loss'],
        label='Invariance Loss',
        color='orange'
    )

    plt.plot(
        var['Step'],
        var['centralized_office_home_epochs_200 - train/variance_loss'],
        label='Variance Loss',
        color='green'
    )

    plt.plot(
        cov['Step'],
        cov['centralized_office_home_epochs_200 - train/covariance_loss'],
        label='Covariance Loss',
        color='red'
    )

    plt.title('Training Loss Components Over Time', fontsize=20)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Loss Value', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(0, 3.5)
    plt.tight_layout()
    plt.savefig("figures/loss_terms.pdf")
    plt.show()

elif plot == 'fed_het':
    plt.plot(
        fed_alpha05['Step'],
        fed_alpha05['federated_agents_5_dataset_cifar10 - eval/knn_accuracy'],
        label=r'$\alpha=0.5$',
        color='blue'
    )

    plt.plot(
        fed_alpha5['Step'],
        fed_alpha5['federated_agents_5_dataset_cifar10 - eval/knn_accuracy'],
        label=r'$\alpha=5$',
        color='orange'
    )

    plt.plot(
        fed_alpha100['Step'],
        fed_alpha100['federated_agents_5_dataset_cifar10 - eval/global_knn_accuracy'],
        label=r'$\alpha=100$',
        color='green'
    )

    plt.title('Federated Learning with Varying Data Heterogeneity', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('k-NN Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(45, 70)
    plt.tight_layout()
    plt.savefig("figures/fed_het.pdf")
    plt.show()

elif plot == 'lsp':
    plt.figure(figsize=(10, 6.5))

    plt.plot(
        lsp5fully['Step'],
        lsp5fully['decentralized_fully_connected_agents_5_hetero_label_skew_personalized - eval/avg_global_accuracy'],
        label='5 Agents Fully Connected',
        color='blue'
    )

    plt.plot(
        lsp5disc['Step'],
        lsp5disc['decentralized_disconnected_agents_5_hetero_label_skew_personalized - eval/avg_global_accuracy'],
        label='5 Agents Disconnected',
        color='orange'
    )

    plt.plot(
        lsp5rand['Step'],
        lsp5rand['decentralized_random_agents_5_hetero_label_skew_personalized - eval/avg_global_accuracy'],
        label='5 Agents Random',
        color='green'
    )

    plt.plot(
        lsp10fully['Step'],
        lsp10fully['decentralized_fully_connected_agents_10_hetero_label_skew_personalized - eval/avg_global_accuracy'],
        label='10 Agents Fully Connected',
        color='blue',
        linestyle='--'
    )

    plt.plot(
        lsp10disc['Step'],
        lsp10disc['decentralized_disconnected_agents_10_hetero_label_skew_personalized - eval/avg_global_accuracy'],
        label='10 Agents Disconnected',
        color='orange',
        linestyle='--'
    )

    plt.plot(
        lsp10rand['Step'],
        lsp10rand['decentralized_random_agents_10_hetero_label_skew_personalized - eval/avg_global_accuracy'],
        label='10 Agents Random',
        color='green',
        linestyle='--'
    )

    plt.plot(
        lsp15fully['Step'],
        lsp15fully['decentralized_fully_connected_agents_15_hetero_label_skew_personalized - eval/avg_global_accuracy'],
        label='15 Agents Fully Connected',
        color='blue',
        linestyle=':'
    )

    plt.plot(
        lsp15disc['Step'],
        lsp15disc['decentralized_disconnected_agents_15_hetero_label_skew_personalized - eval/avg_global_accuracy'],
        label='15 Agents Disconnected',
        color='orange',
        linestyle=':'
    )

    plt.plot(
        lsp15rand['Step'],
        lsp15rand['decentralized_random_agents_15_hetero_label_skew_personalized - eval/avg_global_accuracy'],
        label='15 Agents Random',
        color='green',
        linestyle=':'
    )

    plt.title('Different Graphs Learning with High Class Heterogeneity', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('k-NN Accuracy (%)', fontsize=14)
    plt.legend(fontsize=10, loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(30, 60)
    plt.xlim(5, 200)
    plt.tight_layout()
    plt.savefig("figures/lsp_het.pdf")
    plt.show()

elif plot == 'ds':
    plt.figure(figsize=(10, 6))

    plt.plot(
        ds_rand['Step'],
        ds_rand['decentralized_random_agents_5_alpha_100.0 - eval/avg_personalized_accuracy'],
        label='Random Graph',
        color='blue'
    )
    plt.plot(
        ds_disc['Step'],
        ds_disc['decentralized_disconnected_agents_5_alpha_100.0 - eval/avg_personalized_accuracy'],
        label='Disconnected Graph',
        color='orange'
    )

    plt.title('Communication Impact with Domain Shift', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('k-NN Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(30, 50)
    plt.xlim(0, 200)
    plt.tight_layout()
    plt.savefig("figures/ds.pdf")
    plt.show()

elif plot == 'align':
    plt.figure(figsize=(10, 6))

    plt.plot(
        ali_1['Step'],
        ali_1['combo_domain_fully_connected_agents_4 - eval/avg_combo_accuracy'],
        label=r'$\lambda=1$',
        color='green'
    )

    plt.plot(
        ali_5['Step'],
        ali_5['combo_domain_fully_connected_agents_4 - eval/avg_combo_accuracy'],
        label=r'$\lambda=5$',
        color='orange'
    )

    plt.plot(
        ali_25['Step'],
        ali_25['combo_domain_fully_connected_agents_4 - eval/avg_combo_accuracy'],
        label=r'$\lambda=25$',
        color='blue'
    )

    plt.plot(
        ali_50['Step'],
        ali_50['combo_domain_fully_connected_agents_4 - eval/avg_combo_accuracy'],
        label=r'$\lambda=50$',
        color='red'
    )

    plt.plot(
        ali_100['Step'],
        ali_100['combo_domain_fully_connected_agents_4 - eval/avg_combo_accuracy'],
        label=r'$\lambda=100$',
        color='purple'
    )

    plt.plot(
        ali_125['Step'],
        ali_125['combo_domain_fully_connected_agents_4 - eval/avg_combo_accuracy'],
        label=r'$\lambda=125$',
        color='brown'
    )

    plt.plot(
        ali_150['Step'],
        ali_150['combo_domain_fully_connected_agents_4 - eval/avg_combo_accuracy'],
        label=r'$\lambda=150$',
        color='pink'
    )

    plt.title('Effect of Alignment Strength on Performance', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(20, 60)
    plt.xlim(5, 75)
    plt.tight_layout()
    plt.savefig("figures/align.pdf")
    plt.show()

elif plot == 'method_lsp':
    plt.figure(figsize=(10, 6))

    plt.plot(
        method_lsp05['Step'],
        method_lsp05['combo_label_skew_fully_connected_agents_5 - eval/avg_combo_accuracy'],
        label=r'$\alpha=0.5$',
        color='blue'
    )

    plt.plot(
        method_lsp5['Step'],
        method_lsp5['combo_label_skew_fully_connected_agents_5 - eval/avg_combo_accuracy'],
        label=r'$\alpha=5$',
        color='orange'
    )

    plt.plot(
        method_lsp100['Step'],
        method_lsp100['combo_label_skew_fully_connected_agents_5 - eval/avg_combo_accuracy'],
        label=r'$\alpha=100$',
        color='green'
    )

    plt.plot(
        method_lsp05disc['Step'],
        method_lsp05disc['combo_label_skew_disconnected_agents_5 - eval/avg_combo_accuracy'],
        label=r'$\alpha=0.5$ Disconnected',
        color='red'
    )

    plt.title('Effect of Heterogeneity on Our Method', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(15, 60)
    plt.xlim(5, 200)
    plt.tight_layout()
    plt.savefig("figures/method_lsp.pdf")
    plt.show()

elif plot == 'lsp_alpha':
    plt.figure(figsize=(10, 6))

    plt.plot(
        lsp_05['Step'],
        lsp_05['decentralized_fully_connected_agents_5_hetero_label_skew_personalized - eval/avg_global_accuracy'],
        label=r'$\alpha=0.5$',
        color='blue'
    )

    plt.plot(
        lsp_5['Step'],
        lsp_5['decentralized_fully_connected_agents_5_hetero_label_skew_personalized - eval/avg_global_accuracy'],
        label=r'$\alpha=5$',
        color='orange'
    )

    plt.plot(
        lsp_100['Step'],
        lsp_100['decentralized_fully_connected_agents_5_hetero_label_skew_personalized - eval/avg_global_accuracy'],
        label=r'$\alpha=100$',
        color='green'
    )

    plt.title('Effect of Data Heterogeneity on Naive Performance', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('k-NN Accuracy (%)', fontsize=14)
    plt.legend(fontsize=12, loc='lower right')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.ylim(40, 65)
    plt.xlim(5, 200)
    plt.tight_layout()
    plt.savefig("figures/lsp_alpha.pdf")
    plt.show()