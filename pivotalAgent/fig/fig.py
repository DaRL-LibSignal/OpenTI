import numpy as np
import matplotlib.pyplot as plt

# Create a matrix

rows, cols = 6, 5  # Adjust as needed
line1 = np.linspace(1, 0.1, 5).reshape(-1, 1).T
line2 = np.linspace(0.9, 0.2, 5).reshape(-1, 1).T
line3 = np.linspace(0.7, 0.12, 5).reshape(-1, 1).T
line4 = np.linspace(0.5, 0.14, 5).reshape(-1, 1).T
line5 = np.linspace(0.9, 0.21, 5).reshape(-1, 1).T
line6 = np.linspace(0.8, 0.15, 5).reshape(-1, 1).T
matrix = np.concatenate((line1, line2, line3, line4, line5, line6), axis=0) # (6, 5)

print(matrix)

# matrix = np.linspace(1, 0, rows * cols).reshape(rows, cols)  # Generate values from -1 to 1

# Plot the matrix
fig, ax = plt.subplots()
cax = ax.imshow(matrix, aspect='auto', cmap='coolwarm')

x_labels = ["C1", "C2", "C3", "C4", "C5"]
y_labels = ["T1", "T2", "T3", "T4", "T5", "T6"]

# Adding grid to mimic cell borders
# ax.set_xticks(np.arange(matrix.shape[1]+1)-.5, minor=True)
# ax.set_yticks(np.arange(matrix.shape[0]+1)-.5, minor=True)

ax.tick_params(which="minor", size=0)  # Remove tick marks

ax.set_xticks(np.arange(matrix.shape[1]))
ax.set_xticklabels(x_labels, rotation=45)

ax.set_yticks(np.arange(matrix.shape[0]))
ax.set_yticklabels(y_labels, rotation=45) 

ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)


plt.colorbar(cax, label='Accuracy Score')

# Set axis labels
ax.set_xlabel('Prompt Components')
ax.set_ylabel('Sample Tasks')
ax.set_title('Performance Ablation on Function Level Prompts')



plt.savefig("./fig/save/function_scores.png", dpi=600)

plt.show()
