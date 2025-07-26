# analysis/plot_results.ipynb

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load summary or ablation results
df = pd.read_csv("../results/summary.csv")

# Maybe we want to compare "teacher_lr" vs. "test_acc" for each method
sns.set_style("whitegrid")
plt.figure(figsize=(8,6))

sns.lineplot(
    data=df,
    x="teacher_lr",
    y="test_acc",
    hue="method",
    marker="o"
)

plt.title("Ablation: teacher_lr vs test_acc by method")
plt.xlabel("Teacher Learning Rate")
plt.ylabel("Test Accuracy (%)")
plt.legend(title="Method")
plt.tight_layout()
plt.show()
