import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Script that generates graphs from training logs using Matplotlib.

# Define path to log file.
log_file = ''
# Define name of generated graph.
graph_name = ''

# DRUNet pattern matching.
pattern = (r"\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} : <epoch:\s*(\d+),.*?Average PSNR\s*:\s*([\d.]+)dB, "
           r"Average SSIM\s*:\s*([\d.]+), Average edgeJaccard\s*:\s*([\d.]+), Average loss\s*:\s*([\deE.-]+)")

# DnCNN pattern matching.
# pattern = r"\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3} : <epoch:\s*(\d+),.*?Average PSNR\s*:\s*([\d.]+)dB, Average SSIM\s*:\s*([\d.]+), Average loss\s*:\s*([\deE.-]+)"

epochs, psnr_values, ssim_values, edge_jaccard_values, loss_values = [], [], [], [], []
# epochs, psnr_values, ssim_values, loss_values = [], [], [], []

with open(log_file, "r") as file:
    for line in file:
        match = re.search(pattern, line)
        if match:
            epoch = int(match.group(1))
            psnr = float(match.group(2))
            ssim = float(match.group(3))
            edge_jaccard = float(match.group(4))
            loss = float(match.group(4))

            epochs.append(epoch)
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            edge_jaccard_values.append(edge_jaccard)
            loss_values.append(loss)
            print(f"Matched: {line.strip()}")
            print(epoch, psnr, ssim, edge_jaccard, loss)
            # print(epoch, psnr, ssim, loss)
        else:
            print(f"No match: {line.strip()}")

if not epochs:
    print(f"No valid log entries found in {log_file}")
    exit()

plt.figure(figsize=(20, 14))

plt.subplot(2, 2, 1)
plt.plot(epochs, psnr_values, marker='o', color='b', label="PSNR (dB)")
plt.xlabel("Epoch")
plt.ylabel("PSNR (dB)")
plt.title("PSNR Over Epochs")
plt.legend()
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(epochs, ssim_values, marker='s', color='g', label="SSIM")
plt.xlabel("Epoch")
plt.ylabel("SSIM")
plt.title("SSIM Over Epochs")
plt.legend()
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(epochs, edge_jaccard_values, marker='^', color='r', label="Edge Jaccard")
plt.xlabel("Epoch")
plt.ylabel("Edge Jaccard")
plt.title("Edge Jaccard Over Epochs")
plt.legend()
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(epochs, loss_values, marker='x', color='m', label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Over Epochs")
plt.legend()
plt.grid()
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(10))

plt.tight_layout()
plt.savefig(graph_name, dpi=300)
plt.show()

print("Plot saved.")
