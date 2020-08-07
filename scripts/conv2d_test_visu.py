import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


try:
  matplotlib.use("GTK3Agg")
except ImportError:
  pass


def plot_exec_time_vs_batch_size(data):
  g = sns.FacetGrid(data, row="input_channel", col="output_channel", hue="engine", sharey=False)
  g.map(sns.lineplot, "batch_size", "execution time")
  g.add_legend()
  plt.gcf().canvas.set_window_title("Execution time vs batch size")

def plot_exec_time_vs_iter_for_specific_bs_imgsize(data, batch_size=1, image_size=224):
  data = data[data['batch_size'].eq(batch_size)]
  data = data[data['image_size'].eq(image_size)]
  g = sns.FacetGrid(data, row="input_channel", col="output_channel", hue="engine", sharey=False)
  g.map(sns.lineplot, "iter", "execution time")
  g.add_legend()
  plt.gcf().canvas.set_window_title(f"Execution time per iteration for {image_size} pix input and batch size of {batch_size}")

def plot_exec_time_vs_iter_for_specific_channels(data, input_channel=64, output_channel=64):
  # User can manually change the value depending on what he want to plot
  data = data[data['input_channel'].eq(input_channel)]
  data = data[data['output_channel'].eq(output_channel)]
  g = sns.FacetGrid(data, row="batch_size", col="image_size", hue="engine", sharey=False)
  g.map(sns.lineplot, "iter", "execution time")
  g.add_legend()
  plt.gcf().canvas.set_window_title(f"Execution time per iteration for {input_channel} input channels and {output_channel} output channels")

def main():
  sns.set(style="darkgrid")
  data = pd.read_csv("../output.csv")
  plot_exec_time_vs_batch_size(data)
  plot_exec_time_vs_iter_for_specific_bs_imgsize(data)
  plot_exec_time_vs_iter_for_specific_channels(data)
  plt.show()
  


  
if __name__ == "__main__":
  main()
