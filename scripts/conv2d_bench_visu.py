import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

try:
  matplotlib.use("GTK3Agg")
except ImportError:
  pass

operation = "conv2d"

def plot_exec_time_vs_batch_size(data, image_size=112):
  data = data[data['image_size'].eq(image_size)]
  g = sns.FacetGrid(data, row="input_channel", col="output_channel", hue="engine", sharey=False)
  g.map(sns.lineplot, "batch_size", "execution time")
  g.set_titles('%d pix, {row_name} to {col_name} channels' % (image_size))
  g.set_ylabels('Execution time, ms')
  g.add_legend()
  plt.gcf().canvas.set_window_title("Execution time vs batch size")
  plt.savefig(f'{operation}/{operation}_execution_time_vs_batch_size_for_{image_size}_pix_input.png')

def plot_exec_time_vs_iter_for_specific_batchsize_imgsize(data, batch_size=1, image_size=112):
  data = data[data['batch_size'].eq(batch_size)]
  data = data[data['image_size'].eq(image_size)]
  g = sns.FacetGrid(data, row="input_channel", col="output_channel", hue="engine", sharey=False)
  g.map(sns.lineplot, "iter", "execution time")
  g.set_titles('%d pix, {row_name} to {col_name} channels' % (image_size))
  g.set_ylabels('Execution time, ms')
  g.add_legend()
  plt.gcf().canvas.set_window_title(f"Execution time per iteration for {image_size} pix input and batch size of {batch_size}")
  plt.savefig(f"{operation}/{operation}_Execution_time_per_iteration_for_{image_size}_pix_input_and_batch_size_of_{batch_size}.png")

def plot_exec_time_vs_iter_for_specific_channels(data, input_channel=32, output_channel=32):
  # User can manually change the value depending on what he want to plot
  data = data[data['input_channel'].eq(input_channel)]
  data = data[data['output_channel'].eq(output_channel)]
  g = sns.FacetGrid(data, row="batch_size", col="image_size", hue="engine", sharey=False)
  g.map(sns.lineplot, "iter", "execution time")
  g.set_titles('{col_name} pix, batch of {row_name}')
  g.set_ylabels('Execution time, ms')
  g.add_legend()
  plt.gcf().canvas.set_window_title(f"Execution time per iteration for {input_channel} input channels and {output_channel} output channels")
  plt.savefig(f"{operation}/{operation}_Execution_time_per_iteration_for_{input_channel}_input_channels_and_{output_channel}_output_channels.png")

def main():
  sns.set(style="darkgrid")
  data = pd.read_csv(f"{operation}/{operation}_output.csv")
  plot_exec_time_vs_batch_size(data)
  plot_exec_time_vs_iter_for_specific_batchsize_imgsize(data)
  plot_exec_time_vs_iter_for_specific_channels(data)
  # plt.show()
  


  
if __name__ == "__main__":
  main()
