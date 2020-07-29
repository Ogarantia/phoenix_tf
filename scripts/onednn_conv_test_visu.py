import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("GTK3Agg")


def plot_exec_time_vs_batch_size(data):
  g = sns.FacetGrid(data, row="input_channel", col="output_channel", hue="engine", sharey=False)
  g.map(sns.lineplot, "batch_size", "execution time")
  g.add_legend()
  plt.show()

def plot_exec_time_vs_iter_for_specific_bs_imgsize(data):
  data = data[data['batch_size'].eq(1)]
  data = data[data['image_size'].eq(224)]
  g = sns.FacetGrid(data, row="input_channel", col="output_channel", hue="engine", sharey=False)
  g.map(sns.lineplot, "iter", "execution time")
  g.add_legend()
  plt.show()

def plot_exec_time_vs_iter_for_specific_channels(data):
  # User can manually change the value depending on what he want to plot
  data = data[data['input_channel'].eq(64)]
  data = data[data['output_channel'].eq(64)]
  g = sns.FacetGrid(data, row="batch_size", col="image_size", hue="engine", sharey=False)
  g.map(sns.lineplot, "iter", "execution time")
  g.add_legend()
  plt.show()

def main():
  sns.set(style="darkgrid")
  data = pd.read_csv("../output.csv")
  # plot_exec_time_vs_batch_size(data)
  plot_exec_time_vs_iter_for_specific_bs_imgsize(data)
  # plot_exec_time_vs_iter_for_specific_channels(data)
  


  
if __name__ == "__main__":
  main()
