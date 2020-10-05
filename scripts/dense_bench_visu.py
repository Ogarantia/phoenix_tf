import seaborn as sns
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os 

try:
  matplotlib.use("GTK3Agg")
except ImportError:
  pass

context = {
  "op_name": "dense",
  "logdir": "."
}

def plot_exec_time_vs_batch_size(data, use_bias=True):
  data = data[data['use_bias'].eq(use_bias)]
  g = sns.FacetGrid(data, row="input_features", 
                          col="output_features", 
                          hue="engine", 
                          sharey=False)
  g.map(sns.lineplot, "batch_size", "execution time")
  g.set_titles('use_bias %d, {row_name} to {col_name} features' % (use_bias))
  g.set_ylabels('Execution time, ms')
  g.add_legend()
  plt.gcf().canvas.set_window_title("Execution time vs batch size")
  plt.savefig(f'{context["logdir"]}/{context["op_name"]}/{context["op_name"]}_execution_time_vs_batch_size_for_use_bias_{use_bias}.png')

def plot_exec_time_vs_iter_for_specific_batchsize_imgsize(data, 
                                                          batch_size=16, 
                                                          use_bias=True):
  data = data[data['batch_size'].eq(batch_size)]
  data = data[data['use_bias'].eq(use_bias)]
  g = sns.FacetGrid(data, row="input_features", 
                          col="output_features", 
                          hue="engine", 
                          sharey=False)
  g.map(sns.lineplot, "iter", "execution time")
  g.set_titles('use_bias %d, {row_name} to {col_name} features' % (use_bias))
  g.set_ylabels('Execution time, ms')
  g.add_legend()
  plt.gcf().canvas.set_window_title(f"Execution time per iteration for use_bias {use_bias} and batch size of {batch_size}")
  plt.savefig(f'{context["logdir"]}/{context["op_name"]}/{context["op_name"]}_Execution_time_per_iteration_for_use_bias_{use_bias}_and_batch_size_of_{batch_size}.png')

def plot_exec_time_vs_iter_for_specific_features(data, 
                                                 input_features=4096, 
                                                 output_features=4096):
  # Users can manually modify the value depending on what they intend to plot
  data = data[data['input_features'].eq(input_features)]
  data = data[data['output_features'].eq(output_features)]
  g = sns.FacetGrid(data, row="batch_size", 
                          col="use_bias", 
                          hue="engine", 
                          sharey=False)
  g.map(sns.lineplot, "iter", "execution time")
  g.set_titles('use_bias {col_name}, batch of {row_name}')
  g.set_ylabels('Execution time, ms')
  g.add_legend()
  plt.gcf().canvas.set_window_title(f"Execution time per iteration for {input_features} input features and {output_features} output features")
  plt.savefig(f'{context["logdir"]}/{context["op_name"]}/{context["op_name"]}_Execution_time_per_iteration_for_{input_features}_input_features_and_{output_features}_output_features.png')

def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('--logdir', "-ld", type=str, default=".", help='')
  parser.add_argument('--operation_name', "-op", type=str, default="dense", help='')
  args = parser.parse_args()

  context["logdir"]  = args.logdir
  context["op_name"] = args.operation_name
  
  if not os.path.exists(f'{context["logdir"]}/{context["op_name"]}'):
      print ("[Dense Bench_visu.py: Error: The directory ",context["logdir"],"/",context["op_name"]," doesn't exist.")
      return 1
      
  sns.set(style="darkgrid")
  data = pd.read_csv(f'{context["logdir"]}/{context["op_name"]}/{context["op_name"]}_output.csv')
  plot_exec_time_vs_batch_size(data, use_bias=True)
  plot_exec_time_vs_batch_size(data, use_bias=False)
  plot_exec_time_vs_iter_for_specific_batchsize_imgsize(data)
  plot_exec_time_vs_iter_for_specific_features(data)
  # plt.show()


if __name__ == "__main__":
  main()
