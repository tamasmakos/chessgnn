# pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel already ships torch 2.7.1+cu118
FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-devel

# Install PyTorch Geometric and dependencies
RUN pip install torch_geometric
RUN pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.7.1+cu118.html
