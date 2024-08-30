# LocalContextPropagation

This repo is an implementation of [LCPFormer: Towards Effective 3D Point Cloud Analysis via Local Context Propagation in Transformers](https://ieeexplore.ieee.org/document/10049597) (IEEE Transactions on Circuits and Systems for Video Technology)

![image](doc/lcp.png)

## Install

install [PointNet++](https://arxiv.org/abs/1706.02413) layers:

```
cd pointnet2
python setup.py install --user
cd ..
```

## Citation

If you find this code useful for your research, please cite the following paper.

```bibtex
@ARTICLE{10049597,
  author={Huang, Zhuoxu and Zhao, Zhiyou and Li, Banghuai and Han, Jungong},
  journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
  title={LCPFormer: Towards Effective 3D Point Cloud Analysis via Local Context Propagation in Transformers}, 
  year={2023},
  volume={33},
  number={9},
  pages={4985-4996},
  keywords={Three-dimensional displays;Point cloud compression;Transformers;Feature extraction;Task analysis;Convolution;Solid modeling;3D vision;point cloud learning;transformer;context propagation},
  doi={10.1109/TCSVT.2023.3247506}}
```
