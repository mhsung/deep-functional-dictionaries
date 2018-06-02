## Deep Functional Dictionaries: Learning Consistent Semantic Structures on 3D Models from Functions 

[Minhyuk Sung](http://mhsung.github.io), [Hao Su](http://cseweb.ucsd.edu/~haosu/), [Ronald Yu](https://ronaldiscool.github.io/), and [Leonidas Guibas](https://geometry.stanford.edu/member/guibas/)<br>
[[arXiv]](https://arxiv.org/abs/1805.09957)

### Citation
```
@misc{Sung:2018,
  Author = {Minhyuk Sung and Hao Su and Ronald Yu and Leonidas Guibas},
  Title = {Deep Functional Dictionaries: Learning Consistent Semantic Structures on 3D Models from Functions},
  Year = {2018},
  Eprint = {arXiv:1805.09957},
}
```

### Introduction
This neural-network-based framework analyzes an uncurated collection of 3D models from the same category and learns two important types of semantic relations among full and partial shapes: complementarity and interchangeability. The former helps to identify which two partial shapes make a complete plausible object, and the latter indicates that interchanging two partial shapes from different objects preserves the object plausibility. These two relations are modeled as *fuzzy set* operations performed across the *dual* partial shape embedding spaces, and within each space, respectively, and *jointly* learned by encoding partial shapes as *fuzzy sets* in the dual spaces.

### Requirements
- Numpy (tested with ver. 1.14.2)
- TensorFlow-gpu (tested with ver. 1.4.0)
- Gurobi (tested with 7.5.1)
  The [Gurobi](http://www.gurobi.com/) is a commercial optimization solver. The free academic license can be obtained from [here](http://www.gurobi.com/academia/for-universities).

### Applications
#### ShapeNet semantic part segmentation
Download the following data in your preferred location:
```
wget https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip
```

In `global_variables.py`, change `g_shapent_parts_dir` path to the directory containing the data.

We provide the pretrained model
```
cd experiments
wget https://shapenet.cs.stanford.edu/media/minhyuk/deep-functional-dictionaries/pretrained/ShapeNetParts_10_1.000000.tgz
tar xzvf ShapeNetParts_10_1.000000.tgz
rm -rf ShapeNetParts_10_1.000000.tgz
cd ..
```



### Acknowledgements
The files in [network/utils](network/utils) are directly brought from the [PointNet++](https://github.com/charlesq34/pointnet2).

### License
This code is released under the MIT License. Refer to [LICENSE](LICENSE) for details.

### To-Do
- [ ] Instruction for ShapeNet keypoint correspondence experiment.
- [x] Instruction for ShapeNet semantic part segmentation experiment.
- [ ] Instruction for S3DIS instance segmentation experiment.
- [ ] Instruction for MPI-FAUST human shape bases synchronization experiment.
- [ ] CVX code for the constrained least square problem.
