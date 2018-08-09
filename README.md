## Deep Functional Dictionaries: Learning Consistent Semantic Structures on 3D Models from Functions 

[Minhyuk Sung](http://mhsung.github.io), [Hao Su](http://cseweb.ucsd.edu/~haosu/), [Ronald Yu](https://ronaldiscool.github.io/), and [Leonidas Guibas](https://geometry.stanford.edu/member/guibas/)<br>
[[arXiv]](https://arxiv.org/abs/1805.09957)

### Citation
```
@misc{Sung:2018,
  Author = {Minhyuk Sung and Hao Su and Ronald Yu and Leonidas Guibas},
  Title = {Deep Functional Dictionaries: Learning Consistent Semantic Structures on 3D Models from
    Functions},
  Year = {2018},
  Eprint = {arXiv:1805.09957},
}
```

### Introduction
Various 3D semantic attributes such as segmentation masks, geometric features, keypoints, and materials can be encoded as per-point probe functions on 3D geometries. Given a collection of related 3D shapes, we consider how to jointly analyze such probe functions over different shapes, and how to discover common latent structures using a neural network --- even in the absence of any correspondence information. Our network is trained on point cloud representations of shape geometry and associated semantic functions on that point cloud. These functions express a shared semantic understanding of the shapes but are not coordinated in any way. For example, in a segmentation task, the functions can be indicator functions of arbitrary sets of shape parts, with the particular combination involved not known to the network. Our network is able to produce a small dictionary of basis functions for each shape, a dictionary whose span includes the semantic functions provided for that shape. Even though our shapes have independent discretizations and no functional correspondences are provided, the network is able to generate latent bases, in a consistent order, that reflect the shared semantic structure among the shapes. We demonstrate the effectiveness of our technique in various segmentation and keypoint selection applications.

### Requirements
- Numpy (tested with ver. 1.14.2)
- TensorFlow-gpu (tested with ver. 1.4.0)
- Gurobi / Gorubipy (tested with 7.5.1)<br>
  [Gurobi](http://www.gurobi.com/) is a commercial optimization solver. The free *academic* license can be obtained from [here](http://www.gurobi.com/academia/for-universities).<br>
  After installing, check whether gurobipy is properly installed with the following script:
  ```
  python -c "import gurobipy"
  ```

### Applications
#### ShapeNet Semantic Part Segmentation
Download and unzip the [PointNet](https://github.com/charlesq34/pointnet) part segmentation data in your preferred location.
```
wget https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip
unzip shapenet_part_seg_hdf5_data.zip
```

In `global_variables.py` file, change `g_shapent_parts_dir` path to the directory containing the data.

In `experiments`, train the network as follows:
```
./run_shapenet_parts.py --train
```
You can change the parameters (`k` and `\gammma` in the paper) with `-K` and `--l21_norm_weight` options, respectively.

We also provide the pretrained model for parameter `k=10` and `\gammma=1.0`:
```
cd experiments
wget https://shapenet.cs.stanford.edu/media/minhyuk/deep-functional-dictionaries/pretrained/ShapeNetParts_10_1.000000.tgz
tar xzvf ShapeNetParts_10_1.000000.tgz
rm -rf ShapeNetParts_10_1.000000.tgz
cd ..
```

Run the evaluation (Table 1 and 2 in the paper) with the same `./run_shapenet_parts.py` file (without `--train` option).
<br>


#### S3DIS Instance Segmentation
Download and unzip the [S3DIS](http://buildingparser.stanford.edu/dataset.html#Download) instance segmentation data in your preferred location.
(The data is provided by [SGPN](https://github.com/laughtervv/SGPN/issues/3).)
```
FILE_ID="1UjcXB2wMlLt5qwYPk5iSAnlhttl1AO9u"
OUT_FILE="S3DIS.zip"
CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$FILE_ID" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$FILE_ID" -O $OUT_FILE
rm -rf /tmp/cookies.txt
unzip S3DIS.zip
```

Download the train/test split files to the unzipped directory.
```
wget https://shapenet.cs.stanford.edu/media/minhyuk/deep-functional-dictionaries/data/S3DIS/train_hdf5_file_list.txt
wget https://shapenet.cs.stanford.edu/media/minhyuk/deep-functional-dictionaries/data/S3DIS/test_hdf5_file_list.txt
```

In `global_variables.py` file, change `g_S3DIS_dir` path to the directory containing the data.

In `experiments`, train the network as follows:
```
./run_S3DIS_instances.py --train
```
You can change the parameters (`k` and `\gammma` in the paper) with `-K` and `--l21_norm_weight` options, respectively.

We also provide the pretrained model for parameter `k=150` and `\gammma=1.0`:
```
cd experiments
wget https://shapenet.cs.stanford.edu/media/minhyuk/deep-functional-dictionaries/pretrained/S3DIS_150_1.000000.tgz
tar xzvf S3DIS_150_1.000000.tgz
rm -rf S3DIS_150_1.000000.tgz
cd ..
```

Run the evaluation (Table 3 in the paper) with the same `./run_shapenet_parts.py` file (without `--train` option).
<br>


### Acknowledgements
The files in [network/utils](network/utils) are directly brought from the [PointNet++](https://github.com/charlesq34/pointnet2).

### License
This code is released under the MIT License. Refer to [LICENSE](LICENSE) for details.

### To-Do
- [ ] Instruction for ShapeNet keypoint correspondence experiment.
- [x] Instruction for ShapeNet semantic part segmentation experiment.
- [x] Instruction for S3DIS instance segmentation experiment.
- [ ] Instruction for MPI-FAUST human shape bases synchronization experiment.
- [ ] CVX code for the constrained least square problem.
