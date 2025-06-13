这是<a href="https://openmmlab.com">
        <i><font size="4">别人家(openmmlab)的框架库</font></i>
      </a> ，拿来自己训模型。特此说明！

English | [中文readME说明](/README_zh-CN.md)



## Installation

Below are quick steps for installation:

```shell
conda create -n open-mmlab python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate open-mmlab
pip install openmim
git clone https://github.com/open-mmlab/mmpretrain.git
cd mmpretrain
mim install -e .
```

Please refer to [installation documentation](https://mmpretrain.readthedocs.io/en/latest/get_started.html) for more detailed installation and dataset preparation.

For multi-modality models support, please install the extra dependencies by:

```shell
mim install -e ".[multimodal]"
```



## Acknowledgement

MMPreTrain is an open source project that is contributed by researchers and engineers from various colleges and companies. We appreciate all the contributors who implement their methods or add new features, as well as users who give valuable feedbacks.
We wish that the toolbox and benchmark could serve the growing research community by providing a flexible toolkit to reimplement existing methods and supporting their own academic research.

## Citation

If you find this project useful in your research, please consider cite:

```BibTeX
@misc{2023mmpretrain,
    title={OpenMMLab's Pre-training Toolbox and Benchmark},
    author={MMPreTrain Contributors},
    howpublished = {\url{https://github.com/open-mmlab/mmpretrain}},
    year={2023}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).
