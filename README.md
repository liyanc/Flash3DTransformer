<div align="center">
  <h1>Flash3D Point Transformer⚡</h1>
  <strong>
    <a href="https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Flash3D_Super-scaling_Point_Transformers_through_Joint_Hardware-Geometry_Locality_CVPR_2025_paper.html">[CVPR 2025 Highlight✨]</a>
    <br>
    <a href="https://cvpr.thecvf.com/media/PosterPDFs/CVPR%202025/33156.png?t=1748866393.0026948">[Project Poster]</a>
  </strong>
</div>
<br>

> Flash3D is a scalable 3D point cloud transformer backbone built for top speed and minimal memory cost by targeting modern GPU architectures.

## 📰 News
- **(04/02/2025)**: Upgraded to ThunderKittens 2.

## 🚀 Installation
Flash3D requires newer versions of **CUDA >= 12.4**, **gcc >= 12**, **PyTorch >= 2.4**, and **[TransformerEngine](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/index.html)**. Earlier versions might work but have not been tested.

### Docker Workflow
We recommend using our provided Docker environment for fast development. Please refer to the [doc/build_docker.md](doc/build_docker.md) for instructions.

### Quick Install
If you prefer to set up the environment from scratch, you can install the package as follows:
```bash
# Clone the repository
git clone --recursive https://github.com/liyanc/Flash3DTransformer.git Flash3D
cd Flash3D

# Declare target GPU architecture (e.g., SM80 for A100, SM89 for L4, SM90a for H100)
export F3D_CUDA_ARCH=89

# Install Flash3D
python setup.py install
```

## 🧪 Unit Tests
Flash3D includes a comprehensive suite of unit tests in the `test/` directory to ensure functional correctness and facilitate CI/CD.
Many tests utilize the KITTI dataset, which requires configuring the path to its root directory prior to execution. 
For detailed setup and run instructions, please refer to the documentation in [doc/unit_tests.md](doc/unit_tests.md).

You can run all tests with two commands:
```bash
export KITTI_RT={You KITTI dataset root}
python -m unittest discover -s tests
```

## ⚡ Running Flash3D
Below is a minimal example of how to run Flash3D on a batch of sample 3D point clouds:

```python
import torch
from flash3dxfmr.layers import Flash3D

# Generate or load your configuration
config = ...

# Initialize the Flash3D model
f3d_xfmr = Flash3D(config)

# Load input data and process it with Flash3D
input_pcd, input_feat, batch_sep = ...
output_feats = f3d_xfmr(input_pcd, input_feat, batch_sep)

# Your results
print(output_feats.shape)
```

## 🗺️ Roadmap
Here are some of our future development goals:
- [ ] Automate CI/CD with GitHub Actions
- [ ] Attach CI/CD with PyPA for `pip install`
- [ ] Native fusion with FlashAttention-3 and beyond for best Hopper and Blackwell(ThorU) support.
- [ ] Add multi-architecture compilation for Ampere, Ada, Hopper, and Thor in one package
- [ ] Push pre-trained NuScenes models
- [ ] Include more flexible position encoding modules

## 📜 Citation
#### [Flash3D: Super-scaling Point Transformers through Joint Hardware-Geometry Locality](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Flash3D_Super-scaling_Point_Transformers_through_Joint_Hardware-Geometry_Locality_CVPR_2025_paper.html)
If you find our work useful in your research, please consider citing it as follows:

```bibtex
@InProceedings{Chen_2025_CVPR,
    author    = {Chen, Liyan and Meyer, Gregory P. and Zhang, Zaiwei and Wolff, Eric M. and Vernaza, Paul},
    title     = {Flash3D: Super-scaling Point Transformers through Joint Hardware-Geometry Locality},
    booktitle = {Proceedings of the Computer Vision and Pattern Recognition Conference (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {6595-6604}
}
```

---

## 🤝 Contributing
Contributions are welcome and greatly appreciated! This is the community version of Flash3D, and we encourage you to help improve it. Please adhere to standard community guidelines for cooperative and respectful conduct.

## 📣 Acknowledgements
This work would not be possible without the foundational contributions of the following projects. We extend our sincere gratitude to their authors and maintainers.

- **[Flash3D](https://github.com/cruise-automation/Flash3D)**: This project is a community-driven fork of the original Flash3D. We are deeply grateful for the original authors and General Motors for open-sourcing the core components.

- **[ThunderKittens](https://github.com/HazyResearch/ThunderKittens)**: Our development has greatly benefited from the architecture and components provided by the ThunderKittens project. We thank its contributors for their excellent and pioneering research.

## 📄 License
This project is released under the terms of the license found in the [LICENSE](LICENSE) file. Licenses for third-party dependencies are available in the `LICENSES/` directory.