# Visual location estimation (inetrpretability version)
This repository contains the PyTorch implementation of the papers 
[Leveraging EfficientNet and Contrastive Learning for Accurate Global-scale Location Estimation](https://dl.acm.org/doi/abs/10.1145/3460426.3463644)
and [Leveraging Selective Prediction for Reliable Image Geolocation](https://link.springer.com/chapter/10.1007/978-3-030-98355-0_31). 
It provides all necessary code and pre-trained models for the evaluation of the location estimation method on the 
datasets presented in the paper to facilitate the reproducibility of the results. It provides code for the 
Search within Cell (SwC) scheme for the accurate location prediction, and the calculation of the Prediction Density (PD) 
that indicates image localizability.

<img src="banner.png" width="100%">

This repository is modifies to include Grad-CAM for interpretability.

## Citation
If you use this code for your research, please consider citing our papers:
```bibtex
@inproceedings{kordopatis2021leveraging,
  title={Leveraging efficientnet and contrastive learning for accurate global-scale location estimation},
  author={Kordopatis-Zilos, Giorgos and Galopoulos, Panagiotis and Papadopoulos, Symeon and Kompatsiaris, Ioannis},
  booktitle={Proceedings of the International Conference on Multimedia Retrieval},
  year={2021}
}

@inproceedings{panagiotopoulos2022leveraging,
  title={Leveraging Selective Prediction for Reliable Image Geolocation},
  author={Panagiotopoulos, Apostolos and Kordopatis-Zilos, Giorgos and Papadopoulos, Symeon},
  booktitle={Proceedings of the International Conference on MultiMedia Modeling},
  year={2022},
}
```
