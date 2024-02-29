# GeoSAM: Fine-tuning SAM with Sparse and Dense Visual Prompting for Automated Segmentation of Mobility Infrastructure

This repository is dedicated to the work of GeoSAM. Please find the paper here: <a href="https://arxiv.org/abs/2311.11319">Link</a>


Also, please find the <a href="https://waynestateprod-my.sharepoint.com/:u:/g/personal/hm4013_wayne_edu/EXvJFrshs9RAm68KYnkKJ7gB4D4gB65CCXmasoDYUIplMw?e=6h7CKx">link</a> for the weights.


This work has been submitted. Waiting for the decision.


## Abstract:
<p class="justified-text">The Segment Anything Model (SAM) has shown impressive performance when applied to natural image segmentation. However, it struggles with geographical images like aerial and satellite imagery, especially when segmenting mobility infrastructure including roads, sidewalks, and crosswalks. This inferior performance stems from the narrow features of these objects, their textures blending into the surroundings, and interference from objects like trees, buildings, vehicles, and pedestrians - all of which can disorient the model to produce inaccurate segmentation maps. To address these challenges, we propose Geographical SAM (GeoSAM), a novel SAM-based framework that implements a fine-tuning strategy using the dense visual prompt from zero-shot learning, and the sparse visual prompt from a pre-trained CNN segmentation model. The proposed GeoSAM outperforms existing approaches for geographical image segmentation, specifically by 26%, 7%, and 17% for road infrastructure, pedestrian infrastructure, and on average, respectively, representing a momentous leap in leveraging foundation models to segment mobility infrastructure including both road and pedestrian infrastructure in geographical images.</p>

<img src="Pipeline.png" alt="GeoSAM">

## Citations

If these codes are helpful for your study, please cite:

```bibtex
@article{sultan2023geosam,
  title={GeoSAM: Fine-tuning SAM with sparse and dense visual prompting for automated segmentation of mobility infrastructure},
  author={Sultan, Rafi Ibn and Li, Chengyin and Zhu, Hui and Khanduri, Prashant and Brocanelli, Marco and Zhu, Dongxiao},
  journal={arXiv preprint arXiv:2311.11319},
  year={2023}
}
