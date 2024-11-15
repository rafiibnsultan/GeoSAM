# GeoSAM: Fine-tuning SAM with Sparse and Dense Visual Prompting for Automated Segmentation of Mobility Infrastructure

## (Hello, this is currently the old approach we tried. The "updated" approach that can take "text" as prompts in addition to sparse or click prompts can be found in the <a href="https://github.com/rafiibnsultan/GeoSAM/tree/GeoSAM_with_text">GeoSAM_with_text branch</a>)

This repository is dedicated to the work of GeoSAM. Please find the paper here: <a href="https://arxiv.org/abs/2311.11319">Link</a>

If you have any questions, fill out this [form](https://forms.gle/yagVwyw89mUbYb879)  or just email me at hm4013@wayne.edu. I will get back to you as soon as possible.

Also, please find the <a href="https://waynestateprod-my.sharepoint.com/:u:/g/personal/hm4013_wayne_edu/EXvJFrshs9RAm68KYnkKJ7gB4D4gB65CCXmasoDYUIplMw?e=6h7CKx">link</a> for the weights.

See the [demo](https://github.com/rafiibnsultan/GeoSAM/blob/GeoSAM_with_text/demo/) here.

This work has been submitted. Waiting for the decision.


## Abstract:
<p class="justified-text">The Segment Anything Model (SAM) has shown impressive performance when applied to natural image segmentation. However, it struggles with geographical images like aerial and satellite imagery, especially when segmenting mobility infrastructure including roads, sidewalks, and crosswalks. This inferior performance stems from the narrow features of these objects, their textures blending into the surroundings, and interference from objects like trees, buildings, vehicles, and pedestrians - all of which can disorient the model to produce inaccurate segmentation maps. To address these challenges, we propose Geographical SAM (GeoSAM), a novel SAM-based framework that implements a fine-tuning strategy using the dense visual prompt from zero-shot learning, and the sparse visual prompt from a pre-trained CNN segmentation model. The proposed GeoSAM outperforms existing approaches for geographical image segmentation, specifically by 26%, 7%, and 17% for road infrastructure, pedestrian infrastructure, and on average, respectively, representing a momentous leap in leveraging foundation models to segment mobility infrastructure including both road and pedestrian infrastructure in geographical images.</p>

<img src="Pipeline.png" alt="GeoSAM">
## Acknowledgement
We want to thank these two works for their open-source code and contributions to the respective fields!

<a href="https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html">Segment Anything Model (SAM)</a>

<a href="https://proceedings.aesop-planning.eu/index.php/aesopro/article/view/39">MAPPING THE WALK: A SCALABLE COMPUTER VISION APPROACH FOR GENERATING SIDEWALK NETWORK DATASETS FROM AERIAL IMAGERY.</a>



## Citations

If these codes are helpful for your study, please cite:

```bibtex
@article{sultan2023geosam,
  title={GeoSAM: Fine-tuning SAM with sparse and dense visual prompting for automated segmentation of mobility infrastructure},
  author={Sultan, Rafi Ibn and Li, Chengyin and Zhu, Hui and Khanduri, Prashant and Brocanelli, Marco and Zhu, Dongxiao},
  journal={arXiv preprint arXiv:2311.11319},
  year={2023}
}
