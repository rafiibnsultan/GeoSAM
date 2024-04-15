# GeoSAM: Fine-tuning SAM with Multi-Modal Prompts for Mobility Infrastructure Segmentation

Hello,
This is an updated version of GeoSAM, here, we implement a fine-tuning approach with automatically generated multi-modal prompts, specifically, point prompts from a pre-trained, task-specific traditional model, complemented by text prompts provided by users.

Contrasting with the previous version, which you can find in the <a href="https://github.com/rafiibnsultan/GeoSAM/tree/geosam_old">GeoSAM_old branch</a>.

In the previous approach, we were using feature embeddings from a traditional model to create dense prompts which was assisting our sparse or click prompts generation. However, in the new version, we have decided to use the assistance of natural language. So, instead of dense prompts, we use text prompts from natural language to provide SAM with a more natural language context to assist the click prompts. We incorporate a multi-prompts system by using texts as direct prompts for SAM which aids the model by providing more semantic context. We will provide a copy of the updated manuscript whenever it is ready.


Also, please find the <a href="https://waynestateprod-my.sharepoint.com/:u:/g/personal/hm4013_wayne_edu/EXvJFrshs9RAm68KYnkKJ7gB4D4gB65CCXmasoDYUIplMw?e=6h7CKx">link</a> for the weights.




## Abstract:
<p class="justified-text">The Segment Anything Model (SAM) has shown impressive performance when applied to natural image segmentation. However, it struggles with geographical images like aerial and satellite imagery, especially when segmenting mobility infrastructure including roads, sidewalks, and crosswalks. This inferior performance stems from the narrow features of these objects and their textures blending into the surroundings. To address these challenges, we propose Geographical SAM (GeoSAM), a novel SAM-based framework that implements a fine-tuning approach with automatically generated multi-modal prompts, specifically, point prompts from a pre-trained, task-specific traditional model, complemented by text prompts provided by users. GeoSAM uses point prompts to serve as the main guidance for the model, Whereas text prompts act as secondary prompts, providing a semantic understanding of natural language to enhance the model's comprehension abilities. The proposed GeoSAM outperforms existing approaches for geographical image segmentation, specifically by 30%, and 7% for road infrastructure, and pedestrian infrastructure, respectively, representing a momentous leap in leveraging foundation models to segment mobility infrastructure including both road and pedestrian infrastructure in geographical images.</p>

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
