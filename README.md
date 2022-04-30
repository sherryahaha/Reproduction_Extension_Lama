# Reproduction_Extension_Lama
Dsa project of the reproduction and extension of Lama.

## The original paper and project
[[Project page](https://saic-mdal.github.io/lama-project/)] [[GitHub](https://github.com/saic-mdal/lama)] [[arXiv](https://arxiv.org/abs/2109.07161)] [[Supplementary](https://ashukha.com/projects/lama_21/lama_supmat_2021.pdf)] [[BibTeX](https://senya-ashukha.github.io/projects/lama_21/paper.txt)]

## The extension work
- Use Mask RCNN model with a dilated transformation to generate masks which can remove the specified category(eg.person) and then use Lama for inpainting.
- We provide several hyperparameters to help generate a better mask. 
  - category: specify the category we want to remove. (eg. person)
  - detect_model: the detection method which will be used to identify the object location. Possible values: segmentation or object_detect
  - detect_threthold: the detect_threthold in the Mask RCNN model.

## Run the code
Run the Reproduction_Extension_Code_for_LaMa_inpainting.ipynb code. You need to do some changes as the instruction of CodeRunning.txt.
