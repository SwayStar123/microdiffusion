uses sdxl_vae from https://huggingface.co/madebyollin/sdxl-vae-fp16-fix/tree/main

Paper link: https://arxiv.org/pdf/2407.15811

20 epochs of training on a 1b param model (commoon canvas cc-by dataset)
![image](https://github.com/user-attachments/assets/4c25004e-f8a1-4980-b096-6e9852ae7d70)

You can download weights here: https://huggingface.co/SwayStar123/MicroDiT/blob/main/no_cfg/microdit_model_epoch_19.pt
and put them in models folder, and use test_model.ipynb to inference

50 epochs of training on a 150m param model (CelebA dataset)
![image](https://github.com/user-attachments/assets/8abef7d3-71df-4ba0-9e07-d8faa0360159)

Credit to original authors: Vikash Sehwag, Xianghao Kong, Jingtao Li, Michael Spranger, Lingjuan Lyu

Thank you to Baber from discord for saving me hours of my time
