# Diagnose and Explain
This repository contains code for the paper: **"Diagnose and Explain: A Multi-Level Attention Model for Automated Radiology Reporting from Chest X-Ray"** (will update with link when a preprint is available)

Our model takes a Chest X-ray image as input and generates a complete radiology report using Natural Langugage. The generated report contains 2 sections:
* **Findings:** observations regarding each part of the chest examined. Generally a paragraph with 6+ sentences.
* **Impression:** generally a one-sentence diagnostic based on findings reported. Can contain multiple sentences.

## Sample
### Ground truth:
<image align='left' src="samples/image_3707.png"  width="150px" height="180px"/>

---

**Findings:** the cardiomediastinal silhouette is within normal limits for size and contour. The lungs are normally inflated without evidence of focal airspace disease, pleural effusion or pneumothorax. Stable calcified granuloma within the right upper lung. No acute bone abnormality.

**Impression:** no acute cardiopulmonary process.

---
### Model output:
<image align='left' src="samples/image_3707.png" width="150px" height="160px"/>

---

**Findings:** heart size and mediastinal contours appear within normal limits. No focal airspace disease. No pleural effusion or pneumothorax is seen.

**Impression:** no acute osseus abnormality.

---
### Visual Attention Plot
Findings (left), Impression (right).
<p>
	<image align='left' src="samples/findings_attention_plot_3707.png" width="425px"/>    
	<image align='right' src="samples/impression_attention_plot_3707.png" width="400px"/>
</p>
<br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/><br/>

## Dataset
We trained our model on the Indiana University [Chest X-Ray collection](https://openi.nlm.nih.gov/faq.php). The dataset
comes with **3955** chest radiology reports from various hospital systems and **7470** associated chest x-rays 
(most reports are associated with 2 or more images representing frontal and lateral views).

## Model architecture
Our model uses a CNN-LSTM to generate words. Features extracted from a CNN are encoded and used by an hierarchical RNN to generate paragraphs (findings) sentence by sentence. We use an attention mechanism to extract visual and semantic features at many levels of the word and sentence generation to guide the word decoder.

More details on our model architecture and proposed approach will be present in the soon-to-be-released preprint of our paper.

## Training on Cloud TPU
Our code was designed for training on Google Cloud TPU.

* Head over to TensorFlow's quickstart on [setting up a TPU instance](https://cloud.google.com/tpu/docs/quickstart) to get started with running models on Cloud TPU.
* Clone this repository and `cd` into directory 
  ```
	git clone https://github.com/wisdal/diagnose-and-explain && cd diagnose-and-explain
  ```
* Start training
  ```
    export STORAGE_BUCKET=<Your Storage Bucket>
    python main.py --tpu=$TPU_NAME --model_dir=${STORAGE_BUCKET}/tpu --train_steps=20 --iterations_per_loop=100 --batch_size=512
  ```
  
## Acknowledgements
Parts of these experiments were possible thanks to the TensorFlow Research Cloud Program which offers free TPUs for training and running models on Google Cloud for a limited period of time.
