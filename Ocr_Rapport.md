What you built and why
We built a document scanning and text extraction pipeline that detects the document inside an image, corrects its perspective, enhances it, and then extracts the text using two OCR approaches:
•	a traditional OCR approach based on computer vision preprocessing with Tesseract
•	a deep learning OCR approach using EasyOCR
The reason for building this system was to study how different OCR approaches perform on document images and to understand the full pipeline required before text recognition. In real images, text extraction is not only about applying OCR directly. The image often needs several preprocessing steps such as document detection, noise reduction, perspective correction, and binarization to improve readability.
This project also aims to compare the strengths and weaknesses of:
•	a traditional computer vision + OCR pipeline
•	a deep learning-based OCR pipeline
through both visual results and quantitative metrics such as accuracy, Character Error Rate (CER), and processing time.
How the traditional approach works
Diagram :
Input Image
   ↓
Grayscale
   ↓
Denoising
   ↓
Edge Detection
   ↓
Contour Detection
   ↓
Perspective Transform
   ↓
Thresholding
   ↓
Tesseract OCR
   ↓
Extracted Text
How the Deep Learning Approach Works
The deep learning OCR approach uses neural networks to automatically detect and recognize text from images without requiring complex preprocessing.
In this project, we used EasyOCR, which performs two main tasks:
•	Text detection: locating text regions in the image
•	Text recognition: reading the characters inside those regions
The system uses:
•	CNN to extract visual features from text
•	LSTM to understand character sequences
•	CTC to generate the final readable text
Side-by-Side Comparison (Visual + Metrics)
Visual Comparison
The following results show the difference between the two approaches on the same input image:
•	The traditional approach produces cleaner text when the image is well processed and noise is reduced.
•	The deep learning approach is more robust to noise and complex backgrounds but may introduce more recognition errors.
[Tesseract]
Predicted Text: contours of training data, and reconstruct spine contours via latent contour coefficients. To suppress inaccurate predictions and mask connectivity, we integrate latent contour coefficient regression with anchor box classification to obtain spinal segment contours, in which sparse and dense assignments are integrated to more precisely detect spinal segments. Privacy-Preserving Scoliosis Generation. We explore the latent diffusion model (LDM) [12] for scoliosis image self-generation. Specifically, we train LDM on our private dataset Spinal2023, utilizing a diffusion process to map raw images from pixel to latent space. During inference, the trained LDM with text-conditioned prompts can generate X-ray images. To mitigate privacy leaks [13], we employ data augmentation during LDM training, promote data balance, and introduce a privacy review [14] to exclude potentially privacy-compromising samples based on structural and pixel simi- larities, coupled with manual verification. Cost-Effective Semi-Supervised Annotation. Inspired from semi-supervised pseudo- T labeling [15] and multi-stage interactive annotation [16], we propose a new data engine, which drives an iterative pipeline with automatic annotation and selection. It consists of four stages: pseudo-labeling, auto-annotation, manual-assisted annotation, and pri- vacy review. At each iteration, spine contour detection network is first re-trained using current dataset, and its trained model is used for pseudo-labeling so as (0 further update the dataset. Thls iterative pipeline culminates in our open-source clean dataset named Spmal-AIz02,4, which is the largest released scoliosis X-ray dataset to our knowledge. e main contributions of this paper are threefold: e se a novel low-rank approximation parameterized spine representation T curvature estimation, in which latent contour coefficient regression an nchor box classifica with sparse and dense assignments are combined semi-supervised labeling and privacy review er, x‘&ﬁf@?@@ . the generation of large-scale clean scoliosis g ¢ 'S inal- ut privacy leaks. To our knowledge, our open-source P
Word Accuracy: 0.4952
CER: 0.0975
Time (sec): 3.6748
[DL Model – EasyOCR]
Predicted Text: ن .نن { contours of training data, and reconstruct spine contours via latent contour coefficients predictions and mask connectivity, we integrate latent contour To suppress inaccurate coefficient regression with anchor box classification to obtain spinal segment contours, in which sparse and dense assignments are integrated to more precisely detect spinal segments. the latent diffusion model explore We Privacy Preserving Scoliosis Generation LDM) [I2] for scoliosis image self-generation Specifically we train LDM on our dataset Spinal2O23, utilizing a diffusion process to map raw images from pixel private to latent space During inference, the trained LDM with text conditioned prompts can generate X ray images To mitigate privacy leaks [13], we employ data augmentation during LDM training, promote data balance, and introduce a privacy review [14] to exclude potentially privacy compromising samples based on structural and pixel simi- larities, coupled with manual verification pseudo- Cost Efective Semi Supervised Annototion Inspired from semi supervised _ labeling [lS] and multi stage interactive annotation [l6], we propose a new data engine, pipeline with automatic annotation and selection It consists which drives an iterative pri- of four stages: pseudo labeling, auto-annotation, manual assisted annotation, and vacy review At each iteration spine contour detection network is first re trained using pseudo labeling so as to further update current dataset, and its trained model is used for the dataset. This iterative pipeline culminates in our open source clean dataset named Spinal AI2O24, which is the largest released scoliosis X ray dataset to our knowledge The main contributions of this paper are threefold: approximation parameterized spine representation We propose a novel low rank and regression for curvature angle estimation, in which latent contour coefficient and dense assignments are combined to anchor box Classification with sparse detect inegular contours of spinal segments. labeling and privacy review semi supervised We propose a new data engine with large scale clean scoliosis 0f generation in an iterative manner which enables the images without privacy leaks To our knowledge, our open source Spinal X-ay AI2O24 is the largest released scoliosis X ray dataset 1-
Word Accuracy: 0.0286
CER: 0.2516
Time (sec): 35.8732
------------------------------------------------------------
FINAL AVERAGES
============================================================
Tesseract Avg Word Accuracy: 0.4952
Tesseract Avg CER: 0.0975
Tesseract Avg Time: 3.6748
EasyOCR Avg Word Accuracy: 0.0286
EasyOCR Avg CER: 0.2516
EasyOCR Avg Time: 35.8732
The results show a significant difference between the two approaches.
The traditional OCR using Tesseract achieved much higher word accuracy (0.4952) compared to the deep learning approach using EasyOCR (0.0286). Additionally, Tesseract produced a lower Character Error Rate (CER), indicating more accurate character recognition.
In terms of performance, Tesseract was significantly faster, taking only 3.67 seconds on average, while EasyOCR required 35.87 seconds.
This difference can be explained by the nature of the input data. The document images used in this project were relatively clean, well-structured, and close to standard printed text. Such conditions are ideal for traditional OCR systems like Tesseract, which rely on clear character shapes and high contrast.
On the other hand, deep learning models like EasyOCR are designed to handle complex and noisy real-world images. However, when applied to clean documents without proper tuning, they may not perform optimally. In addition, the lack of model adaptation or parameter tuning can negatively impact accuracy.
What I Learned Through this project:
I learned that the performance of an OCR system depends heavily on the quality of image preprocessing. Choosing the right filters and tuning parameters such as denoising and thresholding has a significant impact on the final text recognition accuracy.  I also learned that selecting the appropriate approach depends on the type of data. Traditional OCR methods perform well on clean and structured documents, while deep learning models are more suitable for complex and noisy images.  Additionally, I understood that combining different techniques can improve results. For example, preprocessing using computer vision followed by OCR can significantly enhance performance.
What I Would Improve With more time :
I would improve the system by:  Testing different preprocessing techniques and optimizing parameters to achieve better image quality  Fine-tuning deep learning models like EasyOCR to better match the dataset  Combining multiple OCR approaches and selecting the best output automatically  Exploring the use of large language models (LLMs) to correct OCR errors, especially for handwritten or low-quality text  This project showed that building an effective OCR system is not only about choosing a model, but also about understanding the data and designing the right processing pipeline.
A hybrid system combining traditional preprocessing, deep learning OCR, and LLM-based correction could achieve significantly better results.

