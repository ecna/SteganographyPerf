### Scientific Summary of the Paper

**Title**: Performance of Image Steganography Detection Based on Deep Learning: Feasibility for Real-Time Use

**Authors**: Richard Holleman, Alcuin van Zijl

**Institution**: The Hague University of Applied Sciences

**Summary**:
This paper investigates the feasibility of using deep learning algorithms for the detection of steganography in images in a real-time environment, specifically for use in web browsers. Steganography is the technique of hiding information within digital images, often using methods such as manipulating the least significant bits (LSB) of pixels. Its counterpart, steganalysis, focuses on detecting this hidden information.

The research focuses on the performance of deep learning algorithms in identifying hidden information in images, evaluating both the accuracy and speed of detection. Various steganographic techniques are discussed, including LSB manipulation, the use of frequency domain transformations like the Discrete Cosine Transform (DCT), and adaptive steganography that utilizes statistical characteristics of images.

The experimental setup includes two test machines with different hardware configurations and a range of image sizes from 100 KB to 30 MB. The performance of a convolutional neural network (CNN) model, trained using the PyTorch library, is measured on both CPU and GPU-based systems. Results show that, once loaded, the detection algorithms can analyze images within milliseconds, which is acceptable for real-time use in browsers.

Key findings are that the image size does not significantly affect performance and that the operating system can influence the initial load time of the model. The paper concludes that deep learning-based steganography detection is feasible for real-time applications, provided the initial load time of the model is optimized.

**Keywords**: Image Steganography, Deep Learning, Real-Time Detection, Convolutional Neural Network, PyTorch.
