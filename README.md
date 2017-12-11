<h1> Hello! </h1>

Hello, my name is Mirko and I am a phycisist and machine learning enthusiast. In this repository you can find a variety of machine learning/optimization techniques applied to a set of toy problems/datasets. I decided to write notebooks for each of these topics where I usually give an introduction, some math background and then I present an implementation of the algorithm under discussion in python. The aim of these notebooks is to keep track of what I learn and maybe (hopefully) help people who, like me, want to know more about machine learning.

<h1> LIST OF FILES AND FOLDERS </h1>
<ol>
<li>DATASETS: containing the datasets used in the examples
<ul>
<li><b>weight_height.csv</b> (used in the Multiple Linear Regression notebook)</li>
<li><b>faithful.csv</b> (used in the Kmeans clustering ang Gaussian Mixture notebooks)</li>
<li><b>train-images.idx3-ubyte</b> (used in the BMM .cpp code, binary dataset of handwritten digits)</li>
<li><b>italy_cities.csv</b> (Traveling Salesman problem data set, used in the Simulated Annealing and Genetic Algorithm notebooks)</li>
<li><b>iris_dataset.csv</b> (Used in the NNET benchmark notebook)</li>
</ul>
</li>
<li>NOTEBOOKS: containing jupyter notebooks with theory and python codes
<ul>
<li><b>MLR.ipynb</b> (Multiple Linear Regression)</li>
<li><b>KMEANS.ipynb</b> (Kmeans clustering)</li>
<li><b>LDA.ipynb</b> (Latent Dirichlet Allocation for topics/text analysis)</li>
<li><b>GMM.ipynb</b> (Gaussian Mixture Models for clustering)</li>
<li><b>BMM.ipynb</b> (Bernoulli Mixture Models for recognition of handwritten digits)</li>
<li><b>SA.ipynb</b> (Simulated Annealing for the solution of the Traveling Salesman problem)</li>
<li><b>GA.ipynb</b> (Genetic Algorithm for the solution of the Traveling Salesman problem)</li>
<li><b>NNET.ipynb</b> (Single layer neural network implementation, with details on back propagation)</li>
<li><b>NNET_Keras_vs_Scratch.ipynb</b> (Benchmark of Keras single layer neural network with neural network implemented from scratch. The benchmark is done on the iris data set)</li>
<li><b>MNNET_CPP.ipynb</b> (Benchmark of Keras multilayer neural network with neural network implemented from scratch in C++. The benchmark is done on the iris data set)</li>
</ul>
</li>
<li>CPP: .cpp version of some of the implementations in the notebooks
<ul>
<li>BMM: Bernoulli Mixture Models (cpp version of the BMM.ipynb for handwritten digits recognition)</li>
<li>MNNET: Multi-layer neural network. Can be compiled as a stand-alone exe or can be used to build a library for python, see the MNNET_CPP.ipynb notebook for details on how to use it as a library in python (work in progress to include more features in it, e.g. automatic differentiation for custom loss and activation functions)</li>
</ul>
</li>
<li>IMAGES: some images and data files used in the notebooks</li>
</ol>
