# Towards Understanding the Benefits of Neural Network Parameterization in Geophysical Inversions: A Study With Neural Fields (NFs-Inv)

## Summary
In this work, we employ neural fields, which use neural networks to map a coordinate to the corresponding physical property value at that coordinate, in a test-time learning manner. For a test-time learning method, the weights are learned during the inversion, as compared to traditional approaches which require a network to be trained using a training data set. Results for synthetic examples in seismic tomography and direct current resistivity inversions are shown first. We then perform a singular value decomposition analysis on the Jacobian of the weights of the neural network (SVD analysis) for both cases to explore the effects of neural networks on the recovered model. The results show that the test-time learning approach can eliminate unwanted artifacts in the recovered subsurface physical property model caused by the sensitivity of the survey and physics. Therefore, NFs-Inv improves the inversion results compared to the conventional inversion in some cases such as the recovery of the dip angle or the prediction of the boundaries of the main target. In the SVD analysis, we observe similar patterns in the left-singular vectors as were observed in some diffusion models, trained in a supervised manner, for generative tasks in computer vision. This observation provides evidence that there is an implicit bias, which is inherent in neural network structures, that is useful in supervised learning and test-time learning models. This implicit bias has the potential to be useful for recovering models in geophysical inversions.

## Contents
'Seismic' directory:
- Case 1.ipynb: Reproduce the NFs-Inv result for Case 1 (cross-hole seismic tomography inversion with homogenous background).
- Case 2.ipynb: Reproduce the NFs-Inv result for Case 2 (cross-hole seismic tomography inversion with heterogeneous background). 
- Conventional.ipynb: Reproduce the conventional inversion results for cross-hole seismic tomography.
  
All notesbook can be run on Colab (!pip install is included in the first cell).

'DCR' (Direct Current Resistivity) directory:

- Generate_Synthetic_Voltages.ipynb: Runs the forward simulation for the DCR cases shown in the paper. 5% Gaussion noises are added to the synthetic data.
- Conventional_inversion_using_SimPEG_z=225.ipynb: Reproduce the conventional inversion results for DCR cases.
- 'Case_*_truncated_z_225' folders: Contain the voltage measurements and topology information for each DCR trial.
- 'Case_* .py' files: Reproduce the NFs-Inv result for each DCR trial. The final results (both the final predicted conductivity model and the weights) will be stored in the proper format with name defined in the corresponding ' NF_*' folders.

'SVD_analylsis' directory:

- Case_1.ipynb, Case_2.ipynb: Conduct SVD analylsis for Case 1 or 2 for the cases where no position encoding is employed. The results are saved in "no_encoding_USV_Case_*.pkl'.
- Plot_SVD_seismic.ipynb: Reproduce the plots for left-singular vectors and singular values for Case 1 or 2.
- SVD_Case_3.ipynb, SVD_Case_4.ipynb: Conduct SVD analylsis for Case 3 or 4. The results are saved in "no_encoding_USV_Case_*.pkl'. Reproduce the plots for left-singular vectors and singular values for Case 3 or 4.
  
## Usage

Dependencies are specified in [requirements.txt](/requirements.txt)

```
pip install -r requirements.txt
```
You can then clone this repository. From a command line, run

```
git clone https://github.com/anna1963/NFs-Inv.git
```

Then `cd` into the `NFs-Inv` directory:

```
cd NFs-Inv
```

To setup your software environment, we recommend you use the provided conda environment

```
conda env create -f environment.yml
conda activate NFs-Inv-environment
```
## Running the notebooks

For more information on running Colab notebooks, see https://colab.google.

## License
These notebooks are licensed under the [MIT License](/LICENSE).
