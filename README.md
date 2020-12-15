# super-model
Model-based reinforcement learning experiments


# Installation
It is recommended that you use a virtual environment for this repository. You may find that you need a version of Pytorch that is specific to your hardware if you want to use GPU acceleration. Otherwise, this should install the CPU version by default.

To run the experiments, you will need to install all the dependencies in the requirements.txt file first.

`cd super-model`

`pip install -r requirements.txt`

Then you'll want to install the package itself. It is recommended that you use the -e option in case you want to modify some of the files yourself.

`pip install -e .`

in the same directory as setup.py. This will make the files in super-model/super_model available to you even in a Jupyter notebook. You'll probably want to use one to train the kinematics model since working with the model and saving it becomes much easier.

# Run the experiments
Make sure you're in the super-model directory.

`cd super-model`

As requested in the submission guidelines, this script will generate all the plots. IT WILL TAKE A LONG TIME. If you don't have a good GPU, this script will take forever to train the ensemble of models. I could not include the trained models with the repository for space reasons, so the only option is to run the training here.

`python experiments\main_results.py`

Otherwise, you can run this function which doesn't take forever. The model doesn't actually require any inverse kinematics since the configuration space and the workspace are one-to-one but it does run the same algorithm. The model just learns to return the output, but it demonstrates the use of the automatic differentiation and backpropagation algorithms for setting up the optimal control problem.

`python experiments\point_mass\point_mass_learn_kinematics.py`

If you want to train the ensemble of models for the two-link manipulator I suggest using Google Colab to run the .ipynb notebook file included in this repository. Make sure you set your runtime to a GPU.
