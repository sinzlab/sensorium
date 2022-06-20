# Dataset Structure

Below we provide a brief explanation of the dataset structure and how to access all the information contained in them.

Have a look at our white paper for in depth description of the data. [White paper on arXiv](https://arxiv.org/abs/2206.08666)

We provide the datasets in the .zip format. Unzipping them will create two folders **data** and **meta**.

- **data:** includes the variables that were recorded during the experiment. The experimental variables are saved as a collection of numpy arrays. Each numpy array contains the value of that variable at a specific image presentation (i.e. trial). Note that the name of the files does not contain any information about the order or time at which the trials took place in experimental time. They are randomly ordered.
  - **images:** This directory contains NumPy arrays where each single `X.npy` contains the image that was shown to the mouse in trial `X`.
  - **responses:** This directory contains NumPy arrays where each single `X.npy` contains the deconvolved calcium traces (i.e. responses) recorded from the mouse in trial `X` in response to the particular presented image.
  - **behavior:** Behavioral variables include pupil dilation, the derivative of the pupil size, and running speed. The directory contain NumPy arrays (of size `1 x 3`) where each single `X.npy` contains the behavioral variables (in the same order that was mentioned earlier) for trial `X`.
  - **pupil_center:** the eye position of the mouse, estimated as the center of the pupil. The directory contain NumPy arrays (of size `1 x 2`) for horizontal and vertical eye positions.
- **meta:** includes meta data of the experiment
    - **neurons:** This directory contains neuron-specific information. Below are a list of important variables in this directory
        - `area.npy`: contains the area of each neuron
        - `cell_motor_coordinates.npy`: contains the position (x, y, z) of each neuron in the cortex, given in microns.
        - `layer.npy`: contains the cortex layer to which neuron belongs to
        - `unit_ids.npy`: contains a unique id for each neuron
    - **statistics:** This directory contains statistics (i.e. mean, median, etc.) of the experimental variables (i.e. behavior, images, pupil_center, and responses).
      - **Note:** The statistics of the responses are or particular importance, because we provide the deconvolved calcium traces here in the responses.
      
        However, for the evaluation of submissions in the competition, we require the responses to be **standardized** (i.e. `r = r/(std_r)`), with the `std` computed across all images on the training set.
        
        For more information, please refer to the [**Submission Section**](../submission_tutorial/)
    - **trials:** This directory contains trial-specific meta data. 
        They contain single 1-d NumPy arrays for each trial variable. 
        
        How to relate these meta data to the neuronal data (images, responses, ...)?
        
        The indices of these arrays correspond to the `.npy` files in **data**. For example:
        ``` 
      # get meta data array
      image_ids = np.load('./meta/trials/frame_image_id.npy')
      
      # relate meta data with neuronal data
      trial_image_id = image_ids[0]
      corresponding_image = np.load('./data/images/0.npy')
      corresponding_neuronal_response = np.load('./data/responses/0.npy')
        ```

        Below are a list of important variables in this directory.
        - `frame_image_id.npy`: contains unique image id. If the image is presented multiple times (which is the case in the test set) this image ID will be present multiple times.
        - `tiers.npy`: contains labels that are used to split the data into *train*, *validation*, and *test* set
          - The *training* and *validation* split is only present for convenience, and is used by our ready-to-use PyTorch DataLoaders.
          - The *test* set is used to evaluate the model preformance. In the competition datasets, the responses to all *test* images is withheld.
          - In the 2 competition datasets, there is the additional tier *final_test*, which contains 100 images and their repetitions. The model performance on these *tiers* will be used to determine the winner of the competition. 

        - `trial_idx.npy`: contains a unique index for each trial. While the true trial index is available for the “pre-training” datasets, it is hidden (i.e. hashed) in the competition datasets. 
          - The *trial_idx* corresponds to the actual order of image presentations to the mouse. We hide the *trial_idx* in the competition scans (i.e. by hashing them).


# Competition Datasets (Sensorium & Sensorium+)

The datasets `26872-17-20` (Sensorium) `27204-5-13` (Sensorium+) are different from the 5 other full datasets in these ways:

- They have 2 types of test images, that reflects how we evaluate the submissions:
  - **live test images**: These are 100 images, and can be found under the tiers *test*. They are also present in all the pre-training datasets
  - **final test images**: Another 100 images, that are not present in the other datasets. The tier for these images is * live_test* 
- The responses to all *test* or *final_test* images is withheld (the response arrays are present, but zeroed out)
- Information about order of trials is withheld (i.e. the *trial_idx* is hashed)
- For **Sensorium**, the behavioral variables and eye position are withheld (arrays are present, but zeroed out)
