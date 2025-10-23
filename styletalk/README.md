### What this folder currently has
1. **setup_environment.sh**: this script sets up the environment by doing the following:
    - Getting the dependencies using apt-get and pip
    - Fetching the necessary git repos (Deep3DFaceRecon and styletalk) and their dependencies
        - Here the user has to specify their username and password to download the basel face model (you can request access at https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads)
    - Moving some scripts into the Deep3DFaceRecon repo
2. **Scripts for extracting keypoints and the 3dmm parameters from a video**
    - These are originally from the PIRenderer repo, but heavily modified and should work for our purpose after being moved to the Deep3DFaceRecon repo by the setup script
3. **tools.py**: Has all necessary functions to extract phonemes from audio and align them to the frames of the video, along with a function to call the above scripts to process a video file
4. **styletalk.py**: in theory, this should run the whole pipeline

### Current issues:
- There is a file in Deep3DFaceRecon that needs a couple of fixes before it works. The following lines currenty have to be changed for the scripts to work:
    - util/preprocess.py:19 - np.VisibleDeprecationWarning -> DeprecationWarning
    - util/preprocess.py:202 - trans_params = np.array([w0, h0, s, t[0], t[1]]) -> trans_params = np.array([w0, h0, s, t[0][0], t[1][0]]) 
