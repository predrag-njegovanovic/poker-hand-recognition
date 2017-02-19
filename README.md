# PokerHandRecognition

 _Description of project_ 
 
 Recognition of playing cards and classification of poker hands. 

 **Libraries** necessary to run scripts:
    
    -python 2.7
    -numpy 1.12 or greater
    -sckit-learn 0.18
    -sckit-image 0.10
    -h5py
    -keras
    -Theano or Tensorflow

  All libraries can be installed via pip. Run command pip install <_library_name_>.

  Run MainScript.py -> Load models and recognize all cards from folder 'TestCards' and write result in Results.txt. If models aren't saved run TrainAndSaveNN.py first.
  
  CardManipulationFunctions.py -> Script contains functions for image processing
  
  DuplicateDataset.py -> Create dataset to run through neural network
  
  TrainAndSaveNN.py -> Create models for NN and Random Forest and save to file
  
  


