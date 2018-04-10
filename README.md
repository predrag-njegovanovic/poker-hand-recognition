# PokerHandRecognition

 _Description of project_ 
 
 Recognition of playing cards and classification of poker hands. 

 **Libraries** necessary to run scripts:
    
    -python 2.7
    -opencv 3.2
    -numpy 1.12 or greater
    -sckit-learn 0.18
    -sckit-image 0.10
    -h5py
    -keras
    -Theano or Tensorflow

  All libraries can be installed via pip. Run command pip install <_library_name_>.

  Run main.py -> Load models and recognize all cards from folder 'data/test_dataset/' and write result in /results/results.txt. If models aren't saved run train_nn.py first.
  
  card_manipulation.py -> Script contains functions for image processing
  
  crop_cards.py and preprocess_crop_cards.py -> Create dataset to run through neural network
  
  train_nn.py -> Create models for NN and Random Forest and save to file
  
  


