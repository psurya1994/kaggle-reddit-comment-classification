# COMP 551 project 2: reddit comments classification

Here are the most important parts of our code for running experiments. 

- `p2*.ipynb` contains the code for running various independing sklearn models and deep learning models. `v*.py` are just python versions of these notebooks used to submit as batch jobs on the server.
- The experiments in the stacking various models has been done in `stacking.ipynb`, `stacking2.ipynb`, `stacking3.ipynb`. 


Here's some insights into our thought process on choosing the best models for top performance on kaggle.

- Our intention was to challenge ourselves and win the kaggle competition by a margin. So we spent a lot of time reading about winning solutions on kaggle and talking to some people who had experience working on similar problems before. We understood the best practices and stuck to them closely to good scores. \cite{kag1}, \cite{kag2}, \cite{kag3}, \cite{kag4} are some of the most important resources we found online, and many of the techniques we used were ideas of these places. 
- We tried lots of methods to find which ones gave the best accuracy. The ones we finally narrowed down on were: Bert (0.59), MultinoulliNB (.58), LinearSVC (0.56), XLNet (0.56). The numbers in the brackets are their accuracies. These models, we thought were most diverse and gave the best accuracies and hence need to be combined together. We tried various meta classifiers mentioned in the previous sections but the ones that gave us the best performance were Xgboost and LightGBM. We found it interesting that options suggested online such as RFClassifier and LinearSVC did not gave such good numbers. 
- We saw that both Xgboost and LightGBM were giving very good accuracy numbers and found it hard to decide between them. So we blended both of these predictions using a simple average. We verified that it was a good idea because we saw that the training set accuracy went up (from 0.756 for xgboost, 0.833 for LighGBM to 0.86 for the combination). We submitted this model to get a leaderboard score of 0.635.
- We used the libraries for these meta classifiers in python directly without understanding the internals (we were short on time). We hope to spend more time on understanding these algorithms deeply in the coming days.
\end{itemize}
