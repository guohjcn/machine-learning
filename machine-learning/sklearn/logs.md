## Results

```bash
guohj@guo-ubuntu:~/github/machine-learning/machine-learning$ cd /home/guohj/github/machine-learning/machine-learning ; env "PYTHONIOENCODING=UTF-8" "PYTHONUNBUFFERED=1" /usr/bin/python3 /home/guohj/.vscode/extensions/ms-python.python-2018.4.0/pythonFiles/PythonTools/visualstudio_py_launcher.py /home/guohj/github/machine-learning/machine-learning 34915 34806ad9-833a-4524-8cd6-18ca4aa74f14 RedirectOutput,RedirectOutput /home/guohj/github/machine-learning/machine-learning/sklearn/sk-train-test.py
Python version: 3.5.2 (default, Nov 12 2018, 13:43:14)
[GCC 5.4.0 20160609]
reading training and testing data...
******************** Data Info *********************
#training data: 50000, #testing_data: 10000, dimension: 784
******************* NB ********************
training took 0.160418s!
predicting took 0.053717s!
accuracy: 83.69%measure score took 0.012945s!
******************* KNN ********************
training took 10.595217s!
predicting took 678.365815s!
accuracy: 96.64%
measure score took 0.001955s!
******************* LR ********************
/home/guohj/.local/lib/python3.5/site-packages/sklearn/linear_model/logistic.py:433: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
  FutureWarning)
/home/guohj/.local/lib/python3.5/site-packages/sklearn/linear_model/logistic.py:460: FutureWarning: Default multi_class will bechanged to 'auto' in 0.22. Specify the multi_class option to silence this warning.
  "this warning.", FutureWarning)
training took 53.747898s!
predicting took 0.047168s!
accuracy: 92.00%
measure score took 0.004298s!
******************* RF ********************
training took 2.485727s!
predicting took 0.039818s!
accuracy: 93.77%
measure score took 0.001802s!
******************* DT ********************
training took 14.305692s!
predicting took 0.007307s!
accuracy: 87.22%
measure score took 0.002824s!
******************* SVM ********************
/home/guohj/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
training took 3035.487050s!
predicting took 188.868126s!
accuracy: 94.35%
measure score took 0.002105s!
******************* GBDT ********************
training took 3579.662732s!
predicting took 0.933909s!
accuracy: 96.18%
measure score took 0.001985s!

```
