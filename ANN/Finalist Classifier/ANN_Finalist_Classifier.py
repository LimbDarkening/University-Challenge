from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm



Data = pd.read_csv('ANN_Input.txt', index_col=0)
X_train, X_test, y_train, y_test = train_test_split(Data[['FRS','His']],
                                                    Data['F'],
                                                    test_size=0.33,
                                                    random_state=42)

clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                    hidden_layer_sizes=(5,4,3), random_state=56)
clf.fit(X_train, y_train)
   
scores = cross_val_score(clf, Data[['FRS','His']], Data['F'], cv=5)
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))

#Plot the space
points = 200
X = np.linspace(.2, 1.5, points)
Y = np.linspace(.1, 2.5, points)

dat = np.zeros((len(Y),len(X)))

for iy,y in enumerate(Y):#populating data
    for ix,x in enumerate(X):
        inp = np.array([x,y]).reshape(1,-1)
        dat[(iy,ix)] = clf.predict(inp)
        
plt.imshow(dat, extent=(.2,1.5,.1,2.5),origin='lower',cmap=matplotlib.cm.plasma)
plt.show()
