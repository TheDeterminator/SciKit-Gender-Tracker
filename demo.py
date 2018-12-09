#remember to install scikit with pip install -U scikit-learn
from sklearn import tree, ensemble, discriminant_analysis

tree_classifier = tree.DecisionTreeClassifier()
ada_classifier = ensemble.AdaBoostClassifier()
qda_classifier = discriminant_analysis.QuadraticDiscriminantAnalysis()

# List of person measurements representing height(cm), shoulder-width(cm) and shoe size(EU
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#TODO Modify function so that it allows data entry with any measurement system 
#TODO Update function so that it takes in other pieces of data in order to clasify
#TODO Update function so that it uses other classifiers (aside from the decision tree) to classify this data 

tree_classifier = tree_classifier.fit(X,Y)
tree_prediction = tree_classifier.predict([[190, 70, 43]])
tree_prediction2 = tree_classifier.predict([[164, 65, 41]]) 

ada_classifier = ada_classifier.fit(X,Y)
ada_prediction = ada_classifier.predict([[190, 70, 43]])
ada_prediction2 = ada_classifier.predict([[164, 65, 41]]) 

qda_classifier = qda_classifier.fit(X,Y)
qda_prediction = qda_classifier.predict([[190, 70, 43]])
qda_prediction2 = qda_classifier.predict([[164, 65, 41]]) 

print(f'On measurements of [190, 70, 43] Tree predicts... {tree_prediction}, Ada predicts... {ada_prediction}, Quadratic Discriminant Analysis predicts...{qda_prediction}')

print(f'On more difficult measurements of [164, 65, 41] Tree predicts... {tree_prediction2}, Ada predicts... {ada_prediction2}, Quadratic Discriminant Analysis predicts...{qda_prediction2}')
