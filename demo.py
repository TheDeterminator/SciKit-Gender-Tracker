#remember to install scikit with pip install -U scikit-learn
from sklearn import tree

classifier = tree.DecisionTreeClassifier()

# List of person measurements representing height(cm), shoulder-width(cm) and shoe size(EU
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
     'female', 'male', 'male']

#TODO Modify function so that it allows data entry with any measurement system 
#TODO Update function so that it takes in other pieces of data in order to clasify

classifierf = classifier.fit(X,Y)

prediction = classifier.predict([[190, 70, 43]])

print(prediction)
