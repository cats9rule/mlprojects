####### OVO SE NE IZVRŠAVA NITI TREBA DA SE POZIVA ALI MI JE BILO ŽAO DA GA SKROZ POBRIŠEM


# Code koji učitava datatraining i onda sa datatest2 trenira linear svc lošooo
dataTraining = pd.read_csv("occupancy_data/datatraining.txt", sep=',')
dataTraining['date'] = [pd.to_datetime(dataTraining, format='%Y-%m-%d %H:%M:%S') for dataTraining in dataTraining['date']]
dataTraining['date'] = pd.DatetimeIndex(dataTraining.date).asi8


cols = [col for col in dataTraining.columns if col not in ['Occupancy']]
dataTraining_features = dataTraining[cols]
targetTraining = dataTraining['Occupancy']



dataTest_features = data[cols]
targetTest= data['Occupancy']

print(len(dataTraining_features))
print(len(dataTest_features))

scaler = preprocessing.MinMaxScaler()

dataTraining_features = scaler.fit_transform(dataTraining_features)
dataTest_features = scaler.transform(dataTest_features)

pred = svc_model.fit(dataTraining_features, targetTraining).predict(dataTest_features)

print("LinearSVC accuracy : ",accuracy_score(targetTest, pred, normalize = True))
report = classification_report(targetTest,pred)
with io.open('evaluation/classification_report_scaler_new.txt','w',encoding='utf-8') as f: f.write(report)
print(confusion_matrix(targetTest, pred))


# deo koda koji iscrtava cfm za ovo prethodno
result = confusion_matrix(targetTest, pred, normalize='pred')

classes = ["0", "1"]
df_cfm = pd.DataFrame(result, index = classes, columns = classes)
plt.figure(figsize = (10,7))
cfm_plot = sns.heatmap(df_cfm, annot=True)
cfm_plot.figure.savefig("evaluation\cfm.png")




# deo koda koji je predstavljao spajanje vise datasetova u dataConcatinated
dataTest = pd.read_csv("occupancy_data/datatest.txt", sep=',')
dataTest['date'] = [pd.to_datetime(dataTest, format='%Y-%m-%d %H:%M:%S') for dataTest in dataTest['date']]
dataTest['date'] = pd.DatetimeIndex(dataTest.date).asi8

dataTraining = pd.read_csv("occupancy_data/datatraining.txt", sep=',')
dataTraining['date'] = [pd.to_datetime(dataTraining, format='%Y-%m-%d %H:%M:%S') for dataTraining in dataTraining['date']]
dataTraining['date'] = pd.DatetimeIndex(dataTraining.date).asi8

dataConcatinated = pd.concat([data,dataTest,dataTraining],axis=0)
print(f"Broj uzoraka u kompletnom skupu podataka: {len(dataConcatinated)}")

cols = [col for col in dataConcatinated.columns if col not in ['Occupancy']]
dataConcatinated_features = dataConcatinated[cols]
targetConcatinated = dataConcatinated['Occupancy']

data_train,data_test,target_train,target_test = train_test_split(dataConcatinated_features,targetConcatinated,test_size=0.2,random_state=10)

print(f"Broj uzoraka u trening skupu podataka: {len(data_train)}")
print(f"Broj uzoraka u test skupu podataka: {len(data_test)}")


scaler = preprocessing.MinMaxScaler()

data_train = scaler.fit_transform(data_train)
data_test = scaler.transform(data_test)

pred = svc_model.fit(data_train, target_train).predict(data_test)

print("LinearSVC accuracy : ",accuracy_score(target_test, pred, normalize = True))
report = classification_report(target_test,pred)
with io.open('evaluation/classification_report_scaler_concatinated.txt','w',encoding='utf-8') as f: f.write(report)
result = confusion_matrix(target_test, pred)
print(result)





##PRINT TREE KOD RANDOM TREE

dot_data = tree.export_graphviz(rf_random.best_estimator_[0],
 out_file=None,
feature_names=data_features.columns,
 class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())



randomForest = RandomForestClassifier(random_state=0, criterion="entropy") 
# kod decision tree smo koristili gini criterion, možda bi mogli da probamo gini i ovde?

randomForestModel = randomForest.fit(data_train, target_train)

dot_data = tree.export_graphviz(randomForestModel.estimators_[5],
 out_file=None,
feature_names=data_features.columns,
 class_names=['0','1'])

graph = pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())



## SVC NA KRAJU


svc_model = SVC()
hyperparameters = {'C': np.logspace(0,10,10), 
              'gamma': np.logspace(0,10,10,base=0.1),
              'kernel': ['rbf']} 
rnd_search_cv = GridSearchCV(svc_model,hyperparameters, cv=3, verbose=3,
 n_jobs=-1)
best_SVC_model = rnd_search_cv.fit(data_train,target_train)
print(best_SVC_model.best_params_)
best_SVC_pred = best_SVC_model.predict(data_test)

print("Best SVC accuracy : ",accuracy_score(target_test, best_SVC_pred, normalize = True))
print(confusion_matrix(target_test, best_SVC_pred))




#### CROSS VALIDATION SCORE ####
from sklearn.model_selection import cross_val_score,RepeatedStratifiedKFold

svc_model =  SVC(C=166.81005372000593,gamma=1)
model = svc_model.fit(data_train,target_train)
cv = RepeatedStratifiedKFold(n_splits=10,n_repeats=3,random_state=1)

score = cross_val_score(model,data_train,target_train,scoring='accuracy',cv=cv,n_jobs=-1)
print('Accuracy: %.10f (%.10f)' % (np.mean(score), np.std(score)))
score




for clf, label in zip([svc, decisionTree, randomForest,knn,naive_bayes, voting_classifier], ['SVC','Decision Tree', 'Random Forest','KNeighbors' ,'Naive Bayes', 'Ensemble']):
    scores = cross_val_score(clf, data_train, target_train, scoring='accuracy', cv=5)
    print("Accuracy: %0.10f (+/- %0.10f) [%s]" % (scores.mean(), scores.std(), label))


## TODO: DT model je los, promenili smo u pipe/pca i taj se koristi sve ispod
## dt sa balacing najbolji broj features je 7, bez je 4