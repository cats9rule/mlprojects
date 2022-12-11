### SVC

#nisam smestio sve u jedan grid jer moj jadan pc ne moze da izdrzi
#{'C': np.logspace(0,4,10), 'kernel': ['linear']} - ne daje bolje rezultate - 98.98
#{'C': np.logspace(0,4,10), 
#              'gamma': np.logspace(0,4,10,base=0.1),
#              'kernel': ['rbf','poly','sigmoid']}
# bolji rezultati - 99.39
svc_model = SVC()
hyperparameters = {'C': np.logspace(0,4,10), 
              'gamma': np.logspace(0,4,10,base=0.1),
             'kernel': ['rbf','poly','sigmoid']}
                 
rnd_search_cv = RandomizedSearchCV(svc_model,hyperparameters,random_state=1,n_iter=100, cv=3, verbose=2,
 n_jobs=-1)
best_SVC_model = rnd_search_cv.fit(data_train,target_train)
print(best_SVC_model.best_params_)
best_SVC_pred = best_SVC_model.predict(data_test)

print("Best SVC accuracy : ",accuracy_score(target_test, best_SVC_pred, normalize = True))
print(confusion_matrix(target_test, best_SVC_pred))
report = classification_report(target_test,predNN)
with io.open('evaluation/classification_report_best_SVC.txt','w',encoding='utf-8') as f: f.write(report)


##Onaj kod koji koristi pipeline
# useless shit ali neka ga za sad

# svc_model = SVC()
# std_slc = preprocessing.StandardScaler()
# pca = decomposition.PCA()
# pipe = Pipeline(steps=[('std_slc', std_slc),
#                            ('pca', pca),
#                            ('svc', svc_model)])

# hyperparameters = {'svc__C': np.logspace(0,4,10), 
#               'svc__gamma': np.logspace(0,4,10,base=0.1),
#               'pca__n_components': list(range(1,data_train.shape[1]+1,1))} 
# rnd_search_cv = GridSearchCV(pipe, hyperparameters, cv=5, verbose=0,
#  n_jobs=-1)
# best_SVC_model = rnd_search_cv.fit(data_train,target_train)
# print(best_SVC_model.best_params_)
# best_SVC_pred = best_SVC_model.predict(data_test)

# print("Best SVC accuracy : ",accuracy_score(target_test, best_SVC_pred, normalize = True))
# print(confusion_matrix(target_test, best_SVC_pred))


###