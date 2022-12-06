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
