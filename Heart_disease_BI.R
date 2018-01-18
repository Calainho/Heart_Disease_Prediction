# Author: Felipe Calainho - h16feldu@du.se
#### Packages needed to this project
package_list<-c("Metrics","xgboost","kernlab","foreach","doParallel","doSNOW","rgl","ROCR")
lapply(package_list,require,character.only=T)


#### Downloading data
Heart_data<-read.table("http://archive.ics.uci.edu/ml/machine-learning-databases/statlog/heart/heart.dat", header=FALSE)
label<-c("age","sex","cpt","rbp","sc","fbs","rer","mhra","eia","oldpeak","speSTs","nmv","thal","target")
colnames(Heart_data)<-label
#### transform target into binary
Heart_data$target <- (Heart_data$target-1)
#write.table(data,file = "heartdata.dat",col.names = TRUE)

### Dividing data frame into training and testing
train_portion<-floor(nrow(Heart_data)*0.7)
train<-Heart_data[1:train_portion,]
test<-Heart_data[(train_portion+1):(nrow(Heart_data)),]

############ SVM ##############
###### Grid search for best parameter ######
sigma<- seq(0.000000001,1,length.out=100)
c<-seq(1,100,length.out=10)
parameter<-as.data.frame(expand.grid(sigma=sigma,c=c))
performance<-as.data.frame(rep(0.0,nrow(parameter)))
##### Paralel processing ###
##check number of cores
ncl<-detectCores()
## Use that number of cores to do the paralel processing
cl <- makeCluster(ncl)
registerDoParallel(cl)
##### SVM with paralel processing 
acur<-foreach(i=(1:nrow(parameter)),.packages="kernlab", .combine='rbind')%dopar% {
  
  x<-parameter[i,1]
  y<-parameter[i,2]
  
  svm<-ksvm(target~.,data=train,type="C-svc",C=y,kernel="rbfdot"
            ,kpar=list(sigma=x),scale=T,prob.model=TRUE, cross=10)
  
  performance[i,1]<-(1-svm@cross)
}
parameter$performance<-unname(acur)
stopCluster(cl)
##### checking the best parameter
performance<-na.omit(performance)
parameter<-na.omit(parameter)
bestparameter<-parameter[which(parameter$performance==max(parameter$performance))[1],]
##### Ploting grid search
y<-parameter$sigma
x<-parameter$c
z<-parameter$performance
plot3d(x, y, z, col="red", size=3,type = "p",xlab="C", ylab="Sigma",zlab="Accuracy")
############## Testing the best parameter found #########
best_sig<- as.numeric(bestparameter[1])
best_c<-as.numeric(bestparameter[2]) 

### Training SVM
gc()
svmopt<-ksvm(target~.,data=train,type="C-svc",C=best_c,kernel="rbfdot"
             ,kpar=list(sigma=best_sig),scale=T,prob.model=TRUE)
### Testing SVM
final_prediction<-predict(svmopt,test[,1:13],type="probabilities")

### Transform probabilities into class (cutoff) and confusion matrix
pred_svm <-ifelse(final_prediction[,2]>0.50,1,0)
table(pred_svm,test$target)
Comp_table<-cbind(pred_svm,test$target)

### Creating ROC curve for SVM
Rpred<-prediction(final_prediction[,2],test$target)
Rperf<-performance(Rpred, 'tpr','fpr')
AUC_SVM<-performance(Rpred,measure ='auc')
AUC_SVM<-AUC_SVM@y.values[[1]]
AUC_text<- paste0("AUC = ",AUC_SVM)
plot(Rperf,main = AUC_text)
abline(a=0,b=1)
###################### xgboost ########################

# we're trying to predict target
outcomeName <- c('target')
# list of features
predictors <- names(Heart_data)[!names(Heart_data) %in% outcomeName]
# take first 70% of the data only
train_portion_cv <- floor(nrow(Heart_data)*0.7)
trainSet <- Heart_data[1:train_portion_cv,]
testSet <- Heart_data[(floor(train_portion_cv/2)+1):train_portion_cv,]

############### cross validation #############
cv <- 10
cvDivider <- floor(nrow(trainSet) / (cv+1))
par_xgb<-c()
smallestError <- 0.99
for (depth in seq(1,10,1)) { 
  for (rounds in seq(1,20,1)) {
    totalError <- c()
    indexCount <- 1
    for (cv in seq(1:cv)) {
      # assign chunk to data test
      dataTestIndex <- c((cv * cvDivider):(cv * cvDivider + cvDivider))
      dataTest <- trainSet[dataTestIndex,]
      # everything else to train
      dataTrain <- trainSet[-dataTestIndex,]
      
      bst <- xgboost(data = as.matrix(dataTrain[,predictors]),
                     label = dataTrain[,outcomeName],
                     max.depth=depth, nround=rounds,
                     objective = "binary:logistic", verbose=0)
      gc()
      predictions <- predict(bst, as.matrix(dataTest[,predictors]), outputmargin=TRUE)
      pred<-prediction(as.numeric(predictions),dataTest[,outcomeName])
      perf<-performance(pred, 'tpr','fpr')
      AUC_xgb<-performance(pred,measure ='auc')
      AUC_xgb<-AUC_xgb@y.values[[1]]
      err<-1-as.numeric(AUC_xgb)
      totalError <- c(totalError, err)
    }
    if (mean(totalError) < smallestError) {
      smallestError = mean(totalError)
      print(paste(depth,rounds,smallestError))
    }
    par_xgb<-rbind(par_xgb,c(depth,rounds,(1-mean(totalError))))
  }
} 

##### Ploting CV parameter optimization #####
label2<-c("depth","rounds","accuracy")
colnames(par_xgb)<-label2
par_xgb<-as.data.frame(par_xgb)

y<-par_xgb$depth
x<-par_xgb$rounds
z<-par_xgb$accuracy
plot3d(x, y, z, col="red", size=3,type = "p",xlab="rounds", ylab="depth",zlab="Accuracy")
max(par_xgb$accuracy)
#############################################
###### testing parameters on complete data set

xgb_test <- xgboost(data = as.matrix(train[,predictors]),
                    label = train[,outcomeName],
                    max.depth=1, nround=17, objective = "binary:logistic", verbose=0)
pred_xgb <- predict(xgb_test, as.matrix(test[,predictors]), outputmargin=TRUE)
pred<-prediction(as.numeric(pred_xgb),test[,outcomeName])
perf<-performance(pred, 'tpr','fpr')
AUC_xgb<-performance(pred,measure ='auc')
AUC_xgb<-AUC_xgb@y.values[[1]]
AUC_text<- paste0("AUC = ",AUC_xgb)
plot(perf, main= AUC_text)
abline(a=0,b=1)

### Transform probabilities into class (cutoff) and confusion matrix
pred_xgb <-ifelse(pred_xgb>0.50,1,0)
table(pred_xgb,test$target)
Comp_table<-cbind(pred_xgb,test$target)





