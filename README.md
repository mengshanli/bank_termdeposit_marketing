# bank_termdeposit_marketing

## Objective 
Develop an accurate automation forecasting model of identifying who will open a term deposit account based on marketing campaign (Telemarketing) information.  

## The proposed model 
  * Utilize Gradient Boosting Classifier to forecast potential customers. 
  * Adjust classification model threshold to get better results, especially in recall rate.  

## Performance 
  * Precision: 84.73%, Recall: 85.55% 
  * F1 Score: 85.14% 
  * AUC: training data achieves 0.86 and testing data achieves 0.82, which is excellent discrimination (80%-90%).  

## Findings 
  * The proposed classification model has excellent discrimination in identifying potential customer. 
  * Features of potential customer of term deposit account:  
    * Customers who talked longer than 7.1 minutes 
    * People with balance > $1,708 
    * Students and retired people
