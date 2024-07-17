- [Analysis](#analysis)
  - [Predicting 3 dot task behaviors response](#predicting-3-dot-task-behaviors-response)
  - [Predicting 1 dot task behaviors response](#predicting-1-dot-task-behaviors-response)
  - [Micro-movement analysis](#micro-movement-analysis)
    - [3 dot](#3-dot)
    - [1 dot](#1-dot)
  - [Action plan diversity](#action-plan-diversity)
    - [3 dot](#3-dot-1)
    - [1 dot](#1-dot-1)


# Analysis
## Predicting 3 dot task behaviors response
[Compute and save AUC and predicted response](src/ana_three_dot_predicting_individual_beh_profile.ipynb)    
📤 [prediction.csv](data/ana_three_dot_predicting_individual_beh_profile/prediction.csv)

[Compute correlation between predicted and true accuracy](src/ana_three_dot_prediction_corr.ipynb)  
📥 [prediction.csv](data/ana_three_dot_predicting_individual_beh_profile/prediction.csv)  
📤 [correlation_sns.pdf](data/ana_three_dot_prediction_corr/correlation_sns.pdf)  
📤 [correlation_h_sns.pdf](data/ana_three_dot_prediction_corr/correlation_h_sns.pdf): Prediction without action dynamic information


## Predicting 1 dot task behaviors response    
[compute prediction and save result](src/ana_one_dot_predicting_individual_beh_profile.ipynb)  
📤 [prediction.csv](data/ana_one_dot_predicting_individual_beh_profile/prediction.csv)  

[Compute correlation between predicted and true accuracy and save figures](src/ana_one_dot_predicting_individual_beh_profile_report.ipynb)  
📥 [prediction.csv](data/ana_one_dot_predicting_individual_beh_profile/prediction.csv)  
📤 [correlation.pdf](data/ana_one_dot_predicting_individual_beh_profile_report/correlation.pdf)
📤 [b1.pdf](data/ana_one_dot_predicting_individual_beh_profile_report/b1.pdf)
📤 [b2.pdf](data/ana_one_dot_predicting_individual_beh_profile_report/b2.pdf)


## Micro-movement analysis
### 3 dot
[Compute and save AUC and predicted response](src/ana_three_dot_predicting_individual_beh_profile_cossmilarity.ipynb)  
📤 [prediction.csv](data/ana_three_dot_predicting_individual_beh_profile_cossmilarity/prediction.csv)

[Compute correlation between predicted and true accuracy](src/ana_three_dot_prediction_corr_cosinesmilarity.ipynb)  
📥[correlation_sns.pdf](data/ana_three_dot_prediction_corr_cosinesimilarity/correlation_sns.pdf)
📤[correlation_h_sns.pdf](data/ana_three_dot_prediction_corr_cosinesimilarity/correlation_h_sns.pdf)

### 1 dot
[compute prediction and save result](src/ana_one_dot_predicting_individual_beh_profile_cossmilarity.ipynb)  
📥[prediction.csv](data/ana_one_dot_predicting_individual_beh_profile_cossmilarity/prediction.csv)  
📤[correlation.pdf](data/ana_one_dot_predicting_individual_beh_profile_cossmilarity_report/correlation.pdf)📤[b1.pdf](data/ana_one_dot_predicting_individual_beh_profile_cossmilarity_report/b1.pdf)📤[b2.pdf](data/ana_one_dot_predicting_individual_beh_profile_cossmilarity_report/b2.pdf)📤[group.pdf](data/ana_one_dot_predicting_individual_beh_profile_cossmilarity_report/group.pdf)


## Action plan diversity 
[Compute dimensionality for both tasks and save file](src/ana_action_plan_dimentionality.ipynb)   
📤[one_dot_dim.csv](data/ana_action_plan_dimentionality/one_dot_dim.csv)📤[three_dot_dim.csv](data/ana_action_plan_dimentionality/three_dot_dim.csv)

### 3 dot
[Statistics and plot](src/ana_action_plan_dimentionality_report_3d.ipynb)  
📤[dimentionality.pdf](data/ana_action_plan_dimentionality_report_3d/dimentionality.pdf)

### 1 dot
[Statistics and plot](src/ana_action_plan_dimentionality_report_1d.ipynb)  
📤[dimentionality.pdf](data/ana_action_plan_dimentionality_report_1d/dimentionality.pdf)